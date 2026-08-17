"""The tracking orchestrator.

One `Tracker` owns everything that runs while tracking is on:

* a poll loop per enabled channel (`ポーリングレート`), which is the authoritative
  source of prediction and balance state;
* a single PubSub socket that pushes prediction events the instant they happen,
  so a 60-second window is not half over before we notice it;
* the llama-server process and the transcription workers;
* one inference task per open prediction, started the moment it is detected;
* one timer per open prediction that fires `bet_lead_sec` before it locks.

Inference and betting are deliberately split. What the model estimates is which
outcome happens, which does not depend on the pool, so it runs at detection and
gets the whole window to finish. What needs a fresh pool is the Kelly sizing,
and that is arithmetic. Doing it the other way round -- model first, inside the
lead window -- would force `bet_lead_sec` up to 30-45 seconds on any model worth
running locally, and by then the pool is too young to be worth re-reading.

Both event sources funnel into `_observe()`, so it is written to be idempotent:
the same event arriving twice must not produce two bets or two inferences.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from . import config, log, store
from .betting import OutcomeInput, decide, outcome_metrics
from .llm.client import LlamaClient, LlamaError
from .llm.prompt import PromptContext
from .llm.server import LlamaServer
from .search import SearchHit, search
from .transcribe import TranscriptionManager
from .twitch.gql import TwitchError, TwitchGQLClient
from .twitch.models import PredictionEvent, StreamInfo
from .twitch.pubsub import PubSubListener


@dataclass
class ChannelRuntime:
    channel_id: int
    login: str
    display_name: str
    twitch_id: str
    task: asyncio.Task | None = None
    balance: int | None = None
    last_poll: str | None = None
    last_error: str | None = None
    active_event_id: str | None = None
    next_bet_at: str | None = None
    poll_failures: int = 0


@dataclass
class TrackerStatus:
    running: bool = False
    started_at: str | None = None
    stopping: bool = False
    last_error: str | None = None
    channels: dict[int, dict[str, Any]] = field(default_factory=dict)


class Tracker:
    def __init__(self) -> None:
        self.llama = LlamaServer()
        self.transcription = TranscriptionManager()
        self._running = False
        self._started_at: str | None = None
        self._last_error: str | None = None
        self._channels: dict[int, ChannelRuntime] = {}
        self._bet_tasks: dict[str, asyncio.Task] = {}
        self._bet_guard: set[str] = set()
        self._infer_tasks: dict[str, asyncio.Task] = {}
        self._infer_results: dict[str, tuple[dict[str, float], dict[str, Any]]] = {}
        self._twitch: TwitchGQLClient | None = None
        self._pubsub: PubSubListener | None = None
        self._housekeeping: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    # -- public state ------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._running

    def status(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "started_at": self._started_at,
            "last_error": self._last_error,
            "dry_run": config.load().betting.dry_run,
            "llama": self.llama.status,
            "pubsub": self._pubsub.status if self._pubsub else {"connected": False},
            "transcription": self.transcription.status(),
            "channels": {
                str(cid): {
                    "login": rt.login,
                    "display_name": rt.display_name,
                    "balance": rt.balance,
                    "last_poll": rt.last_poll,
                    "last_error": rt.last_error,
                    "active_event_id": rt.active_event_id,
                    "next_bet_at": rt.next_bet_at,
                }
                for cid, rt in self._channels.items()
            },
        }

    def client(self) -> TwitchGQLClient:
        """A Twitch client bound to the current settings, even while stopped."""
        settings = config.load().twitch
        if self._twitch is None:
            self._twitch = TwitchGQLClient(settings)
        else:
            self._twitch.settings = settings
        return self._twitch

    # -- start / stop ------------------------------------------------------

    async def start(self) -> None:
        async with self._lock:
            if self._running:
                return
            settings = config.load()
            self._last_error = None

            if not settings.twitch.oauth_token.strip():
                raise RuntimeError(
                    "Twitch の OAuth トークンが未設定です。設定画面で入力してください"
                )

            channels = [c for c in store.list_channels() if c["enabled"]]
            if not channels:
                raise RuntimeError("追跡が有効なチャンネルがありません")

            needs_llm = any(not _fixed_probs_enabled(c) for c in channels)
            if needs_llm:
                problems = self.llama.validate(settings.llama)
                if problems:
                    raise RuntimeError("; ".join(problems))
                await self.llama.start(settings.llama)

            self._twitch = TwitchGQLClient(settings.twitch)
            self._pubsub = PubSubListener(settings.twitch, self._on_pubsub_event)

            self._channels = {}
            for row in channels:
                rt = ChannelRuntime(
                    channel_id=int(row["id"]),
                    login=str(row["login"]),
                    display_name=str(row["display_name"] or row["login"]),
                    twitch_id=str(row["twitch_id"] or ""),
                    balance=row.get("last_points"),
                )
                self._channels[rt.channel_id] = rt

            self._running = True
            self._started_at = _now_text()

            for rt in self._channels.values():
                rt.task = asyncio.create_task(
                    self._poll_channel(rt), name=f"poll-{rt.login}"
                )

            if settings.twitch.use_pubsub:
                self._pubsub.set_channels(
                    [rt.twitch_id for rt in self._channels.values() if rt.twitch_id]
                )
                await self._pubsub.start()

            await self.transcription.sync(
                [
                    {"id": rt.channel_id, "login": rt.login}
                    for rt in self._channels.values()
                ],
                settings.transcription,
            )

            self._housekeeping = asyncio.create_task(self._housekeep(), name="housekeeping")

            log.info(
                log.CAT_TRACK_START,
                f"追跡を開始しました ({len(self._channels)} チャンネル)"
                + ("" if not settings.betting.dry_run else " / ドライラン"),
                detail={
                    "channels": [rt.login for rt in self._channels.values()],
                    "dry_run": settings.betting.dry_run,
                    "llm": needs_llm,
                },
            )
            log.notify("tracking", running=True)

    async def stop(self) -> None:
        async with self._lock:
            if not self._running:
                return
            self._running = False

            for tasks in (self._bet_tasks, self._infer_tasks):
                for event_id, task in list(tasks.items()):
                    task.cancel()
                    tasks.pop(event_id, None)
            self._infer_results.clear()
            for rt in self._channels.values():
                if rt.task is not None:
                    rt.task.cancel()
            for rt in self._channels.values():
                if rt.task is not None:
                    with contextlib.suppress(asyncio.CancelledError):
                        await rt.task
                    rt.task = None

            if self._housekeeping is not None:
                self._housekeeping.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._housekeeping
                self._housekeeping = None

            if self._pubsub is not None:
                await self._pubsub.stop()
            await self.transcription.stop_all()
            await self.llama.stop()
            if self._twitch is not None:
                await self._twitch.aclose()

            channels = [rt.login for rt in self._channels.values()]
            self._channels = {}
            self._started_at = None

            log.info(
                log.CAT_TRACK_STOP,
                f"追跡を停止しました ({len(channels)} チャンネル)",
                detail={"channels": channels},
            )
            log.notify("tracking", running=False)

    # -- polling -----------------------------------------------------------

    async def _poll_channel(self, rt: ChannelRuntime) -> None:
        settings = config.load()
        last_points_sample = 0.0
        loop = asyncio.get_running_loop()
        log.info(log.CAT_TRACK_START, "チャンネルの追跡を開始しました", channel=rt.login)

        while self._running:
            settings = config.load()
            try:
                balance, events, soft = await self.client().fetch_state(rt.login, count=3)
                rt.last_poll = _now_text()
                rt.poll_failures = 0

                for exc in soft:
                    rt.last_error = str(exc)
                    log.warn(log.CAT_ERROR, str(exc), channel=rt.login,
                             detail={"code": exc.code, "graphql": exc.detail})
                if not soft:
                    rt.last_error = None

                if balance is not None:
                    rt.balance = balance
                    now = loop.time()
                    if now - last_points_sample >= settings.points_poll_sec:
                        store.record_points(rt.channel_id, balance)
                        last_points_sample = now
                        log.notify("points", channel_id=rt.channel_id, points=balance)

                for event in events:
                    await self._observe(rt, event)

            except asyncio.CancelledError:
                raise
            except TwitchError as exc:
                rt.poll_failures += 1
                rt.last_error = str(exc)
                # Auth failures will not fix themselves; say so once and back off hard.
                level_detail = {"code": exc.code, "detail": exc.detail}
                if exc.code == "UNAUTHORIZED" or rt.poll_failures in (1, 5, 20):
                    log.error(f"ポーリングに失敗しました: {exc}", channel=rt.login,
                              detail=level_detail)
                if exc.code in ("UNAUTHORIZED", "RATE_LIMIT"):
                    await asyncio.sleep(min(60.0, 5.0 * rt.poll_failures))
            except Exception as exc:  # noqa: BLE001 - the loop must survive
                rt.poll_failures += 1
                rt.last_error = f"{type(exc).__name__}: {exc}"
                if rt.poll_failures in (1, 5, 20):
                    log.error("ポーリング中に例外が発生しました", channel=rt.login, exc=exc)

            delay = settings.poll_rate_sec
            if rt.poll_failures:
                delay = min(60.0, delay * (1.5 ** min(rt.poll_failures, 6)))
            await asyncio.sleep(delay)

        log.info(log.CAT_TRACK_STOP, "チャンネルの追跡を停止しました", channel=rt.login)

    async def _on_pubsub_event(self, event: PredictionEvent) -> None:
        rt = next(
            (r for r in self._channels.values() if r.twitch_id == event.channel_twitch_id),
            None,
        )
        if rt is None:
            return
        await self._observe(rt, event, source="pubsub")

    # -- event handling ----------------------------------------------------

    async def _observe(
        self, rt: ChannelRuntime, event: PredictionEvent, *, source: str = "poll"
    ) -> None:
        """Record an event and react to state changes. Safe to call repeatedly."""
        if not event.event_id:
            return
        known = store.prediction_by_event(event.event_id)
        prediction_id = store.upsert_prediction(rt.channel_id, event)
        store.record_pool_snapshot(prediction_id, event)

        if known is None:
            total, implied = outcome_metrics(
                [
                    OutcomeInput(o.outcome_id, o.title, o.total_points)
                    for o in event.outcomes
                ]
            )
            log.info(
                log.CAT_PREDICTION,
                f"新しい予想を検出しました: {event.title}",
                channel=rt.login,
                detail={
                    "event_id": event.event_id,
                    "status": event.status,
                    "source": source,
                    "window_seconds": event.window_seconds,
                    "outcomes": [
                        {
                            "title": o.title,
                            "points": o.total_points,
                            "users": o.total_users,
                            "implied": round(implied.get(o.outcome_id, 0.0), 4),
                        }
                        for o in event.outcomes
                    ],
                },
            )
            log.notify("prediction", channel_id=rt.channel_id, event_id=event.event_id)

        previous_status = str(known["status"]).upper() if known else None

        if event.is_open:
            rt.active_event_id = event.event_id
            self._schedule_inference(rt, event, prediction_id)
            self._schedule_bet(rt, event, prediction_id)
        else:
            if rt.active_event_id == event.event_id:
                rt.active_event_id = None
                rt.next_bet_at = None
            for tasks in (self._bet_tasks, self._infer_tasks):
                task = tasks.pop(event.event_id, None)
                if task is not None and not task.done():
                    task.cancel()
            self._infer_results.pop(event.event_id, None)

        # Settle once, on the transition into a final state. `settle_bets` is
        # idempotent, but `_settle` also logs and samples points, which is not.
        if event.is_final and previous_status != event.status:
            await self._settle(rt, event, prediction_id)

    def _schedule_inference(
        self, rt: ChannelRuntime, event: PredictionEvent, prediction_id: int
    ) -> None:
        """Start estimating probabilities as soon as the prediction is seen."""
        event_id = event.event_id
        if event_id in self._infer_tasks or event_id in self._infer_results:
            return
        channel = store.get_channel(rt.channel_id)
        if channel is None:
            return

        if _fixed_probs_enabled(channel):
            if _fixed_probs_usable(channel, event):
                return  # resolved at bet time; no model involved
            log.warn(
                log.CAT_INFERENCE,
                f"固定確率の項目数 ({len(channel['fixed_probs'].get('probs') or [])}) が"
                f"選択肢数 ({len(event.outcomes)}) と一致しないため LLM 推論に切り替えます",
                channel=rt.login,
                detail={"event_id": event_id},
            )

        if not self.llama.running:
            log.error(
                "llama-server が起動していないため推論できません",
                channel=rt.login, category=log.CAT_INFERENCE,
                detail={"event_id": event_id},
            )
            return

        self._infer_tasks[event_id] = asyncio.create_task(
            self._run_inference(rt, event, prediction_id),
            name=f"infer-{event_id[:8]}",
        )

    async def _enrich(
        self,
        rt: ChannelRuntime,
        event: PredictionEvent,
        settings: Any,
        channel: dict,
        transcript: str,
    ) -> tuple[StreamInfo, list[SearchHit]]:
        """Stream context and web search for the prompt.

        Both are enrichments: a failure here costs the model some context but
        must never cost us the prediction, so everything is swallowed.
        """
        stream = StreamInfo()
        try:
            stream = await self.client().fetch_stream_info(rt.login)
        except Exception as exc:  # noqa: BLE001
            log.warn(
                log.CAT_INFERENCE,
                f"配信情報を取得できませんでした ({type(exc).__name__})",
                channel=rt.login,
                detail={"event_id": event.event_id},
            )

        if not settings.search.enabled:
            return stream, []

        ctx = PromptContext(
            channel_login=rt.login,
            channel_display=rt.display_name,
            event=event,
            manual_info=str(channel.get("manual_info") or ""),
            transcript=transcript,
            stream_title=stream.title,
            game_name=stream.game,
        )
        client = LlamaClient(settings.llama, self.llama.base_url)
        try:
            suggestion = await client.suggest_query(
                ctx, settings.search.query_transcript_chars
            )
        except LlamaError as exc:
            log.warn(
                log.CAT_INFERENCE,
                f"検索クエリの生成に失敗しました: {exc} — 検索なしで推論します",
                channel=rt.login,
                detail={"event_id": event.event_id},
            )
            return stream, []

        if not suggestion.query:
            # The model was told to answer empty rather than guess. Searching a
            # topic it could not identify would only feed the prompt noise.
            log.info(
                log.CAT_INFERENCE,
                "話題を特定できなかったため検索を省略します",
                channel=rt.login,
                detail={"event_id": event.event_id, "topic": suggestion.topic},
            )
            return stream, []

        hits = await search(suggestion.query, settings.search)
        log.info(
            log.CAT_INFERENCE,
            f"検索しました: {len(hits)} 件 — {suggestion.topic}",
            channel=rt.login,
            detail={"event_id": event.event_id, "query": suggestion.query,
                    "topic": suggestion.topic,
                    "latency_ms": suggestion.latency_ms,
                    "results": [{"title": h.title, "url": h.url} for h in hits]},
        )
        return stream, hits

    async def _run_inference(
        self, rt: ChannelRuntime, event: PredictionEvent, prediction_id: int
    ) -> None:
        """Task wrapper. Nothing awaits this task until the bet fires, so an
        escaping exception would otherwise vanish until then."""
        try:
            await self._infer(rt, event, prediction_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - the task must not die silently
            log.error("推論処理に失敗しました", channel=rt.login, exc=exc,
                      detail={"event_id": event.event_id})

    async def _infer(
        self, rt: ChannelRuntime, event: PredictionEvent, prediction_id: int
    ) -> None:
        settings = config.load()
        channel = store.get_channel(rt.channel_id)
        if channel is None:
            return
        transcript = store.recent_transcript_text(
            rt.channel_id,
            settings.transcription.retention_min,
            settings.transcription.prompt_chars,
        )
        stream, hits = await self._enrich(rt, event, settings, channel, transcript)
        ctx = PromptContext(
            channel_login=rt.login,
            channel_display=rt.display_name,
            event=event,
            manual_info=str(channel.get("manual_info") or ""),
            history=store.resolved_history_for_prompt(
                rt.channel_id, settings.llama.history_limit
            ),
            transcript=transcript,
            seconds_until_lock=event.seconds_until_lock(),
            stream_title=stream.title,
            game_name=stream.game,
            search_results=[
                {"title": h.title, "description": h.description} for h in hits
            ],
        )
        client = LlamaClient(settings.llama, self.llama.base_url)
        try:
            result = await client.infer(ctx)
        except asyncio.CancelledError:
            raise
        except LlamaError as exc:
            # The task stays in `_infer_tasks` so a failure is not retried on
            # every poll for the rest of the window.
            log.error(f"LLM 推論に失敗しました: {exc}", channel=rt.login,
                      category=log.CAT_INFERENCE,
                      detail={"event_id": event.event_id})
            return

        inference = {
            "rationale": result.rationale,
            "raw_response": result.raw_response,
            "latency_ms": result.latency_ms,
            "warnings": result.warnings,
        }
        self._infer_results[event.event_id] = (result.probabilities, inference)
        self._record_probabilities(
            rt, event, prediction_id, result.probabilities, "llm", inference
        )

    def _schedule_bet(
        self, rt: ChannelRuntime, event: PredictionEvent, prediction_id: int
    ) -> None:
        if event.event_id in self._bet_tasks or event.event_id in self._bet_guard:
            return
        if store.has_bet(prediction_id):
            return

        settings = config.load()
        delay = event.seconds_until_lock()
        if delay is None:
            log.warn(
                log.CAT_PREDICTION,
                "締め切り時刻が取得できないため自動投票を見送ります",
                channel=rt.login,
                detail={"event_id": event.event_id},
            )
            return

        fire_in = delay - settings.betting.bet_lead_sec
        if fire_in < -2.0:
            # Already inside the lead window (e.g. we just started up mid-event).
            return
        fire_in = max(0.0, fire_in)
        rt.next_bet_at = _now_text(offset=fire_in)
        self._bet_tasks[event.event_id] = asyncio.create_task(
            self._bet_after(rt, event.event_id, fire_in), name=f"bet-{event.event_id[:8]}"
        )

    async def _bet_after(self, rt: ChannelRuntime, event_id: str, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
            await self._execute_bet(rt, event_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.error("自動投票の処理に失敗しました", channel=rt.login, exc=exc,
                      detail={"event_id": event_id})
        finally:
            self._bet_tasks.pop(event_id, None)
            rt.next_bet_at = None

    async def _execute_bet(self, rt: ChannelRuntime, event_id: str) -> None:
        if event_id in self._bet_guard:
            return
        self._bet_guard.add(event_id)
        try:
            await self._execute_bet_inner(rt, event_id)
        finally:
            self._bet_guard.discard(event_id)

    async def _execute_bet_inner(self, rt: ChannelRuntime, event_id: str) -> None:
        settings = config.load()
        channel = store.get_channel(rt.channel_id)
        if channel is None:
            return

        # 締め切り五秒前に投票データを取得: re-read rather than trusting the
        # pool we saw when the event opened, since most money arrives late.
        balance, events, soft = await self.client().fetch_state(rt.login, count=5)
        for exc in soft:
            log.warn(log.CAT_ERROR, str(exc), channel=rt.login,
                     detail={"code": exc.code})

        event = next((e for e in events if e.event_id == event_id), None)
        if event is None:
            log.warn(log.CAT_BET, "投票直前に予想が取得できませんでした",
                     channel=rt.login, detail={"event_id": event_id})
            return
        if not event.is_open:
            log.info(log.CAT_BET, f"予想が既に締め切られていました ({event.status})",
                     channel=rt.login, detail={"event_id": event_id})
            return

        prediction_id = store.upsert_prediction(rt.channel_id, event)
        store.record_pool_snapshot(prediction_id, event)
        if store.has_bet(prediction_id):
            return

        if balance is not None:
            rt.balance = balance
        bankroll = int(rt.balance or 0)

        # -- probabilities: the fixed setting wins, otherwise the estimate made
        # when the prediction opened. No model call happens here. --
        probs = self._resolve_probabilities(rt, channel, event, prediction_id)
        if probs is None:
            return

        # -- stake sizing --
        decision = decide(
            [OutcomeInput(o.outcome_id, o.title, o.total_points) for o in event.outcomes],
            probs,
            bankroll,
            settings.betting,
        )
        candidates = [
            {
                "title": c.title,
                "probability": round(c.probability, 4),
                "market": round(c.market_probability, 4),
                "stake": c.stake,
                "edge": round(c.edge, 4),
                "net_odds": round(c.net_odds, 4),
            }
            for c in decision.candidates
        ]

        if not decision.should_bet:
            store.record_bet(
                prediction_id,
                outcome_id=None, outcome_title=None, amount=0,
                dry_run=1 if settings.betting.dry_run else 0,
                probability=None, edge=decision.edge, kelly_stake=decision.kelly_stake,
                bankroll=bankroll, status="skipped", error=decision.reason,
            )
            log.info(
                log.CAT_BET,
                f"投票を見送りました: {decision.reason}",
                channel=rt.login,
                detail={"event_id": event_id, "title": event.title,
                        "bankroll": bankroll, "candidates": candidates},
            )
            log.notify("bet", channel_id=rt.channel_id, event_id=event_id)
            return

        dry = settings.betting.dry_run
        status = "dry_run" if dry else "placed"
        error: str | None = None
        if not dry:
            try:
                await self.client().make_prediction(
                    event_id, decision.outcome_id or "", decision.amount
                )
            except TwitchError as exc:
                status, error = "failed", str(exc)

        bet_id = store.record_bet(
            prediction_id,
            outcome_id=decision.outcome_id,
            outcome_title=decision.outcome_title,
            amount=decision.amount,
            dry_run=1 if dry else 0,
            probability=decision.probability,
            edge=decision.edge,
            kelly_stake=decision.kelly_stake,
            bankroll=bankroll,
            status=status,
            error=error,
        )

        detail = {
            "event_id": event_id,
            "bet_id": bet_id,
            "title": event.title,
            "outcome": decision.outcome_title,
            "amount": decision.amount,
            "bankroll": bankroll,
            "probability": round(decision.probability or 0.0, 4),
            "edge": round(decision.edge or 0.0, 4),
            "full_kelly": round(decision.kelly_stake or 0.0, 1),
            "kelly_fraction": settings.betting.kelly_fraction,
            "candidates": candidates,
        }
        if status == "failed":
            log.error(f"投票に失敗しました: {error}", channel=rt.login, detail=detail)
        else:
            prefix = "[ドライラン] " if dry else ""
            log.info(
                log.CAT_BET,
                f"{prefix}{decision.outcome_title} に {decision.amount:,} pt 投票しました"
                f" ({decision.reason})",
                channel=rt.login,
                detail=detail,
            )
        log.notify("bet", channel_id=rt.channel_id, event_id=event_id)

        # 自動投票後にチャンネルポイントを取得
        await self._sample_points(rt)

    def _resolve_probabilities(
        self,
        rt: ChannelRuntime,
        channel: dict,
        event: PredictionEvent,
        prediction_id: int,
    ) -> dict[str, float] | None:
        """Probabilities for the bet about to be placed.

        Deliberately synchronous: by the time this runs there are only
        `bet_lead_sec` seconds left, so anything slow belongs in
        `_run_inference`, which started when the prediction opened.
        """
        if _fixed_probs_usable(channel, event):
            values = list(channel["fixed_probs"].get("probs") or [])
            total = sum(max(0.0, float(v)) for v in values) or 1.0
            probs = {
                o.outcome_id: max(0.0, float(v)) / total
                for o, v in zip(event.outcomes, values, strict=True)
            }
            self._record_probabilities(
                rt, event, prediction_id, probs, "fixed",
                {"rationale": "固定確率設定を使用"},
            )
            return probs

        cached = self._infer_results.get(event.event_id)
        if cached is not None:
            return cached[0]

        task = self._infer_tasks.get(event.event_id)
        if task is not None and not task.done():
            task.cancel()
            log.warn(
                log.CAT_INFERENCE,
                "締め切りまでに推論が終わらなかったため投票を見送ります",
                channel=rt.login,
                detail={"event_id": event.event_id},
            )
        else:
            log.error(
                "推論結果がないため投票できません",
                channel=rt.login, category=log.CAT_INFERENCE,
                detail={"event_id": event.event_id},
            )
        return None

    def _record_probabilities(
        self,
        rt: ChannelRuntime,
        event: PredictionEvent,
        prediction_id: int,
        probs: dict[str, float],
        source: str,
        inference: dict[str, Any],
    ) -> None:
        store.record_inference(
            prediction_id,
            source,
            probs,
            rationale=inference.get("rationale", ""),
            raw_response=inference.get("raw_response"),
            latency_ms=inference.get("latency_ms"),
        )
        log.info(
            log.CAT_INFERENCE,
            f"推論結果 ({'固定確率' if source == 'fixed' else 'LLM'}): {event.title}",
            channel=rt.login,
            detail={
                "event_id": event.event_id,
                "probabilities": [
                    {"title": o.title, "probability": round(probs.get(o.outcome_id, 0.0), 4)}
                    for o in event.outcomes
                ],
                "rationale": inference.get("rationale", ""),
                "latency_ms": inference.get("latency_ms"),
                "warnings": inference.get("warnings", []),
            },
        )

    async def _settle(
        self, rt: ChannelRuntime, event: PredictionEvent, prediction_id: int
    ) -> None:
        settled = store.settle_bets(prediction_id, event.winning_outcome_id, event.status)
        winner = event.outcome(event.winning_outcome_id or "")
        for bet in settled:
            net = int(bet.get("payout") or 0) - int(bet.get("amount") or 0)
            label = {"win": "的中", "lose": "外れ", "refund": "返還"}.get(
                str(bet.get("result")), str(bet.get("result"))
            )
            prefix = "[ドライラン] " if bet.get("dry_run") else ""
            log.info(
                log.CAT_BET,
                f"{prefix}投票結果: {label} / 損益 {net:+,} pt ({event.title})",
                channel=rt.login,
                detail={
                    "event_id": event.event_id,
                    "status": event.status,
                    "winning_outcome": winner.title if winner else None,
                    "bet_outcome": bet.get("outcome_title"),
                    "amount": bet.get("amount"),
                    "payout": bet.get("payout"),
                    "net": net,
                },
            )
        if settled:
            log.notify("bet", channel_id=rt.channel_id, event_id=event.event_id)
        await self._sample_points(rt)

    async def _sample_points(self, rt: ChannelRuntime) -> None:
        try:
            balance, _events, _soft = await self.client().fetch_state(rt.login, count=1)
        except TwitchError:
            return
        if balance is None:
            return
        rt.balance = balance
        store.record_points(rt.channel_id, balance)
        log.notify("points", channel_id=rt.channel_id, points=balance)

    # -- housekeeping ------------------------------------------------------

    async def _housekeep(self) -> None:
        while self._running:
            try:
                settings = config.load()
                store.prune_transcripts(settings.transcription.retention_min)
                log.prune(settings.log_retention_days)
            except Exception as exc:  # noqa: BLE001
                log.error("メンテナンス処理に失敗しました", exc=exc)
            await asyncio.sleep(60.0)


def _fixed_probs_enabled(channel: dict) -> bool:
    fixed = channel.get("fixed_probs")
    return bool(isinstance(fixed, dict) and fixed.get("enabled") and fixed.get("probs"))


def _fixed_probs_usable(channel: dict, event: PredictionEvent) -> bool:
    """Fixed probabilities only apply to predictions with a matching outcome count."""
    if not _fixed_probs_enabled(channel):
        return False
    return len(list(channel["fixed_probs"].get("probs") or [])) == len(event.outcomes)


def _now_text(offset: float = 0.0) -> str:
    from datetime import timedelta

    return (datetime.now(UTC) + timedelta(seconds=offset)).strftime("%Y-%m-%d %H:%M:%S")


tracker = Tracker()
