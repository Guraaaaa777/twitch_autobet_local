"""End-to-end smoke test with a stubbed Twitch API and a stubbed LLM.

Runs the real tracker code paths -- event ingestion, inference, Kelly sizing,
bet placement, settlement, retention -- against fakes, so the logic can be
checked without Twitch credentials or a GGUF model.

    python -m tests.smoke
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="autobet-smoke-"))
os.environ["TWITCH_AUTOBET_DATA"] = str(_TMP)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config, db, log, store  # noqa: E402
from app import tracker as tracker_mod  # noqa: E402
from app.betting import OutcomeInput, decide  # noqa: E402
from app.llm.client import LlamaError, QuerySuggestion, _normalise, _parse  # noqa: E402
from app.llm.prompt import _format_history, build_messages  # noqa: E402
from app.search.client import SearchHit, _clip  # noqa: E402
from app.tracker import ChannelRuntime, Tracker  # noqa: E402
from app.twitch.gql import TwitchError  # noqa: E402
from app.twitch.models import PredictionEvent, StreamInfo  # noqa: E402

PASSED: list[str] = []


def check(name: str, condition: bool, extra: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {extra}")
        raise AssertionError(f"{name} {extra}")


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_event(status: str = "ACTIVE", *, o1: int = 6000, o2: int = 4000,
               winner: str | None = None, lock_in: float = 30.0) -> dict:
    now = datetime.now(UTC)
    return {
        "id": "event-1",
        "channel_id": "999",
        "title": "次のラウンドは勝てる?",
        "status": status,
        "created_at": iso(now - timedelta(seconds=90)),
        "locked_at": iso(now + timedelta(seconds=lock_in)),
        "prediction_window_seconds": 120,
        "winning_outcome_id": winner,
        "outcomes": [
            {"id": "o1", "title": "勝つ", "color": "BLUE",
             "total_points": o1, "total_users": 30},
            {"id": "o2", "title": "負ける", "color": "PINK",
             "total_points": o2, "total_users": 20},
        ],
    }


class StubTwitch:
    """Stands in for TwitchGQLClient."""

    def __init__(self) -> None:
        self.balance = 50_000
        self.event = make_event()
        self.placed: list[tuple[str, str, int]] = []
        self.reject: str | None = None
        self.stream_fails = False
        self.stream = StreamInfo(title="ランクマ耐久", game="Apex Legends")

    async def fetch_state(self, login: str, count: int = 1):
        return self.balance, [PredictionEvent.parse(self.event)], []

    async def fetch_stream_info(self, login: str):
        if self.stream_fails:
            raise TwitchError("配信情報の取得に失敗", code="STREAM")
        return self.stream

    async def make_prediction(self, event_id: str, outcome_id: str, points: int) -> None:
        if self.reject:
            raise TwitchError(f"Twitch が投票を拒否しました ({self.reject})", code=self.reject)
        self.placed.append((event_id, outcome_id, points))
        self.balance -= points


class StubLlamaServer:
    """Stands in for LlamaServer: always up, never spawns anything."""

    running = True
    base_url = "http://stub"


class StubResult:
    def __init__(self, probs: dict[str, float]) -> None:
        self.probabilities = probs
        self.rationale = "スタブ推論"
        self.raw_response = "{}"
        self.latency_ms = 12
        self.warnings: list[str] = []


class StubLlamaClient:
    """Stands in for LlamaClient. Class-level state so the test can inspect it."""

    calls = 0
    probs = {"o1": 0.8, "o2": 0.2}
    delay = 0.0
    last_ctx = None

    query_calls = 0
    last_query_ctx = None
    suggestion = ("Apex Legends のマッチで優勝できるか", "Apex Legends 優勝 確率 チーム数")
    query_raises = False

    def __init__(self, settings, base_url: str) -> None:
        pass

    async def infer(self, ctx):
        StubLlamaClient.calls += 1
        StubLlamaClient.last_ctx = ctx
        if StubLlamaClient.delay:
            await asyncio.sleep(StubLlamaClient.delay)
        return StubResult(dict(StubLlamaClient.probs))

    async def suggest_query(self, ctx, transcript_chars: int):
        StubLlamaClient.query_calls += 1
        StubLlamaClient.last_query_ctx = ctx
        if StubLlamaClient.query_raises:
            raise LlamaError("スタブのクエリ生成失敗")
        topic, query = StubLlamaClient.suggestion
        return QuerySuggestion(topic=topic, query=query, latency_ms=5)


# -- unit-level checks ------------------------------------------------------


def test_kelly() -> None:
    print("\n[ケリー基準]")
    cfg = config.BettingSettings()
    outs = [OutcomeInput("a", "Yes", 5000), OutcomeInput("b", "No", 5000)]

    d = decide(outs, {"a": 0.5, "b": 0.5}, 10_000, cfg)
    check("エッジなしなら投票しない", not d.should_bet)

    d = decide(outs, {"a": 0.7, "b": 0.3}, 10_000, cfg)
    check("優位側に賭ける", d.should_bet and d.outcome_id == "a", str(d.reason))
    check("最大賭け率 5% を超えない", d.amount <= 500, str(d.amount))
    check("フラクショナルケリーが効いている",
          d.amount < (d.kelly_stake or 0), f"{d.amount} vs {d.kelly_stake}")

    # Self-impact: a lopsided pool must not be treated as fixed odds.
    thin = [OutcomeInput("a", "Yes", 10_000), OutcomeInput("b", "No", 100)]
    d = decide(thin, {"a": 0.5, "b": 0.5}, 10_000, cfg)
    naive = (10_100 - 100) / 100
    check("自分の賭け金でオッズが下がることを織り込む",
          d.should_bet and (d.edge or 0) < 0.5 * (1 + naive) - 1,
          f"edge={d.edge}")

    d = decide(outs, {"a": 0.7, "b": 0.3}, 0, cfg)
    check("残高 0 なら投票しない", not d.should_bet)

    strict = config.BettingSettings(min_edge=0.9)
    d = decide(outs, {"a": 0.6, "b": 0.4}, 10_000, strict)
    check("最低エッジ未満なら見送る", not d.should_bet, str(d.reason))


def test_llm_parsing() -> None:
    print("\n[LLM 応答の解釈]")
    raw = '前置き {"probabilities": [{"choice": 1, "probability": 0.7}, ' \
          '{"choice": 2, "probability": 0.3}], "rationale": "根拠"} 後置き'
    parsed, warnings = _parse(raw, 2)
    check("前後に文字があっても JSON を取り出せる", parsed == {1: 0.7, 2: 0.3}, str(parsed))

    probs, warn2 = _normalise({"o1": 3.0, "o2": 1.0}, ["o1", "o2"])
    check("合計が 1 でなければ正規化する", abs(probs["o1"] - 0.75) < 1e-9, str(probs))
    check("正規化を警告として残す", any("正規化" in w for w in warn2))

    parsed, warnings = _parse('{"probabilities": [{"choice": 1, "probability": 0.9}]}', 2)
    probs, _ = _normalise({"o1": 0.9}, ["o1", "o2"])
    check("欠けた選択肢を警告する", any("2" in w for w in warnings), str(warnings))
    check("欠けた選択肢は 0 として正規化", abs(probs["o1"] - 1.0) < 1e-9, str(probs))

    try:
        _parse("これは JSON ではありません", 2)
        check("非 JSON はエラーになる", False)
    except Exception:
        check("非 JSON はエラーになる", True)


# -- integration ------------------------------------------------------------


async def test_flow() -> None:
    print("\n[追跡〜投票〜決着フロー]")
    log.bind_loop(asyncio.get_running_loop())
    config.update({"betting": {"dry_run": True, "kelly_fraction": 0.25,
                               "max_bet_ratio": 0.05, "min_edge": 0.05}})

    channel_id = store.create_channel("teststreamer", "TestStreamer", "999")
    store.update_channel(channel_id, fixed_probs={"enabled": True, "probs": [0.8, 0.2]})

    tracker = Tracker()
    stub = StubTwitch()
    tracker.client = lambda: stub  # type: ignore[method-assign]
    rt = ChannelRuntime(channel_id=channel_id, login="teststreamer",
                        display_name="TestStreamer", twitch_id="999", balance=50_000)
    tracker._channels = {channel_id: rt}
    tracker._running = True

    # 1. Event appears.
    await tracker._observe(rt, PredictionEvent.parse(stub.event))
    prediction = store.prediction_by_event("event-1")
    check("予想が保存される", prediction is not None)
    check("追跡中の予想として保持される", rt.active_event_id == "event-1")
    check("投票データが蓄積される",
          len(db.query("SELECT 1 FROM pool_snapshots")) == 1)
    check("投票情報がログに出る",
          any(r["category"] == "prediction" for r in log.recent(50)))
    check("締め切り前に投票タスクが予約される", "event-1" in tracker._bet_tasks)
    tracker._bet_tasks.pop("event-1").cancel()

    # 2. Bet fires (dry run).
    await tracker._execute_bet(rt, "event-1")
    bets = db.rows_to_dicts(db.query("SELECT * FROM bets"))
    check("投票が 1 件記録される", len(bets) == 1, str(bets))
    check("ドライランでは実際に投票しない", stub.placed == [], str(stub.placed))
    check("ドライランとして記録される", bets[0]["status"] == "dry_run", bets[0]["status"])
    check("優位な選択肢を選ぶ", bets[0]["outcome_id"] == "o1", str(bets[0]))
    check("上限 5% を超えない", bets[0]["amount"] <= 2500, str(bets[0]["amount"]))
    check("固定確率では LLM を呼ばない",
          db.query_one("SELECT source FROM inferences")["source"] == "fixed")
    check("推論結果がログに出る",
          any(r["category"] == "inference" for r in log.recent(50)))
    check("投票結果がログに出る",
          any(r["category"] == "bet" for r in log.recent(50)))
    amount = bets[0]["amount"]

    # 3. Re-running must not double-bet.
    await tracker._execute_bet(rt, "event-1")
    check("同じ予想に二重投票しない",
          len(db.query("SELECT 1 FROM bets")) == 1)

    # 4. Resolution.
    stub.event = make_event("RESOLVED", winner="o1")
    await tracker._observe(rt, PredictionEvent.parse(stub.event))
    bet = db.row_to_dict(db.query_one("SELECT * FROM bets"))
    check("的中として決着する", bet["result"] == "win", str(bet))
    expected = int(round(10_000 * amount / 6000))
    check("パリミュチュアル配当で精算する", bet["payout"] == expected,
          f"{bet['payout']} != {expected}")
    check("決着後にポイントを取得する",
          len(db.query("SELECT 1 FROM point_history")) >= 1)

    # 5. Live betting path.
    print("\n[実投票モード]")
    config.update({"betting": {"dry_run": False}})
    db.execute("DELETE FROM bets")
    db.execute("DELETE FROM predictions")
    stub.event = make_event("ACTIVE")
    stub.event["id"] = "event-2"
    await tracker._observe(rt, PredictionEvent.parse(stub.event))
    tracker._bet_tasks.pop("event-2", asyncio.Future()).cancel()
    await tracker._execute_bet(rt, "event-2")
    check("実投票モードでは Twitch を呼ぶ", len(stub.placed) == 1, str(stub.placed))
    check("送信内容が記録と一致する",
          stub.placed[0][0] == "event-2" and stub.placed[0][1] == "o1",
          str(stub.placed))
    check("placed として記録される",
          db.query_one("SELECT status FROM bets")["status"] == "placed")

    # 6. Twitch rejects the bet.
    db.execute("DELETE FROM bets")
    db.execute("DELETE FROM predictions")
    stub.reject = "NOT_ENOUGH_POINTS"
    stub.event = make_event("ACTIVE")
    stub.event["id"] = "event-3"
    await tracker._observe(rt, PredictionEvent.parse(stub.event))
    tracker._bet_tasks.pop("event-3", asyncio.Future()).cancel()
    await tracker._execute_bet(rt, "event-3")
    row = db.row_to_dict(db.query_one("SELECT * FROM bets"))
    check("拒否は failed として記録される", row["status"] == "failed", str(row))
    check("拒否理由が残る", "NOT_ENOUGH_POINTS" in (row["error"] or ""), str(row["error"]))
    check("拒否がエラーログに出る",
          any(r["level"] == "ERROR" for r in log.recent(20)))
    stub.reject = None

    # 7. Skip when the edge is too thin.
    print("\n[見送り判定]")
    db.execute("DELETE FROM bets")
    db.execute("DELETE FROM predictions")
    store.update_channel(channel_id, fixed_probs={"enabled": True, "probs": [0.6, 0.4]})
    stub.event = make_event("ACTIVE", o1=6000, o2=4000)
    stub.event["id"] = "event-4"
    await tracker._observe(rt, PredictionEvent.parse(stub.event))
    tracker._bet_tasks.pop("event-4", asyncio.Future()).cancel()
    await tracker._execute_bet(rt, "event-4")
    row = db.row_to_dict(db.query_one("SELECT * FROM bets"))
    check("市場と一致する見立てなら見送る", row["status"] == "skipped", str(row))
    check("見送り理由が残る", bool(row["error"]), str(row))


async def test_two_phase() -> None:
    """The model runs at detection; only Kelly sizing runs in the lead window."""
    print("\n[推論と投票の分離]")
    log.bind_loop(asyncio.get_running_loop())
    # A wide cap so the two pools produce genuinely different stakes rather
    # than both clamping to 最大賭け率.
    config.update({"betting": {"dry_run": True, "kelly_fraction": 0.25,
                               "max_bet_ratio": 0.5, "min_edge": 0.05}})
    StubLlamaClient.calls = 0
    StubLlamaClient.delay = 0.0
    tracker_mod.LlamaClient = StubLlamaClient  # type: ignore[misc]

    channel_id = store.create_channel("llmstreamer", "LlmStreamer", "888")
    tracker = Tracker()
    tracker.llama = StubLlamaServer()  # type: ignore[assignment]
    stub = StubTwitch()
    stub.event = make_event("ACTIVE", o1=6000, o2=4000)
    stub.event["id"] = "event-llm"
    stub.event["channel_id"] = "888"
    tracker.client = lambda: stub  # type: ignore[method-assign]
    rt = ChannelRuntime(channel_id=channel_id, login="llmstreamer",
                        display_name="LlmStreamer", twitch_id="888", balance=50_000)
    tracker._channels = {channel_id: rt}
    tracker._running = True
    tracker._llm_enabled = True  # normally set by start(); these drive it directly

    # 1. Detection starts the model immediately.
    await tracker._observe(rt, PredictionEvent.parse(stub.event))
    check("検知した時点で推論が始まる", "event-llm" in tracker._infer_tasks)
    tracker._bet_tasks.pop("event-llm").cancel()
    await tracker._infer_tasks["event-llm"]

    check("推論は 1 回だけ走る", StubLlamaClient.calls == 1, str(StubLlamaClient.calls))
    check("確率がキャッシュされる", "event-llm" in tracker._infer_results)
    pred = store.prediction_by_event("event-llm")
    assert pred
    row = db.query_one("SELECT source FROM inferences WHERE prediction_id = ?", (pred["id"],))
    check("投票を待たずに推論が記録される", row is not None and row["source"] == "llm",
          str(row))

    body = build_messages(StubLlamaClient.last_ctx)[1]["content"]
    check("プロンプトに現在の投票状況を含めない",
          "投票状況" not in body and "6,000 pt" not in body, body[:200])

    # 2. Re-observing an already-inferred event must not re-run the model.
    await tracker._observe(rt, PredictionEvent.parse(stub.event))
    tracker._bet_tasks.pop("event-llm", asyncio.Future()).cancel()
    check("ポーリングのたびに推論し直さない", StubLlamaClient.calls == 1,
          str(StubLlamaClient.calls))

    # 3. The pool moves between inference and the bet. The stake must follow
    #    the pool read in the lead window, not the one the model saw.
    stub.event = make_event("ACTIVE", o1=1000, o2=9000)
    stub.event["id"] = "event-llm"
    stub.event["channel_id"] = "888"
    await tracker._execute_bet(rt, "event-llm")

    check("投票時に LLM を呼び直さない", StubLlamaClient.calls == 1,
          str(StubLlamaClient.calls))
    bet = db.row_to_dict(
        db.query_one("SELECT * FROM bets WHERE prediction_id = ?", (pred["id"],))
    )
    betting = config.load().betting
    fresh = decide([OutcomeInput("o1", "勝つ", 1000), OutcomeInput("o2", "負ける", 9000)],
                   {"o1": 0.8, "o2": 0.2}, 50_000, betting)
    stale = decide([OutcomeInput("o1", "勝つ", 6000), OutcomeInput("o2", "負ける", 4000)],
                   {"o1": 0.8, "o2": 0.2}, 50_000, betting)
    check("締め切り直前のプールで賭け金を決める",
          bet["amount"] == fresh.amount and bet["outcome_id"] == fresh.outcome_id,
          f"{bet['amount']} != {fresh.amount}")
    check("推論時点の古いプールとは結果が変わる", fresh.amount != stale.amount,
          f"{fresh.amount} vs {stale.amount}")

    # 4. Not finished by the deadline: skip rather than bet late.
    print("\n[推論が間に合わない場合]")
    db.execute("DELETE FROM bets")
    StubLlamaClient.delay = 30.0
    stub.event = make_event("ACTIVE")
    stub.event["id"] = "event-slow"
    stub.event["channel_id"] = "888"
    await tracker._observe(rt, PredictionEvent.parse(stub.event))
    tracker._bet_tasks.pop("event-slow").cancel()
    await tracker._execute_bet(rt, "event-slow")

    check("推論が終わっていなければ投票しない",
          db.query_one("SELECT 1 FROM bets") is None)
    check("間に合わなかったことを警告する",
          any("推論が終わらなかった" in (r["message"] or "") for r in log.recent(20)))
    check("間に合わなかった推論は打ち切る",
          tracker._infer_tasks["event-slow"].cancelled()
          or tracker._infer_tasks["event-slow"].cancelling() > 0)
    StubLlamaClient.delay = 0.0

    # 5. History given to the model: options and winner, never the vote split.
    print("\n[過去予想の渡しかた]")
    stub.event = make_event("RESOLVED", winner="o1")
    stub.event["id"] = "event-llm"
    stub.event["channel_id"] = "888"
    await tracker._observe(rt, PredictionEvent.parse(stub.event))
    hist = store.resolved_history_for_prompt(channel_id, 12)
    check("決着した予想が履歴に入る", bool(hist), str(hist))
    check("履歴に当時の投票比率を含めない",
          all("share" not in o for h in hist for o in h["outcomes"]), str(hist))
    check("履歴の書式に比率が出ない", "%" not in _format_history(hist),
          _format_history(hist))
    check("履歴に勝った選択肢は残る",
          all(h["winner_title"] for h in hist), str(hist))


async def test_search() -> None:
    """Search enriches the prompt, and every failure degrades to no search."""
    print("\n[ウェブ検索]")
    log.bind_loop(asyncio.get_running_loop())

    long_hit = SearchHit("t" * 100, "d" * 100, "u")
    check("文字数上限で結果を切る", len(_clip([long_hit] * 10, 500)) == 2,
          str(len(_clip([long_hit] * 10, 500))))

    config.update({"search": {"enabled": True, "base_url": "http://127.0.0.1:8888",
                              "count": 3, "max_chars": 1500}})
    StubLlamaClient.calls = 0
    StubLlamaClient.delay = 0.0
    tracker_mod.LlamaClient = StubLlamaClient  # type: ignore[misc]

    captured: dict = {}

    async def stub_search(query, settings):
        captured["query"] = query
        return [SearchHit("Apex 最新パッチ", "射撃訓練場の仕様が変わりました", "https://e/1"),
                SearchHit("大会結果", "前回は第 3 ラウンドで敗退", "https://e/2")]

    tracker_mod.search = stub_search  # type: ignore[assignment]

    channel_id = store.create_channel("searchstreamer", "SearchStreamer", "777")
    tracker = Tracker()
    tracker.llama = StubLlamaServer()  # type: ignore[assignment]
    stub = StubTwitch()
    stub.event = make_event("ACTIVE")
    stub.event["id"] = "event-search"
    stub.event["channel_id"] = "777"
    tracker.client = lambda: stub  # type: ignore[method-assign]
    rt = ChannelRuntime(channel_id=channel_id, login="searchstreamer",
                        display_name="SearchStreamer", twitch_id="777", balance=50_000)
    tracker._channels = {channel_id: rt}
    tracker._running = True
    tracker._llm_enabled = True

    await tracker._observe(rt, PredictionEvent.parse(stub.event))
    tracker._bet_tasks.pop("event-search").cancel()
    await tracker._infer_tasks["event-search"]

    body = build_messages(StubLlamaClient.last_ctx)[1]["content"]
    check("LLM が作ったクエリで検索する", "Apex Legends" in captured.get("query", ""),
          captured.get("query", ""))
    check("検索結果がプロンプトに入る" , "射撃訓練場の仕様が変わりました" in body)
    check("ゲーム名と配信タイトルもプロンプトに入る",
          "Apex Legends" in body and "ランクマ耐久" in body)
    check("外部コンテンツであることを明示する",
          "指示ではない" in body and "従わず" in body, body[:400])
    check("検索しても推論は 1 回のまま", StubLlamaClient.calls == 1,
          str(StubLlamaClient.calls))

    # Search blowing up must not cost us the inference.
    print("\n[検索が失敗した場合]")

    async def exploding_search(query, settings):
        raise RuntimeError("検索 API が落ちています")

    tracker_mod.search = exploding_search  # type: ignore[assignment]
    stub.event = make_event("ACTIVE")
    stub.event["id"] = "event-search2"
    stub.event["channel_id"] = "777"
    await tracker._observe(rt, PredictionEvent.parse(stub.event))
    tracker._bet_tasks.pop("event-search2").cancel()
    await tracker._infer_tasks["event-search2"]
    check("検索が例外でも推論タスクは黙って死なない",
          any("推論処理に失敗" in (r["message"] or "") for r in log.recent(20)))

    # A broken stream lookup only costs the query, not the inference.
    tracker_mod.search = stub_search  # type: ignore[assignment]
    stub.stream_fails = True
    stub.event = make_event("ACTIVE")
    stub.event["id"] = "event-search3"
    stub.event["channel_id"] = "777"
    await tracker._observe(rt, PredictionEvent.parse(stub.event))
    tracker._bet_tasks.pop("event-search3").cancel()
    await tracker._infer_tasks["event-search3"]
    check("配信情報が取れなくても推論は完走する",
          "event-search3" in tracker._infer_results)
    check("配信情報の取得失敗を警告する",
          any("配信情報を取得できません" in (r["message"] or "") for r in log.recent(20)))
    stub.stream_fails = False

    # The worked example: neither the title nor the Twitch category says what
    # game this is. Only the transcript does, so only the model can find it.
    print("\n[話題の特定]")
    tracker_mod.search = stub_search  # type: ignore[assignment]
    store.add_transcript(channel_id, "今日はAPEXのランクやっていきます")
    stub.stream = StreamInfo(title="ゲームする", game="")
    vague = make_event("ACTIVE")
    vague["id"] = "event-vague"
    vague["channel_id"] = "777"
    vague["title"] = "優勝"
    vague["outcomes"] = [
        {"id": "o1", "title": "する", "total_points": 6000, "total_users": 30},
        {"id": "o2", "title": "しない", "total_points": 4000, "total_users": 20},
    ]
    stub.event = vague
    await tracker._observe(rt, PredictionEvent.parse(vague))
    tracker._bet_tasks.pop("event-vague").cancel()
    await tracker._infer_tasks["event-vague"]

    qctx = StubLlamaClient.last_query_ctx
    check("クエリ生成に文字起こしを渡す", "APEX" in qctx.transcript, qctx.transcript[:120])
    check("クエリ生成に予想タイトルと選択肢を渡す",
          qctx.event.title == "優勝"
          and [o.title for o in qctx.event.outcomes] == ["する", "しない"])
    check("LLM は文字起こしから話題を特定して検索する",
          "Apex Legends" in captured.get("query", ""), captured.get("query", ""))

    # Query generation failing costs the search, never the inference.
    print("\n[クエリ生成が失敗した場合]")
    StubLlamaClient.query_raises = True
    stub.stream = StreamInfo(title="ランクマ耐久", game="Apex Legends")
    captured.clear()
    stub.event = make_event("ACTIVE")
    stub.event["id"] = "event-qfail"
    stub.event["channel_id"] = "777"
    await tracker._observe(rt, PredictionEvent.parse(stub.event))
    tracker._bet_tasks.pop("event-qfail").cancel()
    await tracker._infer_tasks["event-qfail"]
    check("クエリ生成が失敗したら検索しない", "query" not in captured, str(captured))
    check("クエリ生成の失敗を警告する",
          any("検索クエリの生成に失敗" in (r["message"] or "") for r in log.recent(20)))
    check("クエリ生成が失敗しても推論は完走する", "event-qfail" in tracker._infer_results)
    check("検索なしでもプロンプトに検索セクションを作らない",
          "検索結果" not in build_messages(StubLlamaClient.last_ctx)[1]["content"])
    StubLlamaClient.query_raises = False

    # The model is told to answer empty rather than guess; respect that.
    StubLlamaClient.suggestion = ("特定できない", "")
    captured.clear()
    stub.event = make_event("ACTIVE")
    stub.event["id"] = "event-noq"
    stub.event["channel_id"] = "777"
    await tracker._observe(rt, PredictionEvent.parse(stub.event))
    tracker._bet_tasks.pop("event-noq").cancel()
    await tracker._infer_tasks["event-noq"]
    check("話題を特定できなければ検索しない", "query" not in captured, str(captured))
    check("検索を省略したことをログに残す",
          any("話題を特定できなかった" in (r["message"] or "") for r in log.recent(20)))
    check("検索を省略しても推論は完走する", "event-noq" in tracker._infer_results)
    StubLlamaClient.suggestion = ("Apex Legends のマッチで優勝できるか",
                                  "Apex Legends 優勝 確率 チーム数")

    # Leave the DB as we found it: test_retention counts transcripts globally.
    db.execute("DELETE FROM transcripts WHERE channel_id = ?", (channel_id,))
    config.update({"search": {"enabled": False}})


async def test_llm_mode() -> None:
    """LLM の使用 switch: whether llama-server is launched at all."""
    print("\n[LLM の使用切り替え]")
    log.bind_loop(asyncio.get_running_loop())

    fixed = {"fixed_probs": {"enabled": True, "probs": [0.8, 0.2]}}
    plain: dict = {"fixed_probs": None}

    check("自動: 固定確率だけなら起動しない",
          tracker_mod.llm_required("auto", [fixed]) is False)
    check("自動: 固定確率のないチャンネルがあれば起動する",
          tracker_mod.llm_required("auto", [fixed, plain]) is True)
    check("使わない: 固定確率がなくても起動しない",
          tracker_mod.llm_required("never", [plain]) is False)
    check("常に使う: 固定確率だけでも起動する",
          tracker_mod.llm_required("always", [fixed]) is True)

    # The gap the switch had to close: fixed probabilities fall back to the
    # model when the outcome count does not match, and with no model this
    # session that has to be said out loud, not surface as a missing inference
    # five seconds before the lock.
    channel_id = store.create_channel("fixedstreamer", "FixedStreamer", "666")
    store.update_channel(channel_id, fixed_probs={"enabled": True, "probs": [0.8, 0.2]})

    tracker = Tracker()
    stub = StubTwitch()
    stub.event = make_event("ACTIVE")
    stub.event["outcomes"].append(
        {"id": "o3", "title": "引き分け", "color": "BLUE",
         "total_points": 1000, "total_users": 5}
    )
    stub.event["id"] = "event-mismatch"
    stub.event["channel_id"] = "666"
    tracker.client = lambda: stub  # type: ignore[method-assign]
    rt = ChannelRuntime(channel_id=channel_id, login="fixedstreamer",
                        display_name="FixedStreamer", twitch_id="666", balance=50_000)
    tracker._channels = {channel_id: rt}
    tracker._running = True
    tracker._llm_enabled = False

    await tracker._observe(rt, PredictionEvent.parse(stub.event))
    tracker._bet_tasks.pop("event-mismatch", asyncio.Future()).cancel()
    check("LLM 無効なら推論タスクを作らない",
          "event-mismatch" not in tracker._infer_tasks)
    check("項目数不一致を検知時に警告する",
          any("見送ります" in (r["message"] or "") for r in log.recent(20)))

    await tracker._execute_bet(rt, "event-mismatch")
    check("項目数が合わなければ投票しない",
          db.query_one("SELECT 1 FROM bets WHERE prediction_id = "
                       "(SELECT id FROM predictions WHERE event_id = 'event-mismatch')")
          is None)
    check("llama-server 未起動をエラーにはしない",
          not any("llama-server が起動していない" in (r["message"] or "")
                  for r in log.recent(20)))

    db.execute("DELETE FROM channels WHERE id = ?", (channel_id,))


async def test_retention() -> None:
    print("\n[保持時間]")
    channel = store.get_channel_by_login("teststreamer")
    assert channel
    cid = channel["id"]
    store.add_transcript(cid, "新しい発話")
    db.execute(
        "INSERT INTO transcripts (channel_id, ts, text) VALUES (?, datetime('now','-45 minutes'), ?)",
        (cid, "古い発話"),
    )
    text = store.recent_transcript_text(cid, 30, 10_000)
    check("保持時間内の文字起こしだけ渡す", "新しい発話" in text and "古い発話" not in text, text)

    removed = store.prune_transcripts(30)
    check("保持時間を過ぎた行を削除する", removed == 1, str(removed))
    check("新しい行は残る", len(db.query("SELECT 1 FROM transcripts")) == 1)


def test_api() -> None:
    print("\n[HTTP API]")
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        res = client.get("/api/settings")
        check("設定を取得できる", res.status_code == 200 and "llama" in res.json())

        res = client.put("/api/settings", json={"poll_rate_sec": 9.5})
        check("設定を更新できる", res.status_code == 200
              and res.json()["poll_rate_sec"] == 9.5, res.text[:200])

        res = client.put("/api/settings", json={"poll_rate_sec": -1})
        check("不正な設定を拒否する", res.status_code == 400, res.text[:200])

        res = client.put("/api/settings", json={"llama": {"mode": "sometimes"}})
        check("未知の LLM モードを拒否する", res.status_code == 400, res.text[:200])

        client.put("/api/settings", json={"llama": {"mode": "never"}})
        res = client.post("/api/settings/test/llama")
        body = res.json()
        check("使わない設定なら llama のパス未設定を失敗にしない",
              res.status_code == 200 and body["ok"] and not body["command"],
              res.text[:200])
        check("起動しないことを検証結果で伝える", "起動しません" in body.get("note", ""),
              res.text[:200])
        client.put("/api/settings", json={"llama": {"mode": "auto"}})

        res = client.get("/api/channels")
        check("チャンネル一覧を取得できる", res.status_code == 200
              and any(c["login"] == "teststreamer" for c in res.json()))

        cid = next(c["id"] for c in res.json() if c["login"] == "teststreamer")
        res = client.patch(f"/api/channels/{cid}", json={"enabled": False})
        check("追跡除外に切り替えられる", res.status_code == 200
              and res.json()["enabled"] == 0, res.text[:200])

        res = client.patch(f"/api/channels/{cid}",
                           json={"fixed_probs": {"enabled": True, "probs": [1.0]}})
        check("固定確率の項目数不足を拒否する", res.status_code == 400, res.text[:200])

        res = client.patch(f"/api/channels/{cid}", json={"manual_info": "テスト情報"})
        check("手動情報を保存できる",
              res.status_code == 200 and res.json()["manual_info"] == "テスト情報")

        res = client.get(f"/api/channels/{cid}/points")
        check("ポイント推移を取得できる", res.status_code == 200 and "points" in res.json())

        res = client.get(f"/api/channels/{cid}/predictions")
        check("予想履歴を取得できる", res.status_code == 200)

        res = client.get("/api/logs?limit=10")
        check("ログを取得できる", res.status_code == 200 and res.json()["logs"])

        res = client.get("/api/status")
        check("状態を取得できる", res.status_code == 200 and res.json()["running"] is False)

        # The probe must name the problem rather than raise: an unset URL is the
        # state every new install starts in. (Reaching a live SearXNG is not
        # something a test can assume, so only this branch is exercised here.)
        saved = config.load().search.base_url
        config.update({"search": {"base_url": ""}})
        res = client.post("/api/settings/test/search")
        body = res.json()
        check("検索の接続テストは URL 未設定を指摘する",
              res.status_code == 200 and body["ok"] is False
              and any("未設定" in p for p in body["problems"]), res.text[:200])
        config.update({"search": {"base_url": saved}})

        res = client.post("/api/tracking/start")
        check("認証未設定では追跡を開始できない", res.status_code == 400, res.text[:200])

        for path in ("/", "/settings", "/static/app.js", "/static/style.css",
                     "/static/common.js", "/static/settings.js"):
            check(f"{path} を配信できる", client.get(path).status_code == 200)

        res = client.delete(f"/api/channels/{cid}")
        check("登録解除できる", res.status_code == 200)
        check("蓄積データも削除される",
              db.query_one("SELECT 1 FROM predictions WHERE channel_id = ?", (cid,)) is None
              and db.query_one("SELECT 1 FROM transcripts WHERE channel_id = ?", (cid,)) is None)


def main() -> int:
    print(f"data dir: {_TMP}")
    db.connect()
    try:
        test_kelly()
        test_llm_parsing()
        asyncio.run(test_flow())
        asyncio.run(test_two_phase())
        asyncio.run(test_search())
        asyncio.run(test_llm_mode())
        asyncio.run(test_retention())
        test_api()
    except AssertionError as exc:
        print(f"\n失敗: {exc}")
        return 1
    finally:
        db.connect().close()
        shutil.rmtree(_TMP, ignore_errors=True)
    print(f"\n{len(PASSED)} 件すべて成功しました")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
