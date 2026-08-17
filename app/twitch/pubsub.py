"""PubSub listener for `predictions-channel-v1`.

Polling alone would miss the first seconds of a prediction, which matters when a
window is only 60 seconds long. PubSub pushes `event-created` / `event-updated`
the moment they happen, so it runs alongside the poller: PubSub for latency,
polling for the authoritative state right before we bet.

PubSub is a legacy Twitch surface that has been slated for removal more than
once. Everything here is best-effort -- if the socket never connects, the poller
still drives the whole flow.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import random
import secrets
from collections.abc import Awaitable, Callable

import websockets
from websockets.asyncio.client import connect

from .. import log
from ..config import TwitchSettings
from .models import PredictionEvent

PUBSUB_URL = "wss://pubsub-edge.twitch.tv/v1"
PING_INTERVAL = 240.0
PONG_TIMEOUT = 15.0

EventHandler = Callable[[PredictionEvent], Awaitable[None]]


class PubSubListener:
    def __init__(self, settings: TwitchSettings, on_event: EventHandler) -> None:
        self.settings = settings
        self.on_event = on_event
        self._topics: set[str] = set()
        self._task: asyncio.Task | None = None
        self._wake = asyncio.Event()
        self._connected = False
        self._last_error: str | None = None

    @property
    def status(self) -> dict:
        return {
            "connected": self._connected,
            "topics": sorted(self._topics),
            "last_error": self._last_error,
        }

    def set_channels(self, twitch_ids: list[str]) -> None:
        """Replace the subscribed topic set; reconnects if it changed."""
        topics = {f"predictions-channel-v1.{i}" for i in twitch_ids if i}
        if topics != self._topics:
            self._topics = topics
            self._wake.set()

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="pubsub")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self._connected = False

    # -- internals ---------------------------------------------------------

    async def _run(self) -> None:
        backoff = 1.0
        while True:
            if not self._topics or not self.settings.oauth_token:
                self._connected = False
                self._wake.clear()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._wake.wait(), timeout=10.0)
                continue
            try:
                await self._session()
                backoff = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - the socket must keep retrying
                self._connected = False
                self._last_error = f"{type(exc).__name__}: {exc}"
                log.warn(
                    log.CAT_SYSTEM,
                    f"PubSub 接続が切断されました。{backoff:.0f} 秒後に再接続します",
                    detail={"error": self._last_error},
                )
                await asyncio.sleep(backoff + random.uniform(0, 1))
                backoff = min(backoff * 2, 60.0)

    async def _session(self) -> None:
        token = self.settings.oauth_token.strip()
        for prefix in ("OAuth ", "oauth:", "Bearer "):
            if token.lower().startswith(prefix.lower()):
                token = token[len(prefix):].strip()

        async with connect(PUBSUB_URL, open_timeout=20, close_timeout=5) as ws:
            subscribed = set(self._topics)
            await ws.send(
                json.dumps(
                    {
                        "type": "LISTEN",
                        "nonce": secrets.token_hex(8),
                        "data": {"topics": sorted(subscribed), "auth_token": token},
                    }
                )
            )
            self._connected = True
            self._last_error = None
            self._wake.clear()
            log.info(
                log.CAT_SYSTEM,
                f"PubSub に接続しました ({len(subscribed)} トピック)",
            )

            ping_task = asyncio.create_task(self._ping_loop(ws))
            try:
                while True:
                    if self._wake.is_set():
                        # Topic set changed -- drop and reconnect with the new set.
                        return
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    except asyncio.TimeoutError:
                        continue
                    await self._handle(raw)
            finally:
                self._connected = False
                ping_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await ping_task

    async def _ping_loop(self, ws) -> None:
        while True:
            await asyncio.sleep(PING_INTERVAL * random.uniform(0.85, 1.0))
            await ws.send(json.dumps({"type": "PING"}))

    async def _handle(self, raw: str | bytes) -> None:
        try:
            frame = json.loads(raw)
        except (TypeError, ValueError):
            return
        kind = frame.get("type")
        if kind == "RECONNECT":
            raise websockets.ConnectionClosedError(None, None)
        if kind == "RESPONSE" and frame.get("error"):
            self._last_error = str(frame["error"])
            log.warn(
                log.CAT_SYSTEM,
                f"PubSub の購読が拒否されました: {frame['error']}",
                detail=frame,
            )
            return
        if kind != "MESSAGE":
            return

        data = frame.get("data") or {}
        try:
            message = json.loads(data.get("message") or "{}")
        except (TypeError, ValueError):
            return

        if message.get("type") not in ("event-created", "event-updated"):
            return
        payload = (message.get("data") or {}).get("event")
        if not isinstance(payload, dict):
            return

        event = PredictionEvent.parse(payload)
        if not event.channel_twitch_id:
            topic = str(data.get("topic") or "")
            if "." in topic:
                event.channel_twitch_id = topic.rsplit(".", 1)[1]
        if not event.event_id:
            return
        try:
            await self.on_event(event)
        except Exception as exc:  # noqa: BLE001 - one bad event must not kill the socket
            log.error("PubSub イベントの処理に失敗しました", exc=exc)
