"""Structured application log: persisted to SQLite and streamed to the UI."""

from __future__ import annotations

import asyncio
import json
import logging
import traceback
from typing import Any, Literal

from . import db

Level = Literal["INFO", "WARN", "ERROR"]

# Categories map 1:1 onto the log section of the spec.
CAT_TRACK_START = "track_start"   # 追跡開始
CAT_TRACK_STOP = "track_stop"     # 追跡終了
CAT_PREDICTION = "prediction"     # 投票情報
CAT_INFERENCE = "inference"       # 推論結果
CAT_BET = "bet"                   # 投票結果
CAT_TRANSCRIPT = "transcript"
CAT_SYSTEM = "system"
CAT_ERROR = "error"               # エラーログ

CATEGORIES = [
    CAT_TRACK_START,
    CAT_TRACK_STOP,
    CAT_PREDICTION,
    CAT_INFERENCE,
    CAT_BET,
    CAT_TRANSCRIPT,
    CAT_SYSTEM,
    CAT_ERROR,
]

_stderr = logging.getLogger("twitch_autobet")

_subscribers: set[asyncio.Queue] = set()
_loop: asyncio.AbstractEventLoop | None = None


def bind_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Remember the server loop so worker threads can still fan out events."""
    global _loop
    _loop = loop


def subscribe() -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue(maxsize=500)
    _subscribers.add(q)
    return q


def unsubscribe(q: asyncio.Queue) -> None:
    _subscribers.discard(q)


def _fanout(payload: dict[str, Any]) -> None:
    """Push an event to every live UI stream. Never raises."""

    def deliver() -> None:
        for q in list(_subscribers):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                # A stalled client must not block logging; drop its event.
                pass

    loop = _loop
    if loop is None:
        return
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is loop:
        deliver()
    else:
        loop.call_soon_threadsafe(deliver)


def write(
    category: str,
    message: str,
    *,
    level: Level = "INFO",
    channel: str | None = None,
    detail: Any = None,
) -> int:
    detail_json = (
        json.dumps(detail, ensure_ascii=False, default=str) if detail is not None else None
    )
    ts = db.now()
    log_id = db.execute(
        "INSERT INTO logs (ts, level, category, channel, message, detail)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (ts, level, category, channel, message, detail_json),
    )
    _stderr.log(
        logging.ERROR if level == "ERROR" else logging.WARNING if level == "WARN" else logging.INFO,
        "[%s] %s %s",
        category,
        f"({channel}) " if channel else "",
        message,
    )
    _fanout(
        {
            "type": "log",
            "log": {
                "id": log_id,
                "ts": ts,
                "level": level,
                "category": category,
                "channel": channel,
                "message": message,
                "detail": detail_json,
            },
        }
    )
    return log_id


def info(category: str, message: str, **kw: Any) -> int:
    return write(category, message, level="INFO", **kw)


def warn(category: str, message: str, **kw: Any) -> int:
    return write(category, message, level="WARN", **kw)


def error(message: str, *, channel: str | None = None, exc: BaseException | None = None,
          category: str = CAT_ERROR, detail: Any = None) -> int:
    if exc is not None:
        detail = {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )[-4000:],
            **(detail if isinstance(detail, dict) else {}),
        }
    return write(category, message, level="ERROR", channel=channel, detail=detail)


def notify(event: str, **fields: Any) -> None:
    """Push a non-log UI event (state change, new data point, ...)."""
    _fanout({"type": event, **fields})


def recent(
    limit: int = 300,
    categories: list[str] | None = None,
    channel: str | None = None,
    before_id: int | None = None,
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM logs WHERE 1=1"
    params: list[Any] = []
    if categories:
        sql += f" AND category IN ({','.join('?' * len(categories))})"
        params.extend(categories)
    if channel:
        sql += " AND channel = ?"
        params.append(channel)
    if before_id:
        sql += " AND id < ?"
        params.append(before_id)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(max(1, min(limit, 2000)))
    return db.rows_to_dicts(db.query(sql, params))


def prune(retention_days: int) -> int:
    """Drop logs older than the retention window; returns rows removed."""
    row = db.query_one(
        "SELECT COUNT(*) AS n FROM logs WHERE ts < datetime('now', ?)",
        (f"-{int(retention_days)} days",),
    )
    n = int(row["n"]) if row else 0
    if n:
        db.execute("DELETE FROM logs WHERE ts < datetime('now', ?)",
                   (f"-{int(retention_days)} days",))
    return n
