"""SQLite storage.

One process-wide connection guarded by a lock. Every write here is a handful of
rows, so the simplicity is worth more than connection pooling.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from .paths import DB_FILE, ensure_dirs

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS channels (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    login         TEXT    NOT NULL UNIQUE,
    display_name  TEXT,
    twitch_id     TEXT,
    enabled       INTEGER NOT NULL DEFAULT 1,
    fixed_probs   TEXT,
    manual_info   TEXT    NOT NULL DEFAULT '',
    last_points   INTEGER,
    created_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS point_history (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id INTEGER NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
    points     INTEGER NOT NULL,
    ts         TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_point_history_channel
    ON point_history(channel_id, ts);

CREATE TABLE IF NOT EXISTS predictions (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id         INTEGER NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
    event_id           TEXT    NOT NULL UNIQUE,
    title              TEXT    NOT NULL DEFAULT '',
    status             TEXT    NOT NULL DEFAULT 'ACTIVE',
    window_seconds     INTEGER,
    created_at         TEXT,
    locked_at          TEXT,
    ended_at           TEXT,
    winning_outcome_id TEXT,
    first_seen         TEXT    NOT NULL,
    updated_at         TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_predictions_channel
    ON predictions(channel_id, first_seen DESC);

CREATE TABLE IF NOT EXISTS prediction_outcomes (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    outcome_id    TEXT    NOT NULL,
    title         TEXT    NOT NULL DEFAULT '',
    color         TEXT,
    total_points  INTEGER NOT NULL DEFAULT 0,
    total_users   INTEGER NOT NULL DEFAULT 0,
    position      INTEGER NOT NULL DEFAULT 0,
    UNIQUE(prediction_id, outcome_id)
);

CREATE TABLE IF NOT EXISTS pool_snapshots (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    ts            TEXT    NOT NULL,
    data          TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_pool_snapshots_pred
    ON pool_snapshots(prediction_id, ts);

CREATE TABLE IF NOT EXISTS inferences (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    ts            TEXT    NOT NULL,
    source        TEXT    NOT NULL,
    probs         TEXT    NOT NULL,
    rationale     TEXT    NOT NULL DEFAULT '',
    raw_response  TEXT,
    latency_ms    INTEGER
);

CREATE TABLE IF NOT EXISTS bets (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
    ts            TEXT    NOT NULL,
    outcome_id    TEXT,
    outcome_title TEXT,
    amount        INTEGER NOT NULL DEFAULT 0,
    dry_run       INTEGER NOT NULL DEFAULT 1,
    probability   REAL,
    edge          REAL,
    kelly_stake   REAL,
    bankroll      INTEGER,
    status        TEXT    NOT NULL,
    error         TEXT,
    result        TEXT,
    payout        INTEGER
);
CREATE INDEX IF NOT EXISTS ix_bets_pred ON bets(prediction_id);

CREATE TABLE IF NOT EXISTS transcripts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id INTEGER NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
    ts         TEXT    NOT NULL,
    text       TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_transcripts_channel
    ON transcripts(channel_id, ts);

CREATE TABLE IF NOT EXISTS logs (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       TEXT NOT NULL,
    level    TEXT NOT NULL,
    category TEXT NOT NULL,
    channel  TEXT,
    message  TEXT NOT NULL,
    detail   TEXT
);
CREATE INDEX IF NOT EXISTS ix_logs_ts ON logs(id DESC);
"""


TS_FORMAT = "%Y-%m-%d %H:%M:%S"


def now() -> str:
    """Current UTC time in SQLite's own text format.

    Every stored timestamp uses this format so that plain string comparisons
    against `datetime('now', '-30 minutes')` are correct. ISO-8601 with a `T`
    separator would sort *after* SQLite's space-separated output and silently
    break every retention window.
    """
    return datetime.now(UTC).strftime(TS_FORMAT)


def to_ts(value: datetime | None) -> str | None:
    """Convert an aware datetime to the stored UTC text format."""
    if value is None:
        return None
    return value.astimezone(UTC).strftime(TS_FORMAT)


def connect() -> sqlite3.Connection:
    global _conn
    with _lock:
        if _conn is None:
            ensure_dirs()
            _conn = sqlite3.connect(DB_FILE, check_same_thread=False)
            _conn.row_factory = sqlite3.Row
            _conn.execute("PRAGMA journal_mode=WAL")
            _conn.execute("PRAGMA foreign_keys=ON")
            _conn.executescript(SCHEMA)
            _conn.commit()
        return _conn


def query(sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
    conn = connect()
    with _lock:
        return conn.execute(sql, tuple(params)).fetchall()


def query_one(sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
    rows = query(sql, params)
    return rows[0] if rows else None


def execute(sql: str, params: Iterable[Any] = ()) -> int:
    """Run a statement and return lastrowid."""
    conn = connect()
    with _lock:
        cur = conn.execute(sql, tuple(params))
        conn.commit()
        return int(cur.lastrowid or 0)


def executemany(sql: str, seq: Iterable[Iterable[Any]]) -> None:
    conn = connect()
    with _lock:
        conn.executemany(sql, [tuple(p) for p in seq])
        conn.commit()


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def rows_to_dicts(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(r) for r in rows]


def loads(value: str | None, default: Any = None) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default
