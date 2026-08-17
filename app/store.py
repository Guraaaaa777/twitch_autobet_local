"""Persistence helpers shared by the tracker, the API layer, and the workers."""

from __future__ import annotations

import json
from typing import Any

from . import db
from .twitch.models import PredictionEvent

# -- channels --------------------------------------------------------------


def list_channels() -> list[dict[str, Any]]:
    rows = db.rows_to_dicts(db.query("SELECT * FROM channels ORDER BY login"))
    for row in rows:
        row["fixed_probs"] = db.loads(row.get("fixed_probs"))
    return rows


def get_channel(channel_id: int) -> dict[str, Any] | None:
    row = db.row_to_dict(db.query_one("SELECT * FROM channels WHERE id = ?", (channel_id,)))
    if row:
        row["fixed_probs"] = db.loads(row.get("fixed_probs"))
    return row


def get_channel_by_login(login: str) -> dict[str, Any] | None:
    row = db.row_to_dict(
        db.query_one("SELECT * FROM channels WHERE login = ?", (login.strip().lower(),))
    )
    if row:
        row["fixed_probs"] = db.loads(row.get("fixed_probs"))
    return row


def get_channel_by_twitch_id(twitch_id: str) -> dict[str, Any] | None:
    row = db.row_to_dict(
        db.query_one("SELECT * FROM channels WHERE twitch_id = ?", (str(twitch_id),))
    )
    if row:
        row["fixed_probs"] = db.loads(row.get("fixed_probs"))
    return row


def create_channel(login: str, display_name: str, twitch_id: str) -> int:
    return db.execute(
        "INSERT INTO channels (login, display_name, twitch_id, enabled, manual_info, created_at)"
        " VALUES (?, ?, ?, 1, '', ?)",
        (login.strip().lower(), display_name, twitch_id, db.now()),
    )


def update_channel(channel_id: int, **fields: Any) -> None:
    allowed = {
        "display_name", "twitch_id", "enabled", "fixed_probs",
        "manual_info", "last_points",
    }
    sets, params = [], []
    for key, value in fields.items():
        if key not in allowed:
            continue
        if key == "fixed_probs" and not isinstance(value, (str, type(None))):
            value = json.dumps(value, ensure_ascii=False)
        if key == "enabled":
            value = 1 if value else 0
        sets.append(f"{key} = ?")
        params.append(value)
    if not sets:
        return
    params.append(channel_id)
    db.execute(f"UPDATE channels SET {', '.join(sets)} WHERE id = ?", params)


def delete_channel(channel_id: int) -> None:
    """Remove the channel and every row that accumulated for it."""
    db.execute("DELETE FROM channels WHERE id = ?", (channel_id,))


# -- channel points --------------------------------------------------------


def record_points(channel_id: int, points: int) -> None:
    db.execute(
        "INSERT INTO point_history (channel_id, points, ts) VALUES (?, ?, ?)",
        (channel_id, int(points), db.now()),
    )
    db.execute("UPDATE channels SET last_points = ? WHERE id = ?", (int(points), channel_id))


def points_history(channel_id: int, limit: int = 2000) -> list[dict[str, Any]]:
    rows = db.query(
        "SELECT points, ts FROM point_history WHERE channel_id = ?"
        " ORDER BY id DESC LIMIT ?",
        (channel_id, limit),
    )
    return list(reversed(db.rows_to_dicts(rows)))


# -- predictions -----------------------------------------------------------


def upsert_prediction(channel_id: int, event: PredictionEvent) -> int:
    """Insert or refresh a prediction row and its outcomes. Returns its row id."""
    existing = db.query_one(
        "SELECT id FROM predictions WHERE event_id = ?", (event.event_id,)
    )
    now = db.now()
    iso = db.to_ts

    if existing is None:
        prediction_id = db.execute(
            "INSERT INTO predictions (channel_id, event_id, title, status, window_seconds,"
            " created_at, locked_at, ended_at, winning_outcome_id, first_seen, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                channel_id, event.event_id, event.title, event.status,
                event.window_seconds, iso(event.created_at), iso(event.locked_at),
                iso(event.ended_at), event.winning_outcome_id, now, now,
            ),
        )
    else:
        prediction_id = int(existing["id"])
        db.execute(
            "UPDATE predictions SET title = ?, status = ?, window_seconds = ?,"
            " created_at = COALESCE(?, created_at), locked_at = COALESCE(?, locked_at),"
            " ended_at = COALESCE(?, ended_at),"
            " winning_outcome_id = COALESCE(?, winning_outcome_id), updated_at = ?"
            " WHERE id = ?",
            (
                event.title, event.status, event.window_seconds,
                iso(event.created_at), iso(event.locked_at), iso(event.ended_at),
                event.winning_outcome_id, now, prediction_id,
            ),
        )

    for position, outcome in enumerate(event.outcomes):
        db.execute(
            "INSERT INTO prediction_outcomes"
            " (prediction_id, outcome_id, title, color, total_points, total_users, position)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(prediction_id, outcome_id) DO UPDATE SET"
            " title = excluded.title, color = excluded.color,"
            " total_points = excluded.total_points, total_users = excluded.total_users,"
            " position = excluded.position",
            (
                prediction_id, outcome.outcome_id, outcome.title, outcome.color,
                outcome.total_points, outcome.total_users, position,
            ),
        )
    return prediction_id


def record_pool_snapshot(prediction_id: int, event: PredictionEvent) -> None:
    """投票データ蓄積: one row per poll, so the pool's movement is replayable."""
    data = [
        {
            "outcome_id": o.outcome_id,
            "title": o.title,
            "points": o.total_points,
            "users": o.total_users,
        }
        for o in event.outcomes
    ]
    db.execute(
        "INSERT INTO pool_snapshots (prediction_id, ts, data) VALUES (?, ?, ?)",
        (prediction_id, db.now(), json.dumps(data, ensure_ascii=False)),
    )


def get_prediction(prediction_id: int) -> dict[str, Any] | None:
    row = db.row_to_dict(
        db.query_one("SELECT * FROM predictions WHERE id = ?", (prediction_id,))
    )
    if row is None:
        return None
    row["outcomes"] = db.rows_to_dicts(
        db.query(
            "SELECT * FROM prediction_outcomes WHERE prediction_id = ? ORDER BY position",
            (prediction_id,),
        )
    )
    return row


def prediction_by_event(event_id: str) -> dict[str, Any] | None:
    return db.row_to_dict(
        db.query_one("SELECT * FROM predictions WHERE event_id = ?", (event_id,))
    )


def recent_predictions(channel_id: int, limit: int = 30) -> list[dict[str, Any]]:
    rows = db.rows_to_dicts(
        db.query(
            "SELECT * FROM predictions WHERE channel_id = ?"
            " ORDER BY first_seen DESC LIMIT ?",
            (channel_id, limit),
        )
    )
    for row in rows:
        row["outcomes"] = db.rows_to_dicts(
            db.query(
                "SELECT * FROM prediction_outcomes WHERE prediction_id = ?"
                " ORDER BY position",
                (row["id"],),
            )
        )
        row["bets"] = db.rows_to_dicts(
            db.query("SELECT * FROM bets WHERE prediction_id = ? ORDER BY id", (row["id"],))
        )
        row["inference"] = db.row_to_dict(
            db.query_one(
                "SELECT * FROM inferences WHERE prediction_id = ? ORDER BY id DESC LIMIT 1",
                (row["id"],),
            )
        )
    return rows


def resolved_history_for_prompt(channel_id: int, limit: int) -> list[dict[str, Any]]:
    """Past resolved predictions, shaped for the LLM prompt.

    Which options were offered and which one won -- not how the vote split.
    Showing the model how the crowd priced past predictions only teaches it to
    follow the market, and the market's view is already accounted for in the
    Kelly step, which reads the live pool seconds before the deadline.
    """
    if limit <= 0:
        return []
    rows = db.query(
        "SELECT id, title, winning_outcome_id FROM predictions"
        " WHERE channel_id = ? AND status = 'RESOLVED' AND winning_outcome_id IS NOT NULL"
        " ORDER BY first_seen DESC LIMIT ?",
        (channel_id, limit),
    )
    history: list[dict[str, Any]] = []
    for row in rows:
        outcomes = db.rows_to_dicts(
            db.query(
                "SELECT outcome_id, title FROM prediction_outcomes"
                " WHERE prediction_id = ? ORDER BY position",
                (row["id"],),
            )
        )
        winner = next(
            (o["title"] for o in outcomes if o["outcome_id"] == row["winning_outcome_id"]),
            None,
        )
        history.append(
            {
                "title": row["title"],
                "winner_title": winner,
                "outcomes": [{"title": o["title"]} for o in outcomes],
            }
        )
    return list(reversed(history))


# -- inference and bets ----------------------------------------------------


def record_inference(
    prediction_id: int, source: str, probs: dict[str, float], *,
    rationale: str = "", raw_response: str | None = None, latency_ms: int | None = None,
) -> int:
    return db.execute(
        "INSERT INTO inferences (prediction_id, ts, source, probs, rationale,"
        " raw_response, latency_ms) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            prediction_id, db.now(), source,
            json.dumps(probs, ensure_ascii=False), rationale, raw_response, latency_ms,
        ),
    )


def record_bet(prediction_id: int, **fields: Any) -> int:
    columns = [
        "outcome_id", "outcome_title", "amount", "dry_run", "probability",
        "edge", "kelly_stake", "bankroll", "status", "error",
    ]
    values = [fields.get(c) for c in columns]
    return db.execute(
        f"INSERT INTO bets (prediction_id, ts, {', '.join(columns)})"
        f" VALUES (?, ?, {', '.join('?' * len(columns))})",
        [prediction_id, db.now(), *values],
    )


def has_bet(prediction_id: int) -> bool:
    row = db.query_one(
        "SELECT 1 FROM bets WHERE prediction_id = ? AND status IN ('placed', 'dry_run')"
        " LIMIT 1",
        (prediction_id,),
    )
    return row is not None


def settle_bets(prediction_id: int, winning_outcome_id: str | None, status: str) -> list[dict]:
    """Fill in win/lose/refund once the prediction resolves."""
    bets = db.rows_to_dicts(
        db.query(
            "SELECT * FROM bets WHERE prediction_id = ? AND status IN ('placed', 'dry_run')"
            " AND result IS NULL",
            (prediction_id,),
        )
    )
    if not bets:
        return []

    outcomes = db.rows_to_dicts(
        db.query(
            "SELECT outcome_id, total_points FROM prediction_outcomes WHERE prediction_id = ?",
            (prediction_id,),
        )
    )
    total = sum(max(0, o["total_points"]) for o in outcomes)

    settled = []
    for bet in bets:
        if status == "CANCELED" or not winning_outcome_id:
            result, payout = "refund", bet["amount"]
        elif bet["outcome_id"] == winning_outcome_id:
            won = next(
                (max(0, o["total_points"]) for o in outcomes
                 if o["outcome_id"] == winning_outcome_id),
                0,
            )
            payout = int(round(total * bet["amount"] / won)) if won > 0 else bet["amount"]
            result = "win"
        else:
            result, payout = "lose", 0
        db.execute(
            "UPDATE bets SET result = ?, payout = ? WHERE id = ?",
            (result, payout, bet["id"]),
        )
        bet["result"], bet["payout"] = result, payout
        settled.append(bet)
    return settled


# -- transcripts -----------------------------------------------------------


def add_transcript(channel_id: int, text: str) -> None:
    text = text.strip()
    if text:
        db.execute(
            "INSERT INTO transcripts (channel_id, ts, text) VALUES (?, ?, ?)",
            (channel_id, db.now(), text),
        )


def recent_transcript_text(channel_id: int, retention_min: int, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    rows = db.query(
        "SELECT ts, text FROM transcripts WHERE channel_id = ?"
        " AND ts >= datetime('now', ?) ORDER BY id DESC LIMIT 4000",
        (channel_id, f"-{int(retention_min)} minutes"),
    )
    chunks: list[str] = []
    used = 0
    for row in rows:  # newest first, so the tail of the window is what survives
        line = f"[{str(row['ts'])[11:19]}] {row['text']}"
        if used + len(line) > max_chars:
            break
        chunks.append(line)
        used += len(line) + 1
    return "\n".join(reversed(chunks))


def prune_transcripts(retention_min: int) -> int:
    row = db.query_one(
        "SELECT COUNT(*) AS n FROM transcripts WHERE ts < datetime('now', ?)",
        (f"-{int(retention_min)} minutes",),
    )
    n = int(row["n"]) if row else 0
    if n:
        db.execute(
            "DELETE FROM transcripts WHERE ts < datetime('now', ?)",
            (f"-{int(retention_min)} minutes",),
        )
    return n


def transcript_lines(channel_id: int, retention_min: int, limit: int = 200) -> list[dict]:
    rows = db.query(
        "SELECT ts, text FROM transcripts WHERE channel_id = ?"
        " AND ts >= datetime('now', ?) ORDER BY id DESC LIMIT ?",
        (channel_id, f"-{int(retention_min)} minutes", limit),
    )
    return list(reversed(db.rows_to_dicts(rows)))
