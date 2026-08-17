"""Normalised Twitch prediction types.

The GraphQL API answers in camelCase and PubSub in snake_case, with slightly
different field names for the same data. Both are funnelled through here so the
rest of the app only ever sees one shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

ACTIVE = "ACTIVE"
LOCKED = "LOCKED"
RESOLVED = "RESOLVED"
CANCELED = "CANCELED"

# PubSub reports the resolved state as RESOLVE_PENDING then RESOLVED.
OPEN_STATES = {ACTIVE}
FINAL_STATES = {RESOLVED, CANCELED}


def _parse_ts(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    text = value.replace("Z", "+00:00")
    # Twitch sometimes emits more than 6 fractional digits, which fromisoformat
    # rejects on older parsers -- trim to microseconds.
    if "." in text:
        head, _, tail = text.partition(".")
        digits = ""
        rest = ""
        for i, ch in enumerate(tail):
            if ch.isdigit():
                digits += ch
            else:
                rest = tail[i:]
                break
        text = f"{head}.{digits[:6]}{rest}"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


@dataclass
class ChannelInfo:
    twitch_id: str
    login: str
    display_name: str
    is_live: bool = False


@dataclass
class StreamInfo:
    """What is being streamed right now, for search queries and prompt context."""

    title: str = ""
    game: str = ""

    def __bool__(self) -> bool:
        return bool(self.title or self.game)


@dataclass
class PredictionOutcome:
    outcome_id: str
    title: str
    color: str | None = None
    total_points: int = 0
    total_users: int = 0

    @classmethod
    def parse(cls, raw: dict[str, Any]) -> PredictionOutcome:
        return cls(
            outcome_id=str(raw.get("id") or ""),
            title=str(raw.get("title") or ""),
            color=raw.get("color"),
            total_points=int(raw.get("totalPoints") or raw.get("total_points") or 0),
            total_users=int(raw.get("totalUsers") or raw.get("total_users") or 0),
        )


@dataclass
class PredictionEvent:
    event_id: str
    channel_twitch_id: str
    title: str
    status: str
    outcomes: list[PredictionOutcome] = field(default_factory=list)
    created_at: datetime | None = None
    locked_at: datetime | None = None
    ended_at: datetime | None = None
    window_seconds: int | None = None
    winning_outcome_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, raw: dict[str, Any]) -> PredictionEvent:
        outcomes = [PredictionOutcome.parse(o) for o in (raw.get("outcomes") or [])]
        winner = raw.get("winning_outcome_id")
        if not winner and isinstance(raw.get("winningOutcome"), dict):
            winner = raw["winningOutcome"].get("id")
        window = raw.get("predictionWindowSeconds") or raw.get("prediction_window_seconds")
        return cls(
            event_id=str(raw.get("id") or ""),
            channel_twitch_id=str(raw.get("channel_id") or raw.get("channelId") or ""),
            title=str(raw.get("title") or ""),
            status=str(raw.get("status") or ACTIVE).upper(),
            outcomes=outcomes,
            created_at=_parse_ts(raw.get("createdAt") or raw.get("created_at")),
            locked_at=_parse_ts(raw.get("lockedAt") or raw.get("locked_at")),
            ended_at=_parse_ts(raw.get("endedAt") or raw.get("ended_at")),
            window_seconds=int(window) if window else None,
            winning_outcome_id=str(winner) if winner else None,
            raw=raw,
        )

    @property
    def total_points(self) -> int:
        return sum(max(0, o.total_points) for o in self.outcomes)

    @property
    def is_open(self) -> bool:
        return self.status in OPEN_STATES

    @property
    def is_final(self) -> bool:
        return self.status in FINAL_STATES

    def lock_time(self) -> datetime | None:
        """When betting closes, from `locked_at` or created_at + window."""
        if self.locked_at:
            return self.locked_at
        if self.created_at and self.window_seconds:
            return self.created_at + timedelta(seconds=self.window_seconds)
        return None

    def seconds_until_lock(self, now: datetime | None = None) -> float | None:
        lock = self.lock_time()
        if lock is None:
            return None
        return (lock - (now or datetime.now(UTC))).total_seconds()

    def outcome(self, outcome_id: str) -> PredictionOutcome | None:
        return next((o for o in self.outcomes if o.outcome_id == outcome_id), None)
