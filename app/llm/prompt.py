"""Prompt construction for prediction inference.

Outcomes are numbered 1..N in the prompt rather than passed by their Twitch
UUIDs: a small local model reproduces `1` reliably and a 36-character UUID much
less so. The numbers are mapped back to outcome ids after parsing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..twitch.models import PredictionEvent

SYSTEM_PROMPT = """あなたは Twitch のチャンネルポイント予想を分析するアナリストです。
配信の文脈・過去の類似予想の結果・現在の投票状況から、各選択肢が的中する確率を推定します。

厳守事項:
- 出力は指定された JSON のみ。前後に説明文やコードフェンスを付けない。
- probability は 0 以上 1 以下で、全選択肢の合計が 1 になるようにする。
- 視聴者の投票分布は「群衆の予想」であって正解ではない。文脈に根拠があるときは分布から離れてよい。
- 根拠が乏しいときは、無理に偏らせず一様に近い確率を返す。過信は損失に直結する。
- rationale は日本語で 200 文字以内。"""

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "probabilities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "choice": {"type": "integer"},
                    "probability": {"type": "number"},
                },
                "required": ["choice", "probability"],
            },
        },
        "rationale": {"type": "string"},
    },
    "required": ["probabilities", "rationale"],
}


@dataclass
class PromptContext:
    channel_login: str
    channel_display: str
    event: PredictionEvent
    manual_info: str = ""
    history: list[dict[str, Any]] | None = None
    transcript: str = ""
    seconds_until_lock: float | None = None


def _format_pool(event: PredictionEvent) -> str:
    total = event.total_points
    lines = []
    for i, o in enumerate(event.outcomes, start=1):
        share = (o.total_points / total * 100) if total else 0.0
        lines.append(
            f"  {i}. {o.title}"
            f" — {o.total_points:,} pt / {o.total_users:,} 人 ({share:.1f}%)"
        )
    return "\n".join(lines) if lines else "  (選択肢なし)"


def _format_history(history: list[dict[str, Any]] | None) -> str:
    if not history:
        return "  (このチャンネルの過去データはまだありません)"
    lines = []
    for h in history:
        winner = h.get("winner_title") or "不明"
        choices = " / ".join(
            f"{o['title']}({o.get('share', 0):.0%})" for o in h.get("outcomes", [])
        )
        lines.append(f"  - 「{h.get('title', '')}」 選択肢: {choices} → 結果: {winner}")
    return "\n".join(lines)


def build_messages(ctx: PromptContext) -> list[dict[str, str]]:
    event = ctx.event
    parts: list[str] = []

    parts.append(f"# チャンネル\n{ctx.channel_display} (twitch.tv/{ctx.channel_login})")

    if ctx.manual_info.strip():
        parts.append(
            "# 事前情報 (人手で入力された、この配信/予想に関する補足)\n"
            f"{ctx.manual_info.strip()}"
        )

    lock_note = ""
    if ctx.seconds_until_lock is not None:
        lock_note = f"\n締め切りまで: 約 {max(0, int(ctx.seconds_until_lock))} 秒"

    parts.append(
        f"# 現在の予想\nタイトル: {event.title}{lock_note}\n"
        f"現在の投票状況 (合計 {event.total_points:,} pt):\n{_format_pool(event)}"
    )

    parts.append(f"# このチャンネルの過去の予想と結果\n{_format_history(ctx.history)}")

    if ctx.transcript.strip():
        parts.append(
            "# 直近の配信音声の文字起こし (古い順、認識誤りを含む)\n"
            f"{ctx.transcript.strip()}"
        )
    else:
        parts.append("# 直近の配信音声の文字起こし\n(文字起こしなし)")

    numbered = "\n".join(
        f"  {i}: {o.title}" for i, o in enumerate(event.outcomes, start=1)
    )
    parts.append(
        "# 出力\n"
        f"選択肢の番号は次のとおりです:\n{numbered}\n\n"
        "次の JSON だけを出力してください:\n"
        '{"probabilities": [{"choice": <番号>, "probability": <0..1>}, ...], '
        '"rationale": "<根拠>"}'
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def estimate_chars(messages: list[dict[str, str]]) -> int:
    return sum(len(m["content"]) for m in messages)
