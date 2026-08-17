"""Chat-completions client for the local llama-server."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..config import LlamaSettings
from .prompt import (
    QUERY_SCHEMA,
    RESPONSE_SCHEMA,
    PromptContext,
    build_messages,
    build_query_messages,
)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class LlamaError(RuntimeError):
    pass


@dataclass
class InferenceResult:
    probabilities: dict[str, float]
    """Keyed by Twitch outcome id, normalised to sum to 1."""

    rationale: str = ""
    raw_response: str = ""
    latency_ms: int = 0
    prompt_chars: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class QuerySuggestion:
    """What the model thinks the prediction is about, and what to search for."""

    topic: str = ""
    query: str = ""
    latency_ms: int = 0


class LlamaClient:
    def __init__(self, settings: LlamaSettings, base_url: str) -> None:
        self.settings = settings
        self.base_url = base_url.rstrip("/")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.api_key.strip():
            headers["Authorization"] = f"Bearer {self.settings.api_key.strip()}"
        return headers

    async def infer(self, ctx: PromptContext) -> InferenceResult:
        outcomes = ctx.event.outcomes
        if not outcomes:
            raise LlamaError("選択肢のない予想は推論できません")

        messages = build_messages(ctx)
        prompt_chars = sum(len(m["content"]) for m in messages)
        body: dict[str, Any] = {
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "prediction", "schema": RESPONSE_SCHEMA,
                                "strict": True},
            },
        }

        started = time.perf_counter()
        text = await self._complete(body)
        latency_ms = int((time.perf_counter() - started) * 1000)

        parsed, warnings = _parse(text, len(outcomes))
        probs = {
            outcomes[idx - 1].outcome_id: value for idx, value in parsed.items()
        }
        probs, norm_warnings = _normalise(probs, [o.outcome_id for o in outcomes])

        return InferenceResult(
            probabilities=probs,
            rationale=_extract_rationale(text),
            raw_response=text[:8000],
            latency_ms=latency_ms,
            prompt_chars=prompt_chars,
            warnings=warnings + norm_warnings,
        )

    async def suggest_query(
        self, ctx: PromptContext, transcript_chars: int
    ) -> QuerySuggestion:
        """Work out what this prediction is about, and what to search for.

        Runs before the real inference: "優勝 する / しない" says nothing on its
        own, and only the transcript reveals that the game is Apex. A mechanical
        query built from the title alone cannot find that out.
        """
        body: dict[str, Any] = {
            "messages": build_query_messages(ctx, transcript_chars),
            "temperature": min(self.settings.temperature, 0.3),
            "max_tokens": 256,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "search_query", "schema": QUERY_SCHEMA,
                                "strict": True},
            },
        }
        started = time.perf_counter()
        text = await self._complete(body)
        latency_ms = int((time.perf_counter() - started) * 1000)

        data = _loads(text)
        return QuerySuggestion(
            topic=str(data.get("topic") or "").strip()[:200],
            query=str(data.get("query") or "").strip()[:300],
            latency_ms=latency_ms,
        )

    async def _complete(self, body: dict[str, Any]) -> str:
        url = f"{self.base_url}/v1/chat/completions"
        timeout = self.settings.request_timeout_sec
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=body, headers=self._headers())
            if resp.status_code == 400 and "response_format" in body:
                # Older llama.cpp builds reject the json_schema form; json_object
                # is supported much further back and the prompt already carries
                # the schema.
                retry = dict(body)
                retry["response_format"] = {"type": "json_object"}
                resp = await client.post(url, json=retry, headers=self._headers())
            if resp.status_code == 401:
                raise LlamaError("llama-server の認証に失敗しました (API キーを確認してください)")
            if resp.status_code >= 400:
                raise LlamaError(
                    f"llama-server が HTTP {resp.status_code} を返しました: {resp.text[:500]}"
                )
            try:
                payload = resp.json()
            except ValueError as exc:
                raise LlamaError("llama-server の応答が JSON ではありません") from exc

        choices = payload.get("choices") or []
        if not choices:
            raise LlamaError("llama-server が空の応答を返しました")
        return str((choices[0].get("message") or {}).get("content") or "")

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/health", headers=self._headers())
                return resp.status_code == 200
        except httpx.HTTPError:
            return False


def _loads(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        pass
    match = _JSON_BLOCK.search(text or "")
    if match:
        try:
            return json.loads(match.group(0))
        except ValueError:
            pass
    raise LlamaError(f"推論結果を JSON として解釈できませんでした: {(text or '')[:300]}")


def _parse(text: str, outcome_count: int) -> tuple[dict[int, float], list[str]]:
    """Pull `{choice: probability}` out of the model's reply."""
    data = _loads(text)
    warnings: list[str] = []
    items = data.get("probabilities")

    raw: dict[int, float] = {}
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                choice = int(item.get("choice"))
                value = float(item.get("probability"))
            except (TypeError, ValueError):
                continue
            if 1 <= choice <= outcome_count:
                raw[choice] = max(0.0, value)
    elif isinstance(items, dict):
        # Tolerate {"1": 0.6, "2": 0.4}.
        for key, value in items.items():
            try:
                choice = int(key)
                raw[choice] = max(0.0, float(value))
            except (TypeError, ValueError):
                continue

    if not raw:
        raise LlamaError(f"確率を読み取れませんでした: {(text or '')[:300]}")
    missing = [i for i in range(1, outcome_count + 1) if i not in raw]
    if missing:
        warnings.append(f"選択肢 {missing} の確率が返らなかったため 0 として扱いました")
    return raw, warnings


def _normalise(
    probs: dict[str, float], outcome_ids: list[str]
) -> tuple[dict[str, float], list[str]]:
    warnings: list[str] = []
    full = {oid: max(0.0, float(probs.get(oid, 0.0))) for oid in outcome_ids}
    total = sum(full.values())
    if total <= 0:
        warnings.append("確率がすべて 0 だったため一様分布に置き換えました")
        share = 1.0 / len(outcome_ids) if outcome_ids else 0.0
        return {oid: share for oid in outcome_ids}, warnings
    if abs(total - 1.0) > 0.02:
        warnings.append(f"確率の合計が {total:.3f} だったため正規化しました")
    return {oid: value / total for oid, value in full.items()}, warnings


def _extract_rationale(text: str) -> str:
    try:
        data = _loads(text)
    except LlamaError:
        return ""
    value = data.get("rationale")
    return str(value)[:1000] if value else ""
