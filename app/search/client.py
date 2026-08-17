"""Web search used to enrich the inference prompt.

Brave's Search API, called once per prediction with a query the model wrote
(see `llm.client.suggest_query`). Assembling the query here instead would not
work: 「優勝 / する / しない」 identifies nothing, and the game is frequently
named only in the transcript, which no amount of string joining can extract.

Every failure here returns an empty list. Search is an enrichment, and losing it
must never cost us a prediction: the model still has the transcript, the channel
history and its own knowledge.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

import httpx

from .. import log
from ..config import SearchSettings

BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"

# Brave marks matched terms with <strong> in titles and descriptions.
_TAGS = re.compile(r"<[^>]+>")


def _plain(value: object) -> str:
    return html.unescape(_TAGS.sub("", str(value or ""))).strip()


@dataclass
class SearchHit:
    title: str
    description: str
    url: str


def _clip(hits: list[SearchHit], max_chars: int) -> list[SearchHit]:
    """Keep whole hits until the character budget runs out."""
    kept: list[SearchHit] = []
    used = 0
    for hit in hits:
        size = len(hit.title) + len(hit.description)
        if used + size > max_chars:
            break
        kept.append(hit)
        used += size
    return kept


async def search(query: str, settings: SearchSettings) -> list[SearchHit]:
    """Run the query, or return an empty list if anything at all goes wrong."""
    if not settings.enabled or not query.strip() or settings.max_chars <= 0:
        return []
    if not settings.api_key.strip():
        log.warn(log.CAT_INFERENCE, "検索が有効ですが API キーが未設定です")
        return []

    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": settings.api_key.strip(),
    }
    params = {
        "q": query,
        "count": settings.count,
        "country": settings.country,
        "search_lang": settings.lang,
    }
    try:
        async with httpx.AsyncClient(timeout=settings.timeout_sec) as client:
            resp = await client.get(BRAVE_URL, headers=headers, params=params)
            resp.raise_for_status()
            payload = resp.json()
    except httpx.HTTPStatusError as exc:
        log.warn(
            log.CAT_INFERENCE,
            f"検索に失敗しました (HTTP {exc.response.status_code}) — 検索なしで推論します",
            detail={"query": query},
        )
        return []
    except Exception as exc:  # noqa: BLE001 - enrichment must never block inference
        log.warn(
            log.CAT_INFERENCE,
            f"検索に失敗しました ({type(exc).__name__}) — 検索なしで推論します",
            detail={"query": query, "error": str(exc)[:200]},
        )
        return []

    results = ((payload.get("web") or {}).get("results")) or []
    hits = [
        SearchHit(
            title=_plain(r.get("title")),
            description=_plain(r.get("description")),
            url=_plain(r.get("url")),
        )
        for r in results
    ]
    return _clip([h for h in hits if h.title or h.description], settings.max_chars)
