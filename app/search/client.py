"""Web search used to enrich the inference prompt.

A self-hosted SearXNG, called once per prediction with a query the model wrote
(see `llm.client.suggest_query`). Assembling the query here instead would not
work: 「優勝 / する / しない」 identifies nothing, and the game is frequently
named only in the transcript, which no amount of string joining can extract.

SearXNG rather than a hosted search API because this already runs beside
llama-server on one machine: a local metasearch container has no key, no card
and no per-query price, and one query per prediction is exactly the volume that
makes a metered API annoying out of proportion to what it costs.

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

SEARCH_PATH = "/search"

# `probe` needs a query that any working instance answers. A real word, so that
# zero results means the engines are broken rather than the term being obscure.
PROBE_QUERY = "twitch"

# Some SearXNG engines pass the upstream markup through, <strong> on matched
# terms most often. The model wants the words, not the tags.
_TAGS = re.compile(r"<[^>]+>")


def _plain(value: object) -> str:
    return html.unescape(_TAGS.sub("", str(value or ""))).strip()


@dataclass
class SearchHit:
    title: str
    description: str
    url: str


def _parse(payload: dict) -> list[SearchHit]:
    """SearXNG's `results` into hits, dropping the ones with nothing to read."""
    hits = [
        SearchHit(
            title=_plain(r.get("title")),
            description=_plain(r.get("content")),
            url=_plain(r.get("url")),
        )
        for r in (payload.get("results") or [])
    ]
    return [h for h in hits if h.title or h.description]


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
    base = settings.base_url.strip().rstrip("/")
    if not base:
        log.warn(log.CAT_INFERENCE, "検索が有効ですが SearXNG の URL が未設定です")
        return []

    headers = {"Accept": "application/json"}
    params = {
        "q": query,
        "format": "json",
        "language": settings.language,
        "pageno": 1,
    }
    try:
        async with httpx.AsyncClient(timeout=settings.timeout_sec) as client:
            resp = await client.get(base + SEARCH_PATH, headers=headers, params=params)
            resp.raise_for_status()
            payload = resp.json()
    except httpx.HTTPStatusError as exc:
        # A fresh SearXNG returns 403 to both of the mistakes worth naming:
        # `json` missing from search.formats, and the bot limiter turning away
        # a client that is not a browser. Neither is obvious from "403".
        hint = ""
        if exc.response.status_code == 403:
            hint = (
                " — settings.yml の search.formats に json を足し、"
                "server.limiter を false にしてください"
            )
        log.warn(
            log.CAT_INFERENCE,
            f"検索に失敗しました (HTTP {exc.response.status_code}){hint}"
            " — 検索なしで推論します",
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

    return _clip(_parse(payload)[: settings.count], settings.max_chars)


async def probe(settings: SearchSettings) -> dict:
    """Diagnose the configured SearXNG, for the settings screen's test button.

    `search` swallows every failure on purpose, which means a SearXNG that is
    down or still on its default config is invisible until a prediction fires
    and quietly goes out without any search results. This is the same request
    with the failures named instead of dropped -- and it runs whether or not
    search is enabled, so the instance can be checked before switching it on.
    """
    base = settings.base_url.strip().rstrip("/")
    result: dict = {
        "ok": False,
        "enabled": settings.enabled,
        "url": (base + SEARCH_PATH) if base else "",
        "problems": [],
        "results": 0,
        "sample": [],
    }
    if not base:
        result["problems"].append("SearXNG の URL が未設定です")
        return result

    # Long enough that a cold container is not mistaken for a dead one.
    timeout = max(settings.timeout_sec, 10.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(
                base + SEARCH_PATH,
                headers={"Accept": "application/json"},
                params={"q": PROBE_QUERY, "format": "json",
                        "language": settings.language},
            )
    except httpx.ConnectError:
        result["problems"].append(
            f"{base} に接続できません。コンテナが起動しているか確認してください "
            "(docker start searxng)"
        )
        return result
    except httpx.TimeoutException:
        # Accepted the connection but never answered: a container still warming
        # up, or one that is wedged. Same remedy either way, so say so.
        result["problems"].append(
            f"{base} が {timeout:.0f} 秒以内に応答しませんでした。"
            "起動直後か、コンテナが詰まっています (docker restart searxng)"
        )
        return result
    except Exception as exc:  # noqa: BLE001 - the test button reports, never raises
        result["problems"].append(f"接続に失敗しました ({type(exc).__name__}): {exc}")
        return result

    if resp.status_code == 403:
        result["problems"].append(
            "403 が返りました。settings.yml の search.formats に json を足し、"
            "server.limiter を false にしてコンテナを再起動してください"
        )
        return result
    if resp.status_code != 200:
        result["problems"].append(f"HTTP {resp.status_code} が返りました")
        return result

    try:
        payload = resp.json()
    except ValueError:
        # A 200 that is not JSON is SearXNG handing back its HTML search page.
        result["problems"].append(
            "JSON ではなく HTML が返りました。settings.yml の search.formats に "
            "json を足してコンテナを再起動してください"
        )
        return result

    hits = _parse(payload)
    result["results"] = len(hits)
    result["sample"] = [{"title": h.title, "url": h.url} for h in hits[:3]]
    if not hits:
        result["problems"].append(
            "接続できましたが結果が 0 件でした。settings.yml で engines が"
            "すべて無効になっているか、上流の検索エンジンに弾かれています"
        )
        return result

    result["ok"] = True
    return result
