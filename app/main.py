"""FastAPI application: JSON API plus the static UI."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, db, log, store
from .paths import WEB_DIR, ensure_dirs
from .search.client import probe as probe_search
from .tracker import tracker
from .twitch.gql import TwitchError

@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    ensure_dirs()
    db.connect()
    log.bind_loop(asyncio.get_running_loop())
    config.load()
    log.info(log.CAT_SYSTEM, "アプリケーションを起動しました")
    try:
        yield
    finally:
        with contextlib.suppress(Exception):
            await tracker.stop()


app = FastAPI(
    title="twitch_autobet", version="0.1.0", docs_url="/api/docs", lifespan=lifespan
)


# -- request bodies --------------------------------------------------------


class ChannelCreate(BaseModel):
    login: str = Field(min_length=1, max_length=64)


class FixedProbs(BaseModel):
    enabled: bool = False
    probs: list[float] = Field(default_factory=list)


class ChannelPatch(BaseModel):
    enabled: bool | None = None
    manual_info: str | None = None
    fixed_probs: FixedProbs | None = None


class LoginBody(BaseModel):
    login: str = Field(min_length=1, max_length=64)


# -- tracking --------------------------------------------------------------


@app.get("/api/status")
async def get_status() -> dict[str, Any]:
    return tracker.status()


@app.post("/api/tracking/start")
async def start_tracking() -> dict[str, Any]:
    try:
        await tracker.start()
    except Exception as exc:
        log.error(f"追跡の開始に失敗しました: {exc}", exc=exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return tracker.status()


@app.post("/api/tracking/stop")
async def stop_tracking() -> dict[str, Any]:
    await tracker.stop()
    return tracker.status()


# -- channels --------------------------------------------------------------


@app.get("/api/channels")
async def get_channels() -> list[dict[str, Any]]:
    status = tracker.status().get("channels", {})
    channels = store.list_channels()
    for row in channels:
        live = status.get(str(row["id"]), {})
        row["runtime"] = {
            "balance": live.get("balance", row.get("last_points")),
            "last_poll": live.get("last_poll"),
            "last_error": live.get("last_error"),
            "active_event_id": live.get("active_event_id"),
            "next_bet_at": live.get("next_bet_at"),
            "tracking": bool(live),
        }
    return channels


@app.post("/api/channels", status_code=201)
async def add_channel(body: ChannelCreate) -> dict[str, Any]:
    login = body.login.strip().lower().lstrip("@")
    if "/" in login:
        login = login.rstrip("/").rsplit("/", 1)[-1]
    if not login:
        raise HTTPException(status_code=400, detail="チャンネル名を入力してください")
    if store.get_channel_by_login(login):
        raise HTTPException(status_code=409, detail=f"'{login}' は既に登録されています")

    try:
        info = await tracker.client().resolve_channel(login)
    except TwitchError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    channel_id = store.create_channel(info.login, info.display_name, info.twitch_id)
    log.info(
        log.CAT_SYSTEM,
        f"チャンネルを登録しました: {info.display_name} ({info.login})",
        channel=info.login,
    )
    return store.get_channel(channel_id) or {}


@app.patch("/api/channels/{channel_id}")
async def patch_channel(channel_id: int, body: ChannelPatch) -> dict[str, Any]:
    channel = store.get_channel(channel_id)
    if channel is None:
        raise HTTPException(status_code=404, detail="チャンネルが見つかりません")

    fields: dict[str, Any] = {}
    if body.enabled is not None:
        fields["enabled"] = body.enabled
        log.info(
            log.CAT_SYSTEM,
            "追跡を有効にしました" if body.enabled else "追跡対象から除外しました",
            channel=channel["login"],
        )
    if body.manual_info is not None:
        fields["manual_info"] = body.manual_info.strip()
        log.info(log.CAT_SYSTEM, "手動情報を更新しました", channel=channel["login"],
                 detail={"manual_info": fields["manual_info"][:500]})
    if body.fixed_probs is not None:
        payload = body.fixed_probs.model_dump()
        if payload["enabled"]:
            if len(payload["probs"]) < 2:
                raise HTTPException(
                    status_code=400, detail="固定確率は 2 つ以上の値が必要です"
                )
            if any(p < 0 for p in payload["probs"]) or sum(payload["probs"]) <= 0:
                raise HTTPException(
                    status_code=400, detail="固定確率は 0 以上で、合計が正である必要があります"
                )
        fields["fixed_probs"] = payload
        log.info(
            log.CAT_SYSTEM,
            "固定確率を有効にしました" if payload["enabled"] else "固定確率を無効にしました",
            channel=channel["login"],
            detail=payload,
        )

    store.update_channel(channel_id, **fields)
    if body.enabled is not None and tracker.running:
        log.warn(
            log.CAT_SYSTEM,
            "追跡対象の変更は次回の追跡開始から反映されます",
            channel=channel["login"],
        )
    return store.get_channel(channel_id) or {}


@app.delete("/api/channels/{channel_id}")
async def remove_channel(channel_id: int) -> dict[str, str]:
    channel = store.get_channel(channel_id)
    if channel is None:
        raise HTTPException(status_code=404, detail="チャンネルが見つかりません")
    store.delete_channel(channel_id)
    log.info(
        log.CAT_SYSTEM,
        f"チャンネルの登録を解除し、蓄積データを削除しました: {channel['login']}",
        channel=channel["login"],
    )
    return {"status": "deleted"}


@app.get("/api/channels/{channel_id}/points")
async def get_points(channel_id: int, limit: int = Query(2000, ge=1, le=20000)) -> dict:
    if store.get_channel(channel_id) is None:
        raise HTTPException(status_code=404, detail="チャンネルが見つかりません")
    return {"points": store.points_history(channel_id, limit)}


@app.get("/api/channels/{channel_id}/predictions")
async def get_predictions(channel_id: int, limit: int = Query(30, ge=1, le=200)) -> dict:
    if store.get_channel(channel_id) is None:
        raise HTTPException(status_code=404, detail="チャンネルが見つかりません")
    return {"predictions": store.recent_predictions(channel_id, limit)}


@app.get("/api/channels/{channel_id}/transcript")
async def get_transcript(channel_id: int, limit: int = Query(200, ge=1, le=2000)) -> dict:
    if store.get_channel(channel_id) is None:
        raise HTTPException(status_code=404, detail="チャンネルが見つかりません")
    minutes = config.load().transcription.retention_min
    return {
        "retention_min": minutes,
        "lines": store.transcript_lines(channel_id, minutes, limit),
    }


# -- settings --------------------------------------------------------------


@app.get("/api/settings")
async def get_settings() -> dict[str, Any]:
    return config.load().model_dump()


@app.put("/api/settings")
async def put_settings(patch: dict[str, Any] = Body(...)) -> dict[str, Any]:
    try:
        settings = config.update(patch)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"設定が不正です: {exc}") from exc
    log.info(log.CAT_SYSTEM, "設定を更新しました", detail={"keys": sorted(patch)})
    if tracker.running and ("llama" in patch or "twitch" in patch):
        log.warn(
            log.CAT_SYSTEM,
            "llama.cpp / Twitch 認証の変更は追跡を再起動するまで反映されません",
        )
    return settings.model_dump()


@app.post("/api/settings/test/twitch")
async def test_twitch(body: LoginBody) -> dict[str, Any]:
    login = body.login.strip().lower().lstrip("@")
    if not login:
        raise HTTPException(status_code=400, detail="チャンネル名を入力してください")
    checks = await tracker.client().diagnose(login)
    ok = all(c["ok"] for c in checks)
    log.write(
        log.CAT_SYSTEM,
        f"Twitch 接続テスト: {'成功' if ok else '一部失敗'}",
        level="INFO" if ok else "WARN",
        channel=login,
        detail=checks,
    )
    return {"ok": ok, "checks": checks}


@app.post("/api/settings/test/llama")
async def test_llama() -> dict[str, Any]:
    settings = config.load().llama
    problems = tracker.llama.validate(settings)
    if settings.mode == "never":
        # Nothing will be launched, so unset paths are not a misconfiguration.
        # Still report them, so switching the mode back is not a surprise.
        return {
            "ok": True,
            "mode": settings.mode,
            "problems": [],
            "note": "LLM の使用が「使わない」のため llama-server は起動しません"
                    + (f" (未解決の設定: {'; '.join(problems)})" if problems else ""),
            "command": [],
            "server": tracker.llama.status,
        }
    return {
        "ok": not problems,
        "mode": settings.mode,
        "problems": problems,
        "command": tracker.llama.build_args(settings) if not problems else [],
        "server": tracker.llama.status,
    }


@app.post("/api/settings/test/search")
async def test_search() -> dict[str, Any]:
    result = await probe_search(config.load().search)
    log.write(
        log.CAT_SYSTEM,
        f"ウェブ検索 接続テスト: {'成功' if result['ok'] else '失敗'}",
        level="INFO" if result["ok"] else "WARN",
        detail=result,
    )
    return result


# -- logs ------------------------------------------------------------------


@app.get("/api/logs")
async def get_logs(
    limit: int = Query(300, ge=1, le=2000),
    category: list[str] | None = Query(None),
    channel: str | None = Query(None),
    before_id: int | None = Query(None),
) -> dict[str, Any]:
    return {
        "categories": log.CATEGORIES,
        "logs": log.recent(limit, category, channel, before_id),
    }


@app.delete("/api/logs")
async def clear_logs() -> dict[str, str]:
    db.execute("DELETE FROM logs")
    log.info(log.CAT_SYSTEM, "ログを消去しました")
    return {"status": "cleared"}


@app.get("/api/stream")
async def stream() -> StreamingResponse:
    """Server-sent events: log lines and state changes, pushed live."""

    async def generator() -> AsyncIterator[bytes]:
        queue = log.subscribe()
        try:
            yield b": connected\n\n"
            while True:
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=20.0)
                except asyncio.TimeoutError:
                    yield b": keepalive\n\n"
                    continue
                data = json.dumps(payload, ensure_ascii=False, default=str)
                yield f"data: {data}\n\n".encode()
        finally:
            log.unsubscribe(queue)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# -- static UI -------------------------------------------------------------


# The pages and the ES modules that drive them are edited as a pair: a browser
# holding a cached half renders a page whose script cannot find the elements it
# expects, and the screen comes up blank with only a console error. `no-cache`
# does not forbid caching, it forbids reusing without asking -- the ETag still
# turns the ask into a 304 for everything that has not changed.
_NO_CACHE = {"Cache-Control": "no-cache"}


class _RevalidatedStatic(StaticFiles):
    def file_response(self, *args: Any, **kwargs: Any) -> Response:
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html", headers=_NO_CACHE)


@app.get("/settings")
async def settings_page() -> FileResponse:
    return FileResponse(WEB_DIR / "settings.html", headers=_NO_CACHE)


app.mount("/static", _RevalidatedStatic(directory=WEB_DIR), name="static")
