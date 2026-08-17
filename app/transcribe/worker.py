"""Live transcription: streamlink -> ffmpeg -> faster-whisper.

streamlink resolves the HLS playlist URL, ffmpeg demuxes it to raw 16 kHz mono
PCM on stdout, and fixed-length chunks are handed to Whisper.

Chunks go through a bounded queue that drops the oldest entry when full. On CPU
the model can fall behind a live stream, and when that happens the useful thing
is the most recent audio, not a growing backlog of stale audio -- and dropping
here is what stops ffmpeg's pipe from applying backpressure to a live source.
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import sys
from dataclasses import dataclass, field
from typing import Any

from .. import log, store
from ..config import TranscriptionSettings
from ..paths import WHISPER_CACHE

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
READ_SIZE = 1 << 16
OFFLINE_RETRY_SEC = 60.0
ERROR_RETRY_SEC = 15.0


@dataclass
class ChannelState:
    login: str
    running: bool = False
    live: bool = False
    last_error: str | None = None
    chunks_done: int = 0
    chunks_dropped: int = 0
    last_text: str = ""


@dataclass
class _Worker:
    channel_id: int
    login: str
    task: asyncio.Task
    state: ChannelState
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=2))


class TranscriptionManager:
    """Owns one audio pipeline per tracked channel plus the shared Whisper model."""

    def __init__(self) -> None:
        self._workers: dict[int, _Worker] = {}
        self._model: Any = None
        self._model_key: tuple | None = None
        self._model_lock = asyncio.Lock()
        # Whisper inference is serialised: concurrent CPU decodes just thrash.
        self._infer_lock = asyncio.Semaphore(1)
        self._settings = TranscriptionSettings()
        self._last_error: str | None = None

    # -- lifecycle ---------------------------------------------------------

    def status(self) -> dict:
        return {
            "enabled": self._settings.enabled,
            "model_loaded": self._model is not None,
            "last_error": self._last_error,
            "channels": {
                str(cid): {
                    "login": w.state.login,
                    "live": w.state.live,
                    "chunks_done": w.state.chunks_done,
                    "chunks_dropped": w.state.chunks_dropped,
                    "last_error": w.state.last_error,
                    "last_text": w.state.last_text,
                }
                for cid, w in self._workers.items()
            },
        }

    async def sync(self, channels: list[dict], settings: TranscriptionSettings) -> None:
        """Start/stop workers so they match the set of tracked channels."""
        self._settings = settings
        if not settings.enabled:
            await self.stop_all()
            return

        missing = _missing_binaries()
        if missing:
            self._last_error = f"必要なコマンドが見つかりません: {', '.join(missing)}"
            log.warn(log.CAT_TRANSCRIPT, self._last_error)
            await self.stop_all()
            return

        wanted = {int(c["id"]): str(c["login"]) for c in channels}
        for channel_id in list(self._workers):
            if channel_id not in wanted:
                await self._stop_worker(channel_id)
        for channel_id, login in wanted.items():
            if channel_id not in self._workers:
                self._start_worker(channel_id, login)

    async def stop_all(self) -> None:
        for channel_id in list(self._workers):
            await self._stop_worker(channel_id)

    def _start_worker(self, channel_id: int, login: str) -> None:
        state = ChannelState(login=login, running=True)
        queue: asyncio.Queue = asyncio.Queue(maxsize=2)
        task = asyncio.create_task(
            self._run_channel(channel_id, login, state, queue),
            name=f"transcribe-{login}",
        )
        self._workers[channel_id] = _Worker(channel_id, login, task, state, queue)

    async def _stop_worker(self, channel_id: int) -> None:
        worker = self._workers.pop(channel_id, None)
        if worker is None:
            return
        worker.task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker.task

    # -- model -------------------------------------------------------------

    async def _ensure_model(self) -> Any:
        s = self._settings
        key = (s.model_size, s.device, s.compute_type)
        async with self._model_lock:
            if self._model is not None and self._model_key == key:
                return self._model
            from faster_whisper import WhisperModel

            log.info(
                log.CAT_TRANSCRIPT,
                f"Whisper モデルを読み込みます ({s.model_size} / {s.device} / {s.compute_type})",
            )
            self._model = await asyncio.to_thread(
                WhisperModel,
                s.model_size,
                device=s.device,
                compute_type=s.compute_type,
                download_root=str(WHISPER_CACHE),
            )
            self._model_key = key
            log.info(log.CAT_TRANSCRIPT, "Whisper モデルの読み込みが完了しました")
            return self._model

    # -- per-channel pipeline ---------------------------------------------

    async def _run_channel(
        self, channel_id: int, login: str, state: ChannelState, queue: asyncio.Queue
    ) -> None:
        consumer = asyncio.create_task(
            self._consume(channel_id, state, queue), name=f"whisper-{login}"
        )
        try:
            while True:
                try:
                    url = await self._resolve_stream(login)
                except Exception as exc:  # noqa: BLE001
                    state.last_error = str(exc)
                    await asyncio.sleep(ERROR_RETRY_SEC)
                    continue

                if not url:
                    if state.live:
                        log.info(log.CAT_TRANSCRIPT, "配信がオフラインになりました",
                                 channel=login)
                    state.live = False
                    await asyncio.sleep(OFFLINE_RETRY_SEC)
                    continue

                if not state.live:
                    log.info(log.CAT_TRANSCRIPT, "音声の取り込みを開始します", channel=login)
                state.live = True
                state.last_error = None
                try:
                    await self._pump(login, url, queue, state)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    state.last_error = f"{type(exc).__name__}: {exc}"
                    log.warn(log.CAT_TRANSCRIPT, f"音声取り込みが中断しました: {exc}",
                             channel=login)
                await asyncio.sleep(ERROR_RETRY_SEC)
        finally:
            state.running = False
            consumer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await consumer

    async def _resolve_stream(self, login: str) -> str | None:
        args = [
            sys.executable, "-m", "streamlink",
            "--stream-url",
            "--twitch-low-latency",
        ]
        from .. import config as _config

        token = _config.load().twitch.oauth_token.strip()
        if token:
            for prefix in ("OAuth ", "oauth:", "Bearer "):
                if token.lower().startswith(prefix.lower()):
                    token = token[len(prefix):].strip()
            # Lets streamlink open subscriber-only streams with the same account.
            args += ["--twitch-api-header", f"Authorization=OAuth {token}"]
        args += [f"https://www.twitch.tv/{login}", "audio_only,worst"]

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=45)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            raise RuntimeError("streamlink が応答しませんでした") from None

        if proc.returncode != 0:
            message = stderr.decode("utf-8", "replace").strip()
            if "No playable streams found" in message or "offline" in message.lower():
                return None
            raise RuntimeError(message.splitlines()[-1] if message else "streamlink が失敗しました")
        url = stdout.decode("utf-8", "replace").strip()
        return url or None

    async def _pump(
        self, login: str, url: str, queue: asyncio.Queue, state: ChannelState
    ) -> None:
        chunk_bytes = SAMPLE_RATE * BYTES_PER_SAMPLE * self._settings.chunk_sec
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
            "-i", url,
            "-vn", "-f", "s16le", "-acodec", "pcm_s16le",
            "-ac", "1", "-ar", str(SAMPLE_RATE),
            "pipe:1",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        buf = bytearray()
        try:
            assert proc.stdout is not None
            while True:
                data = await proc.stdout.read(READ_SIZE)
                if not data:
                    break
                buf.extend(data)
                while len(buf) >= chunk_bytes:
                    chunk = bytes(buf[:chunk_bytes])
                    del buf[:chunk_bytes]
                    if queue.full():
                        with contextlib.suppress(asyncio.QueueEmpty):
                            queue.get_nowait()
                        state.chunks_dropped += 1
                    queue.put_nowait(chunk)
        finally:
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(proc.wait(), timeout=5)

    async def _consume(
        self, channel_id: int, state: ChannelState, queue: asyncio.Queue
    ) -> None:
        import numpy as np

        while True:
            chunk = await queue.get()
            try:
                model = await self._ensure_model()
            except Exception as exc:  # noqa: BLE001
                self._last_error = f"{type(exc).__name__}: {exc}"
                log.error("Whisper モデルの読み込みに失敗しました",
                          channel=state.login, exc=exc, category=log.CAT_TRANSCRIPT)
                await asyncio.sleep(30)
                continue

            audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            language = None if self._settings.language == "auto" else self._settings.language
            try:
                async with self._infer_lock:
                    text = await asyncio.to_thread(
                        _transcribe, model, audio, language, self._settings.beam_size
                    )
            except Exception as exc:  # noqa: BLE001
                state.last_error = f"{type(exc).__name__}: {exc}"
                log.error("文字起こしに失敗しました", channel=state.login, exc=exc,
                          category=log.CAT_TRANSCRIPT)
                continue

            state.chunks_done += 1
            if text.strip():
                state.last_text = text.strip()
                store.add_transcript(channel_id, text)


def _transcribe(model: Any, audio: Any, language: str | None, beam_size: int) -> str:
    segments, _info = model.transcribe(
        audio,
        language=language,
        beam_size=beam_size,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    return "".join(segment.text for segment in segments).strip()


def _missing_binaries() -> list[str]:
    missing = []
    if shutil.which("ffmpeg") is None:
        missing.append("ffmpeg")
    try:
        import streamlink  # noqa: F401
    except ImportError:
        missing.append("streamlink")
    return missing
