"""Supervises the local llama-server process.

Started when tracking starts, stopped when tracking stops. stdout/stderr are
tailed into a ring buffer because a bad `-ngl` or a mismatched GGUF shows up
there and nowhere else.
"""

from __future__ import annotations

import asyncio
import contextlib
import shlex
import sys
from collections import deque
from pathlib import Path

import httpx

from .. import log
from ..config import LlamaSettings


class LlamaServer:
    def __init__(self) -> None:
        self._proc: asyncio.subprocess.Process | None = None
        self._settings: LlamaSettings | None = None
        self._output: deque[str] = deque(maxlen=400)
        self._reader: asyncio.Task | None = None
        self._ready = False

    @property
    def base_url(self) -> str:
        s = self._settings
        if s is None:
            return ""
        host = "127.0.0.1" if s.host in ("0.0.0.0", "") else s.host
        return f"http://{host}:{s.port}"

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    @property
    def status(self) -> dict:
        return {
            "running": self.running,
            "ready": self._ready,
            "base_url": self.base_url,
            "pid": self._proc.pid if self._proc else None,
            "output": list(self._output)[-60:],
        }

    def build_args(self, s: LlamaSettings) -> list[str]:
        server = s.server_path.strip()
        args = [
            server,
            "-m", s.model_path.strip(),
            "-c", str(s.ctx_size),
            "-ngl", str(s.n_gpu_layers),
            "-t", str(s.threads),
            "-b", str(s.batch_size),
            "--cache-type-k", s.cache_type_k,
            "--cache-type-v", s.cache_type_v,
            "--host", s.host,
            "--port", str(s.port),
        ]
        if s.api_key.strip():
            args += ["--api-key", s.api_key.strip()]
        if s.lora_path.strip():
            if abs(s.lora_scale - 1.0) < 1e-9:
                args += ["--lora", s.lora_path.strip()]
            else:
                args += ["--lora-scaled", s.lora_path.strip(), str(s.lora_scale)]
        if s.extra_args.strip():
            args += shlex.split(s.extra_args.strip(), posix=False)
        return args

    def validate(self, s: LlamaSettings) -> list[str]:
        problems: list[str] = []
        if not s.server_path.strip():
            problems.append("llama-server の実行ファイルパスが未設定です")
        elif not Path(s.server_path).exists():
            problems.append(f"llama-server が見つかりません: {s.server_path}")
        if not s.model_path.strip():
            problems.append("モデル (GGUF) のパスが未設定です")
        elif not Path(s.model_path).exists():
            problems.append(f"モデルが見つかりません: {s.model_path}")
        if s.lora_path.strip() and not Path(s.lora_path).exists():
            problems.append(f"LoRA が見つかりません: {s.lora_path}")
        return problems

    async def start(self, s: LlamaSettings) -> None:
        if self.running:
            return
        problems = self.validate(s)
        if problems:
            raise RuntimeError("; ".join(problems))

        self._settings = s
        self._ready = False
        self._output.clear()
        args = self.build_args(s)
        log.info(
            log.CAT_SYSTEM,
            "llama-server を起動します",
            detail={"args": _redact(args, s.api_key)},
        )

        creationflags = 0
        if sys.platform == "win32":
            # Keep Ctrl-C in the parent console from tearing down the child.
            creationflags = getattr(asyncio.subprocess, "CREATE_NO_WINDOW", 0x08000000)

        self._proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            creationflags=creationflags,
        )
        self._reader = asyncio.create_task(self._pump(), name="llama-output")

        try:
            await self._await_health(s.startup_timeout_sec)
        except Exception:
            await self.stop()
            raise
        self._ready = True
        log.info(log.CAT_SYSTEM, f"llama-server が起動しました ({self.base_url})")

    async def _pump(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        while True:
            line = await self._proc.stdout.readline()
            if not line:
                break
            self._output.append(line.decode("utf-8", "replace").rstrip())

    async def _await_health(self, timeout: float) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        headers = {}
        if self._settings and self._settings.api_key.strip():
            headers["Authorization"] = f"Bearer {self._settings.api_key.strip()}"
        async with httpx.AsyncClient(timeout=5.0) as client:
            while asyncio.get_running_loop().time() < deadline:
                if self._proc is not None and self._proc.returncode is not None:
                    tail = "\n".join(list(self._output)[-15:])
                    raise RuntimeError(
                        f"llama-server が終了コード {self._proc.returncode} で終了しました\n{tail}"
                    )
                try:
                    resp = await client.get(f"{self.base_url}/health", headers=headers)
                    if resp.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(1.0)
        tail = "\n".join(list(self._output)[-15:])
        raise RuntimeError(
            f"llama-server が {timeout} 秒以内に応答しませんでした\n{tail}"
        )

    async def stop(self) -> None:
        proc, self._proc = self._proc, None
        self._ready = False
        if self._reader is not None:
            self._reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader
            self._reader = None
        if proc is None or proc.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(proc.wait(), timeout=5.0)
        log.info(log.CAT_SYSTEM, "llama-server を停止しました")


def _redact(args: list[str], api_key: str) -> list[str]:
    if not api_key.strip():
        return args
    return ["***" if a == api_key.strip() else a for a in args]
