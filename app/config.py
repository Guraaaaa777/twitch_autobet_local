"""Application settings: schema, persistence, and the in-process singleton.

Settings live in a single JSON file so they can be inspected and hand-edited.
Secrets (Twitch OAuth token, llama-server API key) are stored in that file in
plain text -- it never leaves the machine, but keep the data directory private.
"""

from __future__ import annotations

import json
import threading
from typing import Literal

from pydantic import BaseModel, Field

from .paths import SETTINGS_FILE, ensure_dirs

CacheType = Literal["f32", "f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0"]


class LlamaSettings(BaseModel):
    """llama.cpp (llama-server) launch configuration."""

    server_path: str = ""
    """Absolute path to llama-server.exe. Required before tracking can start."""

    model_path: str = ""
    """Absolute path to the GGUF model."""

    ctx_size: int = Field(default=8192, ge=512, le=1_048_576)
    """--ctx-size. Must be large enough for the transcript window."""

    n_gpu_layers: int = Field(default=0, ge=0, le=999)
    """--n-gpu-layers ("総数"). 0 = pure CPU."""

    threads: int = Field(default=8, ge=1, le=256)
    """--threads."""

    batch_size: int = Field(default=512, ge=1, le=65536)
    """--batch-size."""

    cache_type_k: CacheType = "f16"
    cache_type_v: CacheType = "f16"
    """--cache-type-k / --cache-type-v ("量子化タイプ")."""

    lora_path: str = ""
    """--lora. Empty = no adapter."""

    lora_scale: float = Field(default=1.0, ge=0.0, le=10.0)

    host: str = "127.0.0.1"
    port: int = Field(default=8081, ge=1, le=65535)

    api_key: str = ""
    """--api-key for llama-server. Empty = no auth."""

    extra_args: str = ""
    """Raw extra CLI arguments, split with shlex."""

    startup_timeout_sec: int = Field(default=180, ge=10, le=1800)
    request_timeout_sec: int = Field(default=180, ge=5, le=1800)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=768, ge=64, le=32768)

    history_limit: int = Field(default=12, ge=0, le=100)
    """How many past resolved predictions of the channel to put in the prompt."""


class TwitchSettings(BaseModel):
    """Credentials for the private GraphQL API used by the Twitch web client."""

    oauth_token: str = ""
    """Value of the `auth-token` cookie on twitch.tv. Sent as `Authorization: OAuth ...`."""

    client_id: str = "kimne78kx3ncx6brgo4mv6wki5h1ko"
    """Public web-client Client-Id. Only change if Twitch rotates it."""

    device_id: str = ""
    """Optional X-Device-Id header; some endpoints are happier with one set."""

    use_pubsub: bool = True
    """Listen on pubsub-edge for instant prediction events (polling still runs)."""

    request_timeout_sec: int = Field(default=20, ge=5, le=120)


class TranscriptionSettings(BaseModel):
    enabled: bool = True

    retention_min: int = Field(default=30, ge=1, le=600)
    """保持時間: transcript lines older than this are pruned."""

    model_size: str = "base"
    """faster-whisper model name or a local CTranslate2 model directory."""

    language: str = "ja"
    """ISO code, or "auto" to let Whisper detect."""

    device: Literal["cpu", "cuda", "auto"] = "cpu"
    compute_type: str = "int8"
    beam_size: int = Field(default=1, ge=1, le=10)

    chunk_sec: int = Field(default=15, ge=5, le=60)
    """Audio buffered per transcription call."""

    prompt_chars: int = Field(default=6000, ge=0, le=100_000)
    """Transcript characters handed to the LLM (most recent wins)."""


class BettingSettings(BaseModel):
    dry_run: bool = True
    """When true, everything runs except the MakePrediction mutation."""

    kelly_fraction: float = Field(default=0.25, gt=0.0, le=1.0)
    max_bet_ratio: float = Field(default=0.05, gt=0.0, le=1.0)
    """Cap as a fraction of the current channel-point balance."""

    max_bet_points: int = Field(default=250_000, ge=1, le=250_000)
    """Absolute cap. Twitch itself rejects more than 250,000 per prediction."""

    min_bet_points: int = Field(default=10, ge=1, le=250_000)
    """Below this the bet is skipped rather than rounded up."""

    min_edge: float = Field(default=0.05, ge=0.0, le=10.0)
    """Required expected return per point staked, e.g. 0.05 = +5%."""

    min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    """Skip if the winning outcome's estimated probability is below this."""

    bet_lead_sec: int = Field(default=5, ge=1, le=120)
    """Fire the bet this many seconds before the prediction locks."""


class Settings(BaseModel):
    llama: LlamaSettings = Field(default_factory=LlamaSettings)
    twitch: TwitchSettings = Field(default_factory=TwitchSettings)
    transcription: TranscriptionSettings = Field(default_factory=TranscriptionSettings)
    betting: BettingSettings = Field(default_factory=BettingSettings)

    poll_rate_sec: float = Field(default=5.0, ge=1.0, le=300.0)
    """ポーリングレート: how often each tracked channel is polled for prediction state."""

    points_poll_sec: float = Field(default=60.0, ge=5.0, le=3600.0)
    """How often the channel-point balance is sampled for the trend graph."""

    log_retention_days: int = Field(default=30, ge=1, le=3650)


_lock = threading.RLock()
_cached: Settings | None = None


def load() -> Settings:
    """Read settings from disk, falling back to defaults."""
    global _cached
    with _lock:
        if _cached is not None:
            return _cached
        ensure_dirs()
        if SETTINGS_FILE.exists():
            try:
                raw = json.loads(SETTINGS_FILE.read_text("utf-8"))
                _cached = Settings.model_validate(raw)
            except Exception:
                # A corrupt file must not brick startup; keep it for inspection.
                SETTINGS_FILE.replace(SETTINGS_FILE.with_suffix(".json.bad"))
                _cached = Settings()
        else:
            _cached = Settings()
            _write(_cached)
        return _cached


def save(settings: Settings) -> Settings:
    global _cached
    with _lock:
        _write(settings)
        _cached = settings
        return _cached


def update(patch: dict) -> Settings:
    """Deep-merge a partial dict into the current settings and persist."""
    current = load().model_dump()
    merged = _deep_merge(current, patch)
    return save(Settings.model_validate(merged))


def _deep_merge(base: dict, patch: dict) -> dict:
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _write(settings: Settings) -> None:
    ensure_dirs()
    tmp = SETTINGS_FILE.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(settings.model_dump(), ensure_ascii=False, indent=2), "utf-8"
    )
    tmp.replace(SETTINGS_FILE)
