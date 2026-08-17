"""Filesystem layout for the application."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("TWITCH_AUTOBET_DATA", ROOT / "data"))
WEB_DIR = ROOT / "web"

SETTINGS_FILE = DATA_DIR / "settings.json"
DB_FILE = DATA_DIR / "autobet.db"
WHISPER_CACHE = DATA_DIR / "whisper-models"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    WHISPER_CACHE.mkdir(parents=True, exist_ok=True)
