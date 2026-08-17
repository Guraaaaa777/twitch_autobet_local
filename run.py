"""Entry point: starts the local server and opens the UI.

    python run.py [--host 127.0.0.1] [--port 8760] [--no-browser]
"""

from __future__ import annotations

import argparse
import logging
import threading
import webbrowser

import uvicorn

from app.paths import ensure_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="twitch_autobet")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8760)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    )

    ensure_dirs()
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '') else args.host}:{args.port}/"
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"twitch_autobet -> {url}")

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    main()
