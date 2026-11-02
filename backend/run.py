#!/usr/bin/env python3
"""Run entrypoint for AthletiQ FastAPI Backend."""

from __future__ import annotations
import argparse
import multiprocessing
import sys
from pathlib import Path
import uvicorn

# Add project root to sys.path so 'backend' is recognized
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    from backend.app.config import settings

    default_workers = 1

    parser = argparse.ArgumentParser(description="Launch AthletiQ FastAPI Backend")
    parser.add_argument("--host", default=settings.host, help=f"Host address (default: {settings.host})")
    parser.add_argument("--port", type=int, default=settings.port, help=f"Port (default: {settings.port})")
    parser.add_argument("--reload", action="store_true", default=False,
                        help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=default_workers,
                        help=f"Number of uvicorn worker processes (default: {default_workers})")
    args = parser.parse_args()

    workers = 1 if args.reload else args.workers

    print(f"\n========================================================")
    print(f"  ATHLETIQ API SERVER  →  http://{args.host}:{args.port}")
    print(f"  Workers : {workers}")
    print(f"  Docs    : http://{args.host}:{args.port}/docs")
    print(f"========================================================\n")

    uvicorn.run(
        "backend.app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=workers,
        log_level="info",
        timeout_keep_alive=120,   # keep long-polling / SSE connections alive
    )


if __name__ == "__main__":
    main()
