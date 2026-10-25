#!/usr/bin/env python3
"""Run entrypoint for AthletiQ FastAPI Backend."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import uvicorn

# Add project root to sys.path so 'backend' is recognized
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Launch AthletiQ FastAPI Backend")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true", default=False, help="Enable auto-reload")
    args = parser.parse_args()

    print(f"\n========================================================")
    print(f"  ATHLETIQ API SERVER STARTING ON http://{args.host}:{args.port}")
    print(f"  Interactive OpenAPI Docs: http://{args.host}:{args.port}/docs")
    print(f"========================================================\n")

    uvicorn.run(
        "backend.app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
