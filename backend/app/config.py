"""AthletiQ Backend Configuration Settings."""

from __future__ import annotations
import os
from pathlib import Path
from pydantic import BaseModel

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_MATCH_ID = "football4"
DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / f"testing_{DEFAULT_MATCH_ID}"

class Settings(BaseModel):
    app_name: str = "AthletiQ API"
    app_version: str = "3.0.0"
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    project_root: Path = PROJECT_ROOT
    outputs_dir: Path = OUTPUTS_DIR
    default_match_id: str = DEFAULT_MATCH_ID
    default_match_output_dir: Path = DEFAULT_OUTPUT_DIR
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*",
    ]

settings = Settings()
