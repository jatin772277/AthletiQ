"""AthletiQ Backend Configuration Settings."""

from __future__ import annotations
import os
import secrets
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv
from pydantic import BaseModel

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_MATCH_ID = "football4"
DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / f"testing_{DEFAULT_MATCH_ID}"
LOCAL_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

load_dotenv(PROJECT_ROOT / ".env")


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    if value.strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if value.strip().lower() in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def _path_env(name: str, default: Path) -> Path:
    return Path(os.environ[name]).expanduser() if os.getenv(name) else default


def _cors_origins() -> list[str]:
    value = os.getenv("CORS_ORIGINS")
    if not value:
        return list(LOCAL_CORS_ORIGINS)
    origins = [origin.strip().rstrip("/") for origin in value.split(",") if origin.strip()]
    if not origins:
        raise ValueError("CORS_ORIGINS must contain at least one origin")
    if "*" in origins:
        raise ValueError("CORS_ORIGINS cannot contain '*' when credentials are enabled")
    for origin in origins:
        parsed = urlparse(origin)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path not in {"", "/"}:
            raise ValueError(f"CORS_ORIGINS contains an invalid origin: {origin!r}")
    return origins


def _mongo_uri() -> str:
    """MONGO_URI is preferred; retain MONGODB_URI for current local setups."""
    return os.getenv("MONGO_URI") or os.getenv("MONGODB_URI", "mongodb://127.0.0.1:27017/AthletiQ")


def _valid_api_base(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc) and parsed.path in {"", "/"}


def _jwt_secret() -> str:
    """Use JWT_SECRET if set; otherwise persist a generated one so sessions survive restarts."""
    if os.getenv("JWT_SECRET"):
        return os.environ["JWT_SECRET"]
    path = BACKEND_DIR / ".jwt_secret"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    value = secrets.token_urlsafe(48)
    path.write_text(value, encoding="utf-8")
    return value


def _mongodb_db() -> str:
    if os.getenv("MONGODB_DB"):
        return os.environ["MONGODB_DB"]
    uri = _mongo_uri()
    if uri and not uri.startswith("mongomock://"):
        try:
            from urllib.parse import urlparse
            path = urlparse(uri).path.lstrip("/")
            if path:
                return path.split("?")[0]
        except Exception:
            pass
    return "athletiq"


class Settings(BaseModel):
    app_name: str = "AthletiQ API"
    app_version: str = "3.0.0"
    app_environment: str = os.getenv("APP_ENV", "development").strip().lower()
    debug: bool = _bool_env("DEBUG", False)
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    project_root: Path = PROJECT_ROOT

    # A single mounted data directory is convenient on EC2. Individual paths win.
    data_dir: Path | None = Path(os.environ["ATHLETIQ_DATA_DIR"]).expanduser() if os.getenv("ATHLETIQ_DATA_DIR") else None
    model_assets_dir: Path = _path_env("MODEL_ASSETS_DIR", PROJECT_ROOT / "ai-engine" / "models")
    uploads_dir: Path = _path_env(
        "UPLOADS_DIR", (data_dir / "uploads") if data_dir else PROJECT_ROOT / "uploads"
    )
    outputs_dir: Path = _path_env(
        "OUTPUTS_DIR", (data_dir / "outputs") if data_dir else OUTPUTS_DIR
    )
    user_outputs_dir: Path = _path_env("USER_OUTPUTS_DIR", outputs_dir / "user_matches")
    default_match_id: str = DEFAULT_MATCH_ID
    default_match_output_dir: Path = outputs_dir / f"testing_{DEFAULT_MATCH_ID}"
    cors_origins: list[str] = _cors_origins()

    # MongoDB ("mongomock://" gives an in-memory database for tests)
    mongodb_uri: str = _mongo_uri()
    mongodb_db: str = _mongodb_db()

    # Auth
    jwt_secret: str = _jwt_secret()
    jwt_algorithm: str = "HS256"
    session_ttl_hours: int = int(os.getenv("SESSION_TTL_HOURS", "72"))

    # Pre-computed matches every signed-in user can browse
    demo_match_ids: list[str] = [m.strip() for m in os.getenv("DEMO_MATCHES", DEFAULT_MATCH_ID).split(",") if m.strip()]

    # Uploads & pipeline jobs
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "2048"))
    allowed_video_ext: list[str] = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"]
    # "local": run AthletiQ.py with pipeline_python; "wsl": run it inside WSL with the GPU venv
    pipeline_mode: str = os.getenv("PIPELINE_MODE", "local").strip().lower()
    pipeline_python: str = os.getenv("PIPELINE_PYTHON", "")
    pipeline_wsl_venv: str = os.getenv("PIPELINE_WSL_VENV", "~/tf312")
    pipeline_device: str = os.getenv("PIPELINE_DEVICE", "").strip()
    pipeline_default_max_frames: int = int(os.getenv("PIPELINE_MAX_FRAMES", "750"))

    def validate_production(self) -> None:
        """Reject unsafe production configuration without including secret values in errors."""
        if self.app_environment not in {"development", "test", "production"}:
            raise ValueError("APP_ENV must be development, test, or production")
        if self.pipeline_mode not in {"local", "wsl"}:
            raise ValueError("PIPELINE_MODE must be 'local' or 'wsl'")
        device = self.pipeline_device or ("0" if self.pipeline_mode == "wsl" else "cpu")
        if device != "cpu" and not device.isdecimal():
            raise ValueError("PIPELINE_DEVICE must be 'cpu' or a non-negative GPU index")
        if self.app_environment != "production":
            return
        if "DEBUG" not in os.environ or self.debug:
            raise ValueError("Production requires DEBUG=false")
        if not os.getenv("JWT_SECRET") or len(self.jwt_secret) < 32 or "change_me" in self.jwt_secret.lower():
            raise ValueError("Production requires a non-placeholder JWT_SECRET of at least 32 characters")
        if not os.getenv("MONGO_URI"):
            raise ValueError("Production requires MONGO_URI (MONGODB_URI remains supported for local development)")
        if not os.getenv("CORS_ORIGINS"):
            raise ValueError("Production requires explicit CORS_ORIGINS")
        if not _valid_api_base(os.getenv("VITE_API_BASE", "")):
            raise ValueError("Production requires VITE_API_BASE as an absolute http(s) API origin")
        if self.pipeline_mode == "wsl":
            raise ValueError("Production Linux deployment requires PIPELINE_MODE=local; wsl is Windows development only")


settings = Settings()
