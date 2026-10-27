"""Test configuration: in-memory MongoDB and throwaway upload/output folders."""

from __future__ import annotations
import os
import sys
import tempfile
from pathlib import Path

# Must be set before backend.app.config is imported.
_TMP = Path(tempfile.mkdtemp(prefix="athletiq_tests_"))
os.environ["MONGODB_URI"] = "mongomock://"
os.environ["MONGODB_DB"] = "athletiq_test"
os.environ["UPLOADS_DIR"] = str(_TMP / "uploads")
os.environ["USER_OUTPUTS_DIR"] = str(_TMP / "user_matches")
os.environ.setdefault("JWT_SECRET", "test-secret-not-for-production-use-0123456789")
# Unit/integration tests must use the deterministic tactical-answer fallback.
# This prevents test runs from consuming an external provider quota or sending fixtures.
os.environ.pop("GROQ_API_KEY", None)
os.environ["ATHLETIQ_DISABLE_EXTERNAL_LLM"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app import db  # noqa: E402

db.ensure_indexes()

