"""Deployment configuration, CORS, health-readiness, and Linux-runner tests."""

from __future__ import annotations

import pytest

from backend.app.config import _cors_origins, settings
from backend.app.main import app, health_check
from backend.app.services import pipeline_runner
from backend.app.services.pipeline_runner import build_commands


JOB = {
    "_id": "mdeploy",
    "owner_id": "u1",
    "video_filename": "mdeploy.mp4",
    "original_filename": "Linux clip.mp4",
    "start_frame": 0,
    "max_frames": 20,
    "video": {"fps": 25.0},
}


def test_local_cors_defaults_are_explicit_and_no_wildcard(monkeypatch):
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    assert "*" not in _cors_origins()
    assert "http://localhost:5173" in _cors_origins()


def test_credentialed_cors_rejects_wildcard(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "*")
    with pytest.raises(ValueError, match="cannot contain"):
        _cors_origins()


def test_health_reports_database_readiness(monkeypatch):
    monkeypatch.setattr("backend.app.main.db.ping", lambda: False)
    response = health_check()
    assert response["status"] == "degraded"
    assert response["database"] == "unreachable"
    assert response["database_ready"] is False


def test_linux_local_pipeline_never_uses_wsl(monkeypatch):
    monkeypatch.setattr(pipeline_runner.settings, "pipeline_mode", "local")
    monkeypatch.setattr(pipeline_runner.settings, "pipeline_device", "cpu")
    monkeypatch.setattr(pipeline_runner.platform, "system", lambda: "Linux")
    pipeline, enhancer = build_commands(JOB)
    assert pipeline[0] != "wsl.exe"
    assert enhancer[0] != "wsl.exe"
    assert pipeline[pipeline.index("--device") + 1] == "cpu"


def test_linux_rejects_windows_wsl_runner(monkeypatch):
    monkeypatch.setattr(pipeline_runner.settings, "pipeline_mode", "wsl")
    monkeypatch.setattr(pipeline_runner.platform, "system", lambda: "Linux")
    with pytest.raises(RuntimeError, match="only supported on Windows"):
        build_commands(JOB)


def test_production_validation_requires_documented_values(monkeypatch):
    monkeypatch.setattr(settings, "app_environment", "production")
    monkeypatch.setattr(settings, "debug", False)
    monkeypatch.setattr(settings, "pipeline_mode", "local")
    monkeypatch.setattr(settings, "pipeline_device", "cpu")
    monkeypatch.setattr(settings, "jwt_secret", "x" * 48)
    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.setenv("JWT_SECRET", "x" * 48)
    monkeypatch.delenv("MONGO_URI", raising=False)
    with pytest.raises(ValueError, match="MONGO_URI"):
        settings.validate_production()

    monkeypatch.setenv("MONGO_URI", "mongodb://localhost:27017/athletiq")
    monkeypatch.setenv("CORS_ORIGINS", "https://app.example.com")
    monkeypatch.setenv("VITE_API_BASE", "https://api.example.com")
    settings.validate_production()


def test_cors_middleware_uses_explicit_local_origins_only():
    cors = next(middleware for middleware in app.user_middleware if middleware.cls.__name__ == "CORSMiddleware")
    assert cors.kwargs["allow_credentials"] is True
    assert "*" not in cors.kwargs["allow_origins"]
    assert "http://localhost:5173" in cors.kwargs["allow_origins"]
