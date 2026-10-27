"""Tests for accounts, sessions, match access control and video analysis jobs."""

from __future__ import annotations
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services import pipeline_runner
from backend.app.services.pipeline_runner import job_paths, parse_progress, runner, to_wsl_path
from backend.tests.helpers import register

FAKE = Path(__file__).with_name("fake_pipeline.py")


def authed(email: str) -> TestClient:
    c = TestClient(app)
    c.headers["Authorization"] = f"Bearer {register(c, email=email)}"
    return c


@pytest.fixture(scope="module")
def video(tmp_path_factory) -> bytes:
    path = tmp_path_factory.mktemp("vid") / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 48))
    for i in range(50):
        writer.write(np.full((48, 64, 3), i * 5 % 255, dtype=np.uint8))
    writer.release()
    return path.read_bytes()


@pytest.fixture
def fake_mode(monkeypatch):
    """Swap AthletiQ.py for the fake pipeline; returns a setter for its mode."""
    state = {"mode": "ok"}

    def build(job):
        _, out_dir, _ = job_paths(job)
        return [[sys.executable, str(FAKE), state["mode"], job["_id"], str(out_dir)]]

    monkeypatch.setattr(pipeline_runner, "build_commands", build)
    runner.start()
    return lambda m: state.update(mode=m)


def wait_for(client: TestClient, job_id: str, statuses=("completed", "failed", "cancelled"), timeout=30.0) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in statuses:
            return job
        time.sleep(0.2)
    raise AssertionError(f"job {job_id} stuck in {job['status']}")


def upload(client: TestClient, data: bytes, name="match.mp4", **form) -> dict:
    res = client.post("/api/jobs", files={"file": (name, data, "video/mp4")}, data={k: str(v) for k, v in form.items()})
    return res


# ---------------- Auth ----------------

def test_register_login_me_logout():
    c = TestClient(app)
    res = c.post("/api/auth/register", json={"name": "Ana", "email": "Ana@Example.com", "password": "password123"})
    assert res.status_code == 201
    body = res.json()
    assert body["user"]["email"] == "ana@example.com"
    assert "password_hash" not in body["user"]
    token = body["access_token"]

    assert c.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"}).json()["name"] == "Ana"

    login = c.post("/api/auth/login", json={"email": "ANA@example.com", "password": "password123"})
    assert login.status_code == 200
    second = login.json()["access_token"]

    assert c.post("/api/auth/logout", headers={"Authorization": f"Bearer {token}"}).status_code == 204
    # The signed-out session is revoked, other sessions keep working.
    assert c.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401
    assert c.get("/api/auth/me", headers={"Authorization": f"Bearer {second}"}).status_code == 200


def test_register_validation_and_duplicates():
    c = TestClient(app)
    ok = {"name": "Dup", "email": "dup@example.com", "password": "password123"}
    assert c.post("/api/auth/register", json=ok).status_code == 201
    assert c.post("/api/auth/register", json={**ok, "email": "DUP@example.com"}).status_code == 409
    assert c.post("/api/auth/register", json={**ok, "email": "not-an-email"}).status_code == 422
    assert c.post("/api/auth/register", json={**ok, "email": "x@example.com", "password": "short"}).status_code == 422


def test_wrong_password_and_unknown_user():
    c = TestClient(app)
    c.post("/api/auth/register", json={"name": "W", "email": "w@example.com", "password": "password123"})
    assert c.post("/api/auth/login", json={"email": "w@example.com", "password": "nope-nope"}).status_code == 401
    assert c.post("/api/auth/login", json={"email": "ghost@example.com", "password": "password123"}).status_code == 401


def test_match_routes_require_sign_in():
    c = TestClient(app)
    assert c.get("/api/matches").status_code == 401
    assert c.get("/api/matches/football4/summary").status_code == 401
    assert c.get("/api/matches/football4/summary", headers={"Authorization": "Bearer garbage"}).status_code == 401
    assert c.get("/api/health").status_code == 200


def test_unknown_match_is_404_not_demo_data():
    c = authed("lookup@example.com")
    assert c.get("/api/matches/doesnotexist/summary").status_code == 404
    assert c.get("/api/matches/football4/summary").status_code == 200


# ---------------- Upload validation ----------------

def test_upload_rejects_bad_files(video):
    c = authed("uploader@example.com")
    assert upload(c, b"hello", name="notes.txt").status_code == 415
    assert upload(c, b"not really a video", name="fake.mp4").status_code == 400
    assert c.get("/api/jobs").json() == []


def test_upload_computes_frame_window(video, fake_mode):
    c = authed("window@example.com")
    res = upload(c, video, title="Cup final", start_seconds=0.4, duration_seconds=1.0)
    assert res.status_code == 202, res.text
    job = res.json()
    assert job["title"] == "Cup final"
    assert job["video"]["fps"] == pytest.approx(25, abs=0.5)
    assert job["start_frame"] == 10
    assert job["max_frames"] == 25
    wait_for(c, job["id"])


# ---------------- Pipeline jobs ----------------

def test_job_completes_and_is_private(video, fake_mode):
    owner = authed("owner@example.com")
    other = authed("other@example.com")
    job = upload(owner, video).json()
    done = wait_for(owner, job["id"])
    assert done["status"] == "completed", done
    assert done["progress"] == 100

    listed = {m["match_id"]: m for m in owner.get("/api/matches").json()}
    assert listed[job["id"]]["has_rag_package"] is True
    summary = owner.get(f"/api/matches/{job['id']}/summary")
    assert summary.status_code == 200
    assert summary.json()["total_active_players"] == 32

    # Another user can neither see the job nor its match data.
    assert other.get(f"/api/jobs/{job['id']}").status_code == 404
    assert other.get(f"/api/matches/{job['id']}/summary").status_code == 404
    assert job["id"] not in {m["match_id"] for m in other.get("/api/matches").json()}


def test_failed_job_reports_error(video, fake_mode):
    fake_mode("fail")
    c = authed("fail@example.com")
    job = upload(c, video).json()
    done = wait_for(c, job["id"])
    assert done["status"] == "failed"
    assert "code 1" in done["error"]
    assert "Could not open video" in done["error"]
    # Failed jobs never expose match data.
    assert c.get(f"/api/matches/{job['id']}/summary").status_code == 404


def test_unenhanced_outputs_are_not_marked_complete(video, fake_mode):
    # AthletiQ.py alone writes an older RAG schema the API can't serve.
    fake_mode("oldschema")
    c = authed("oldschema@example.com")
    job = upload(c, video).json()
    done = wait_for(c, job["id"])
    assert done["status"] == "failed"
    assert c.get(f"/api/matches/{job['id']}/summary").status_code == 404


def test_cancel_and_delete(video, fake_mode):
    fake_mode("hang")
    c = authed("cancel@example.com")
    job = upload(c, video).json()
    wait_for(c, job["id"], statuses=("running",))
    assert c.delete(f"/api/jobs/{job['id']}").status_code == 409
    assert c.post(f"/api/jobs/{job['id']}/cancel").status_code == 200
    done = wait_for(c, job["id"])
    assert done["status"] == "cancelled"

    stored_video = job_paths({"_id": job["id"], "owner_id": c.get("/api/auth/me").json()["id"], "video_filename": f"{job['id']}.mp4"})[0]
    assert stored_video.exists()
    assert c.delete(f"/api/jobs/{job['id']}").status_code == 204
    assert not stored_video.exists()
    assert c.get(f"/api/jobs/{job['id']}").status_code == 404


# ---------------- Runner helpers ----------------

def test_parse_progress():
    assert parse_progress("  Processed 375/750 frames (50%)...") == (31.0, "Tracking players · frame 375/750")
    assert parse_progress("--- [PHASE 9] Running Unified DSA Engine ---")[0] == 90
    assert parse_progress("--- [PHASE 10] Running ML ---")[0] == 94
    assert parse_progress("random log line") is None


def test_to_wsl_path():
    assert to_wsl_path(Path("D:/My Projects/AthletiQ/x.mp4")) == "/mnt/d/My Projects/AthletiQ/x.mp4"
    # UNC paths must raise instead of silently producing a broken /mnt/ path.
    import pytest as _pytest
    with _pytest.raises(ValueError, match="drive-letter"):
        to_wsl_path(Path("//server/share/file.mp4"))


JOB = {"_id": "mabc", "owner_id": "u1", "video_filename": "mabc.mp4", "original_filename": "Cup Final.mp4",
       "start_frame": 10, "max_frames": 250, "video": {"fps": 25.0}}


def test_build_commands_local(monkeypatch):
    monkeypatch.setattr(pipeline_runner.settings, "pipeline_mode", "local")
    pipeline, enhance = pipeline_runner.build_commands(JOB)
    assert pipeline[2].endswith("AthletiQ.py")
    assert pipeline[pipeline.index("--start-frame") + 1] == "10"
    assert pipeline[pipeline.index("--max-frames") + 1] == "250"
    assert pipeline[pipeline.index("--device") + 1] == "cpu"
    assert enhance[2].endswith("apply_collective_improvements.py")
    assert enhance[enhance.index("--stem") + 1] == "mabc"
    assert enhance[enhance.index("--fps") + 1] == "25.0"
    assert enhance[enhance.index("--video-filename") + 1] == "Cup Final.mp4"


def test_build_commands_wsl_quotes_paths(monkeypatch):
    monkeypatch.setattr(pipeline_runner.settings, "pipeline_mode", "wsl")
    pipeline, enhance = pipeline_runner.build_commands(JOB)
    assert pipeline[:3] == ["wsl.exe", "bash", "-lc"]
    assert pipeline[3].startswith("source ~/tf312/bin/activate && python3 -u ")
    assert "--device 0" in pipeline[3]
    assert "'Cup Final.mp4'" in enhance[3]  # spaces survive the shell
