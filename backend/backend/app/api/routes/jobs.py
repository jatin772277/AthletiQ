"""Video upload and pipeline job routes."""

from __future__ import annotations
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from backend.app import db
from backend.app.config import settings
from backend.app.schemas.models import JobOut
from backend.app.security import get_current_user
from backend.app.services.pipeline_runner import delete_job_files, runner

router = APIRouter(prefix="/api/jobs", tags=["Video Analysis Jobs"])

_CHUNK = 1024 * 1024


def _job_out(job: dict[str, Any]) -> dict[str, Any]:
    return {**job, "id": job["_id"]}


def _owned(job_id: str, user: dict[str, Any]) -> dict[str, Any]:
    job = db.jobs().find_one({"_id": job_id, "owner_id": user["_id"]})
    if not job:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return job


def probe_video(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="This file could not be read as a video")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 30.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ok, _ = cap.read()
        if not ok or frames <= 0:
            raise HTTPException(status_code=400, detail="The video has no readable frames")
        return {
            "fps": round(fps, 3),
            "frame_count": frames,
            "duration_seconds": round(frames / fps, 2),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        cap.release()


@router.post("", response_model=JobOut, status_code=202, summary="Upload a match video and queue the full analysis pipeline")
def create_job(
    request: Request,
    file: UploadFile = File(..., description="Match video (mp4, mov, avi, mkv, webm)"),
    title: str = Form("", max_length=120),
    start_seconds: float = Form(0.0, ge=0),
    duration_seconds: Optional[float] = Form(None, gt=0),
    user: dict[str, Any] = Depends(get_current_user),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in settings.allowed_video_ext:
        raise HTTPException(status_code=415, detail=f"Unsupported file type. Use one of: {', '.join(settings.allowed_video_ext)}")
    limit = settings.max_upload_mb * 1024 * 1024
    if int(request.headers.get("content-length") or 0) > limit + _CHUNK:
        raise HTTPException(status_code=413, detail=f"Video is larger than {settings.max_upload_mb} MB")

    job_id = "m" + secrets.token_hex(6)
    user_dir = settings.uploads_dir / str(user["_id"])
    user_dir.mkdir(parents=True, exist_ok=True)
    # The pipeline names every output after the video's stem, so the stem must be the job id.
    video_filename = f"{job_id}{ext}"
    dest = user_dir / video_filename
    size = 0
    try:
        with open(dest, "wb") as out:
            while chunk := file.file.read(_CHUNK):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(status_code=413, detail=f"Video is larger than {settings.max_upload_mb} MB")
                out.write(chunk)
        info = probe_video(dest)
    except BaseException:
        dest.unlink(missing_ok=True)
        raise

    start_frame = min(int(start_seconds * info["fps"]), info["frame_count"] - 1)
    remaining = info["frame_count"] - start_frame
    max_frames = int(duration_seconds * info["fps"]) if duration_seconds else settings.pipeline_default_max_frames
    max_frames = max(1, min(max_frames, remaining))

    job = {
        "_id": job_id,
        "owner_id": user["_id"],
        "title": title.strip() or Path(file.filename or "Match").stem[:120],
        "original_filename": (file.filename or "")[:255],
        "video_filename": video_filename,
        "size_bytes": size,
        "video": info,
        "start_frame": start_frame,
        "max_frames": max_frames,
        "status": "queued",
        "progress": 0,
        "phase": "Waiting in queue",
        "error": None,
        "log_tail": [],
        "created_at": datetime.now(timezone.utc),
        "started_at": None,
        "finished_at": None,
    }
    db.jobs().insert_one(job)
    runner.submit(job_id)
    return _job_out(job)


@router.get("", response_model=list[JobOut], summary="List the signed-in user's analyses")
def list_jobs(user: dict[str, Any] = Depends(get_current_user)):
    return [_job_out(j) for j in db.jobs().find({"owner_id": user["_id"]}).sort("created_at", -1)]


@router.get("/{job_id}", response_model=JobOut, summary="Get analysis status and progress")
def get_job(job_id: str, user: dict[str, Any] = Depends(get_current_user)):
    return _job_out(_owned(job_id, user))


@router.post("/{job_id}/cancel", response_model=JobOut, summary="Cancel a queued or running analysis")
def cancel_job(job_id: str, user: dict[str, Any] = Depends(get_current_user)):
    job = _owned(job_id, user)
    if job["status"] not in ("queued", "running") or not runner.cancel(job_id):
        raise HTTPException(status_code=409, detail=f"Analysis is already {job['status']}")
    return _job_out(_owned(job_id, user))


@router.delete("/{job_id}", status_code=204, summary="Delete an analysis, its video and outputs")
def delete_job(job_id: str, user: dict[str, Any] = Depends(get_current_user)):
    job = _owned(job_id, user)
    if job["status"] in ("queued", "running"):
        raise HTTPException(status_code=409, detail="Cancel the analysis before deleting it")
    delete_job_files(job)
    db.jobs().delete_one({"_id": job_id})
