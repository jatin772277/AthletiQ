"""Background execution of the AthletiQ.py pipeline for uploaded match videos."""

from __future__ import annotations
import json
import os
import platform
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from backend.app import db
from backend.app.config import settings

# stdout markers printed by AthletiQ.py -> (progress %, label shown to the user)
PHASES: list[tuple[str, float, str]] = [
    ("[PHASE 1 & 2]", 2, "Detecting & tracking players and ball"),
    ("[PHASE 2.5]", 62, "Stitching player identities"),
    ("[PHASE 4 & 5]", 70, "Calibrating pitch & clipping outliers"),
    ("[PHASE 6]", 78, "Computing spatial zone occupancy"),
    ("[PHASE 7]", 82, "Analysing ball possession & touches"),
    ("[PHASE 8]", 86, "Building player interaction network"),
    ("[PHASE 9]", 90, "Ranking players with the DSA engine"),
    ("[PHASE 10]", 94, "Scoring performance with the ML model"),
    ("[PHASE 11]", 97, "Exporting the analysis package"),
    ("[ENHANCE]", 98, "Consolidating collective metrics"),
]
_FRAMES = re.compile(r"Processed (\d+)/(\d+) frames")
_TRACK_START, _TRACK_SPAN = 2.0, 58.0
LOG_TAIL = 40


def job_paths(job: dict[str, Any]) -> tuple[Path, Path, Path]:
    """(video, output dir, log file) for a job."""
    user_dir = settings.uploads_dir / str(job["owner_id"])
    return user_dir / job["video_filename"], settings.user_outputs_dir / job["_id"], user_dir / f"{job['_id']}.log"


def to_wsl_path(path: Path) -> str:
    p = path.resolve()
    drive = p.drive
    if not drive or not drive[0].isalpha() or drive not in (f"{drive[0]}:", f"{drive[0]}:\\"):
        raise ValueError(
            f"to_wsl_path only supports drive-letter Windows paths (e.g. D:\\...), got: {p!r}"
        )
    letter = drive[0].lower()
    return f"/mnt/{letter}" + p.as_posix()[len(p.drive):]


def build_commands(job: dict[str, Any]) -> list[list[str]]:
    """Build native Linux/CPU commands or the existing Windows/WSL development commands."""
    video, out_dir, _ = job_paths(job)
    wsl = settings.pipeline_mode == "wsl"
    if wsl and platform.system() != "Windows":
        raise RuntimeError("PIPELINE_MODE=wsl is only supported on Windows; use PIPELINE_MODE=local on Linux")
    path = to_wsl_path if wsl else str
    device = settings.pipeline_device or ("0" if wsl else "cpu")
    steps = [
        [path(settings.project_root / "AthletiQ.py"), "--video", path(video), "--output-dir", path(out_dir), "--device", device,
         "--start-frame", str(job["start_frame"]), "--max-frames", str(job["max_frames"]), "--clean", "--force-retrack"],
        [path(settings.project_root / "ai-engine" / "scripts" / "apply_collective_improvements.py"), "--output-dir", path(out_dir),
         "--stem", job["_id"], "--fps", str(job["video"]["fps"]), "--frames", str(job["max_frames"]),
         "--video-filename", job.get("original_filename") or job["video_filename"]],
    ]
    if wsl:
        activate = f"source {settings.pipeline_wsl_venv}/bin/activate && python3 -u "
        return [["wsl.exe", "bash", "-lc", activate + " ".join(shlex.quote(a) for a in args)] for args in steps]
    python = settings.pipeline_python or sys.executable
    return [[python, "-u", *args] for args in steps]


def parse_progress(line: str) -> Optional[tuple[float, str]]:
    m = _FRAMES.search(line)
    if m:
        done, total = int(m.group(1)), max(int(m.group(2)), 1)
        return _TRACK_START + _TRACK_SPAN * min(done / total, 1.0), f"Tracking players · frame {done}/{total}"
    for marker, pct, label in PHASES:
        if marker in line:
            return pct, label
    return None


def outputs_ready(out_dir: Path, stem: str) -> bool:
    """AthletiQ.py writes an older RAG schema; only the enhanced v3 document is servable.

    Also validates that the document's own video_id matches the expected stem so a
    stale output from a previous run cannot cause a false-positive completion.
    """
    rag = out_dir / "rag_ready" / f"{stem}_match_and_player_metrics_for_rag.json"
    try:
        doc = json.loads(rag.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    cards = doc.get("player_cards") or []
    meta = doc.get("metadata", {})
    if meta.get("schema_version") != "3.0":
        return False
    if not cards or "collective_metrics" not in cards[0]:
        return False
    # Guard against stale outputs from a previous run with a different job id.
    doc_stem = meta.get("video_id", "") or meta.get("video_filename", "")
    if doc_stem and not (doc_stem == stem or doc_stem.startswith(stem)):
        return False
    return True


def _now() -> datetime:
    return datetime.now(timezone.utc)


class PipelineRunner:
    """Runs one pipeline job at a time on a daemon worker thread."""

    def __init__(self) -> None:
        self._queue: "queue.Queue[Optional[str]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._procs: dict[str, subprocess.Popen] = {}
        self._cancelled: set[str] = set()
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._recover()
        self._thread = threading.Thread(target=self._work, name="athletiq-pipeline", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        with self._lock:
            for proc in self._procs.values():
                proc.terminate()
        self._queue.put(None)

    def submit(self, job_id: str) -> None:
        self._queue.put(job_id)

    def cancel(self, job_id: str) -> bool:
        """Cancel a queued or running job. Returns False if it had already finished."""
        res = db.jobs().update_one(
            {"_id": job_id, "status": "queued"},
            {"$set": {"status": "cancelled", "phase": "Cancelled", "finished_at": _now()}},
        )
        if res.modified_count:
            return True
        if not db.jobs().find_one({"_id": job_id, "status": "running"}, {"_id": 1}):
            return False
        with self._lock:
            # The process may not be registered yet; _run checks this flag once it is.
            self._cancelled.add(job_id)
            proc = self._procs.get(job_id)
            if proc is not None:
                proc.terminate()
        return True

    def _recover(self) -> None:
        """After a restart, fail jobs that were mid-run and re-queue waiting ones."""
        db.jobs().update_many(
            {"status": "running"},
            {"$set": {"status": "failed", "error": "Interrupted because the server restarted", "finished_at": _now()}},
        )
        for job in db.jobs().find({"status": "queued"}, {"_id": 1}).sort("created_at", 1):
            self._queue.put(job["_id"])

    def _work(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is None:
                return
            try:
                self._run(job_id)
            except Exception as e:  # keep the worker alive whatever happens to one job
                db.jobs().update_one({"_id": job_id}, {"$set": {"status": "failed", "error": f"Runner error: {e}", "finished_at": _now()}})

    def _run(self, job_id: str) -> None:
        job = db.jobs().find_one_and_update(
            {"_id": job_id, "status": "queued"},
            {"$set": {"status": "running", "started_at": _now(), "progress": 1, "phase": "Starting pipeline"}},
        )
        if not job:
            return  # cancelled or deleted while waiting
        video, out_dir, log_path = job_paths(job)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "PYTHONIOENCODING": "utf-8",
            # The pipeline resolves all model paths from this configurable directory.
            "ATHLETIQ_MODEL_ASSETS_DIR": str(settings.model_assets_dir),
        }
        tail: list[str] = []
        progress, phase = 1.0, "Starting pipeline"
        last_write = 0.0

        code = 0
        with open(log_path, "w", encoding="utf-8") as log:
            try:
                for cmd in build_commands(job):
                    proc = subprocess.Popen(
                        cmd, cwd=settings.project_root, env=env,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                    )
                    with self._lock:
                        self._procs[job_id] = proc
                        if job_id in self._cancelled:
                            proc.terminate()
                    assert proc.stdout is not None
                    for raw in proc.stdout:
                        line = raw.rstrip()
                        log.write(raw)
                        if not line.strip():
                            continue
                        tail = (tail + [line])[-LOG_TAIL:]
                        parsed = parse_progress(line)
                        if parsed:
                            progress, phase = max(progress, parsed[0]), parsed[1]
                        if parsed or time.monotonic() - last_write > 2:
                            db.jobs().update_one({"_id": job_id}, {"$set": {"progress": round(progress, 1), "phase": phase, "log_tail": tail}})
                            last_write = time.monotonic()
                    code = proc.wait()
                    if code != 0 or job_id in self._cancelled:
                        break
            finally:
                with self._lock:
                    self._procs.pop(job_id, None)
                    cancelled = job_id in self._cancelled
                    self._cancelled.discard(job_id)

        update: dict[str, Any] = {"log_tail": tail, "finished_at": _now()}
        if cancelled:
            update |= {"status": "cancelled", "phase": "Cancelled"}
        elif code == 0 and outputs_ready(out_dir, job_id):
            update |= {"status": "completed", "progress": 100, "phase": "Analysis complete", "error": None}
        else:
            reason = next((l for l in reversed(tail) if "Error" in l or "error" in l), tail[-1] if tail else "")
            update |= {"status": "failed", "error": f"Pipeline exited with code {code}" + (f": {reason}" if reason else "")}
        db.jobs().update_one({"_id": job_id}, {"$set": update})


def delete_job_files(job: dict[str, Any]) -> None:
    video, out_dir, log_path = job_paths(job)
    for f in (video, log_path):
        f.unlink(missing_ok=True)
    shutil.rmtree(out_dir, ignore_errors=True)


runner = PipelineRunner()
