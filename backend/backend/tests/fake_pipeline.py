"""Stand-in for AthletiQ.py in tests.

Usage: fake_pipeline.py <mode> <job_id> <output_dir>
  ok     - prints phase markers and copies the football4 outputs renamed to the job id
  fail   - prints a phase and exits with an error
  hang   - sleeps until terminated (for cancellation tests)
  oldschema - exits cleanly but leaves only the pre-enhancement RAG schema
"""

import json
import shutil
import sys
import time
from pathlib import Path

mode, job_id, out_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
src = Path(__file__).resolve().parents[2] / "outputs" / "testing_football4"

print("--- [PHASE 1 & 2] Running Player & Ball Tracking ---", flush=True)
print("  Processed 50/100 frames (50%)...", flush=True)
if mode == "fail":
    print("RuntimeError: Could not open video", flush=True)
    sys.exit(1)
if mode == "hang":
    time.sleep(120)
    sys.exit(0)
if mode == "oldschema":
    rag = out_dir / "rag_ready" / f"{job_id}_match_and_player_metrics_for_rag.json"
    rag.parent.mkdir(parents=True, exist_ok=True)
    rag.write_text('{"metadata": {"schema_version": "2.0"}, "player_cards": [{"player_id": 1, "kinematics": {}}]}')
    sys.exit(0)
for marker in ["[PHASE 2.5]", "[PHASE 4 & 5]", "[PHASE 6]", "[PHASE 7]", "[PHASE 8]", "[PHASE 9]", "[PHASE 10]", "[PHASE 11]"]:
    print(f"--- {marker} ---", flush=True)
for f in src.rglob("*"):
    if f.is_file():
        dest = out_dir / f.relative_to(src).parent / f.name.replace("football4", job_id)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)

# Patch the RAG document's video_id so outputs_ready() stem check passes.
rag_dest = out_dir / "rag_ready" / f"{job_id}_match_and_player_metrics_for_rag.json"
if rag_dest.exists():
    try:
        doc = json.loads(rag_dest.read_text(encoding="utf-8"))
        doc.setdefault("metadata", {})["video_id"] = job_id
        rag_dest.write_text(json.dumps(doc), encoding="utf-8")
    except (OSError, ValueError):
        pass  # If patching fails the test will report a clear failure

print("STATUS: ALL PIPELINE PHASES COMPLETED SUCCESSFULLY", flush=True)
