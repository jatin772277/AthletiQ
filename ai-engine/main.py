"""AthletiQ AI Engine — subprocess-invoked pipeline scripts.

This module intentionally has no standalone HTTP server.
All ai-engine scripts are executed as child processes by
``backend/app/services/pipeline_runner.py``, which manages
subprocess lifecycle, stdout streaming and progress reporting.

Entry points invoked by the pipeline runner:
  ai-engine/scripts/apply_collective_improvements.py  -- schema v3.0 enhancer
  (AthletiQ.py at the project root handles all CV tracking phases 1-11)

To run the AI engine pipeline manually:
  python AthletiQ.py --help
  python ai-engine/scripts/apply_collective_improvements.py --help
"""
