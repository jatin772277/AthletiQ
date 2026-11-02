# AthletiQ API

FastAPI service that serves match analytics, user accounts (MongoDB) and video-upload analysis jobs.

## Setup

1. **MongoDB**: run a local server (MongoDB Community, default `mongodb://localhost:27017`) or create a free MongoDB Atlas cluster and use its connection string.
2. Copy `.env.example` to `.env` in the repo root and set `MONGODB_URI`, `JWT_SECRET` and the pipeline settings.
3. Install and run:

```bash
pip install -r backend/requirements.txt
python backend/run.py
```

On start-up the API logs whether MongoDB is reachable and starts the pipeline worker. `GET /api/health` reports `"database": "connected"` when it is.

## Accounts

- `POST /api/auth/register` `{name, email, password}`: creates the user (scrypt-hashed password) and signs in
- `POST /api/auth/login` `{email, password}`: returns `{access_token, user}`
- `POST /api/auth/logout`: revokes the current session
- `GET /api/auth/me`

Tokens are JWTs that reference a session document in MongoDB, so signing out revokes the token immediately. Sessions expire after `SESSION_TTL_HOURS`. Every `/api/matches/...` route needs an `Authorization: Bearer <token>` header and only serves the demo matches (`DEMO_MATCHES`) and the user's own completed analyses.

## Video analysis jobs

- `POST /api/jobs` (multipart: `file`, `title`, `start_seconds`, `duration_seconds`): stores the video in `uploads/<user_id>/`, reads its fps and length, and queues it
- `GET /api/jobs`, `GET /api/jobs/{id}`: status, progress %, current phase and the last log lines
- `POST /api/jobs/{id}/cancel`, `DELETE /api/jobs/{id}`

A single background worker runs one job at a time:

1. `AthletiQ.py` (tracking, stitching, calibration, spatial, possession, network, DSA, ML, RAG export)
2. `ai-engine/scripts/apply_collective_improvements.py` (collective metrics + removed-players buckets, schema v3)

Outputs go to `outputs/user_matches/<job_id>/`, and the job's match ID is used as the video stem, so the normal match endpoints serve it. A job is marked `completed` only once the schema v3 RAG document exists. With `PIPELINE_MODE=wsl` both steps run inside WSL using `PIPELINE_WSL_VENV` (your GPU environment); `local` runs them with the backend's Python on CPU, which needs the full pipeline dependencies installed there. Jobs that were running when the server stopped are marked failed on the next start; queued jobs resume.

## Tests

```bash
python -m pytest backend/tests
```

Tests use an in-memory MongoDB (`mongomock`) and a fake pipeline, so no database or GPU is needed. They do read the demo data in `outputs/testing_football4`.
