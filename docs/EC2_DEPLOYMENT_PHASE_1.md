# AthletiQ Phase 1: Ubuntu EC2 preparation

This guide prepares a single Ubuntu host for the current application. It does not
create AWS resources, deploy code, install a GPU driver, or claim that the video
inference environment is reproducible yet.

## What persists

Mount persistent storage at `/srv/athletiq-data` and use it for uploads and
generated outputs. The API supports this with `ATHLETIQ_DATA_DIR`; explicit
`UPLOADS_DIR` and `OUTPUTS_DIR` take precedence. Model files may remain in the
repository or be mounted separately with `MODEL_ASSETS_DIR`.

```text
/srv/athletiq-data/
  uploads/
  outputs/
  models/                 # optional, if MODEL_ASSETS_DIR points here
```

## Host and API setup

Use an Ubuntu LTS host with an EBS-backed filesystem. The following commands set
up the API environment only; they intentionally do not attempt to guess or
install the unverified CPU inference stack described below.

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip build-essential git
sudo mkdir -p /srv/athletiq-data/{uploads,outputs}
sudo chown -R "$USER":"$USER" /srv/athletiq-data

git clone <your-private-repository-url> athletiq
cd athletiq
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt
```

MongoDB is required by the current application for users, sessions, and job
metadata. Install and operate MongoDB separately, or use a reachable managed
MongoDB-compatible service. Do not expose MongoDB publicly. Confirm that the
connection string works before starting the API.

Create a protected host environment file (not in Git), for example
`/etc/athletiq/athletiq.env`:

```dotenv
APP_ENV=production
DEBUG=false
HOST=0.0.0.0
PORT=8000

# Required production validation values.
MONGO_URI=mongodb://127.0.0.1:27017/athletiq
JWT_SECRET=replace-with-a-random-secret-of-at-least-32-characters
CORS_ORIGINS=https://app.example.com
VITE_API_BASE=https://api.example.com

# Persistent paths.
ATHLETIQ_DATA_DIR=/srv/athletiq-data
# MODEL_ASSETS_DIR=/srv/athletiq-data/models

# Native Linux execution. CPU is explicit; do not set a GPU index until a
# compatible NVIDIA/PyTorch environment has been separately validated.
PIPELINE_MODE=local
PIPELINE_DEVICE=cpu
PIPELINE_PYTHON=/home/ubuntu/athletiq/.venv/bin/python
PIPELINE_MAX_FRAMES=750
MAX_UPLOAD_MB=2048
```

`MONGODB_URI` remains supported for existing local development only. Production
requires `MONGO_URI`, an explicit non-wildcard `CORS_ORIGINS`, `DEBUG=false`, a
non-placeholder `JWT_SECRET`, and an absolute `VITE_API_BASE`.

Start manually for a smoke test:

```bash
cd /home/ubuntu/athletiq
set -a; . /etc/athletiq/athletiq.env; set +a
.venv/bin/python backend/run.py
```

In a separate shell:

```bash
curl -fsS http://127.0.0.1:8000/api/health
```

Success requires `"status":"healthy"`, `"database":"connected"`, and
`"database_ready":true`. A `degraded` response means the API process is alive
but MongoDB is not ready; uploads and authentication will fail.

For the frontend, build from the same production environment so `VITE_API_BASE`
is embedded into the static bundle:

```bash
cd /home/ubuntu/athletiq
set -a; . /etc/athletiq/athletiq.env; set +a
npm --prefix frontend ci
npm --prefix frontend run build
```

Serve `frontend/dist` from a static HTTPS host or reverse proxy. Do not use the
Vite development server as the production web server.

## Shutdown and resume

Before stopping the host, cancel or allow active analysis jobs to finish. The
current in-process runner terminates running work during shutdown and marks it
failed after restart; queued work is re-queued. Stop the API process/service,
then stop the EC2 instance from AWS. EBS data persists; instance-store data does
not. On resume, start MongoDB first, then the API, and repeat the health check.

## Inference reproducibility blocker

`backend/requirements.txt` is sufficient only for the FastAPI API. A Linux CPU
upload job also imports these packages from `AthletiQ.py` and the pipeline:

- `opencv-python-headless`/OpenCV, `joblib`, NumPy, pandas, Pillow, PyTorch,
  TorchVision, Ultralytics, and PyYAML;
- MixSort's own dependencies from
  `ai-engine/third_party/MixSort/requirements.txt` and
  `ai-engine/third_party/MixSort/MixViT/requirements.txt` (including `lap`,
  `filterpy`, `motmetrics`, `timm`, `yacs`, `einops`, `pycocotools`, and
  `scipy`).

Only MixSort's `numpy==1.22.3`, ONNX, and ONNX Runtime entries are versioned in
those files; its own file says the exact module versions still need updating.
The application has no verified Python, PyTorch, TorchVision, Ultralytics, or
CPU wheel-version matrix. This phase deliberately does not invent one.

The following required runtime artifacts are ignored/untracked and must be
provided from a trusted, versioned artifact store before real inference:

- `ai-engine/models/football_player_detector.pt`
- `ai-engine/models/athletiq_ball_detector_champion_1280.pt`
- `ai-engine/models/team_identification/best_team_classifier.pt`
- `ai-engine/models/performance_scoring/performance_model.joblib`
- the complete `ai-engine/third_party/MixSort` checkout, including its Python
  sources and `pretrained/MixFormer_sports_train.pth.tar` and
  `pretrained/MixFormerOnlineScore.pth.tar`

Optional Re-ID is enabled only if
`models/reid/osnet_x1_0_msmt17_combineall.pth` is supplied. The current
`mixsort_final.yaml` also contains a Windows `/mnt/d/...` checkpoint path; its
consumer currently reads tracker parameters but the path must be audited before
declaring MixSort reproducible on Linux.

Until the dependency lockfile and artifact manifest are verified against a real
CPU clip, use the deployed API only for health/auth/demo-data verification, not
as evidence that full video inference is production-ready.
