#!/usr/bin/env bash
set -e
source ~/tf312/bin/activate
cd "/mnt/d/My Projects/AthletiQ/AthletiQ"

echo "=== Running Centralized AthletiQ.py Master Pipeline on WSL GPU ==="
python3 -c "import torch; print('Using GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python3 AthletiQ.py --video "datasets/football/raw/videos/football4.mp4" --output-dir "outputs/testing_football4" --start-frame 450 --max-frames 750 --device 0 --clean --force-retrack
