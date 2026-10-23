
from pathlib import Path
import argparse
import csv
import json
import sys

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import yaml

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[2]
REPO_DIR = (
    PROJECT_DIR / "ai-engine" / "notebooks" / "PnLCalib"
)

DEFAULT_VIDEO = (
    PROJECT_DIR / "datasets" / "football" / "raw" / "videos"
    / "football2.mp4"
)

DEFAULT_OUTPUT = (
    PROJECT_DIR / "outputs" / "calibration" / "pnlcalib"
)


def json_safe(value):
    """Convert NumPy and PyTorch values into JSON-compatible values."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def main():
    parser = argparse.ArgumentParser(
        description="Run experimental frame-wise PnLCalib calibration."
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Seconds between sampled frames.",
    )
    parser.add_argument(
        "--pnl-refine",
        action="store_true",
        help="Enable PnL line refinement.",
    )
    args = parser.parse_args()

    if not args.video.is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")

    if not (REPO_DIR / "SV_kp").is_file():
        raise FileNotFoundError(f"Keypoint weights missing: {REPO_DIR / 'SV_kp'}")

    if not (REPO_DIR / "SV_lines").is_file():
        raise FileNotFoundError(f"Line weights missing: {REPO_DIR / 'SV_lines'}")

    if args.interval <= 0:
        raise ValueError("--interval must be greater than zero.")

    sys.path.insert(0, str(REPO_DIR))
    import os
    os.chdir(REPO_DIR)

    import inference as pnl
    from model.cls_hrnet import get_cls_net
    from model.cls_hrnet_l import get_cls_net as get_cls_net_l
    from utils.utils_calib import FramebyFrameCalib

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pnl.device = device
    pnl.transform2 = T.Resize((540, 960))

    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    with open(REPO_DIR / "config/hrnetv2_w48.yaml", encoding="utf-8") as f:
        cfg_kp = yaml.safe_load(f)

    with open(REPO_DIR / "config/hrnetv2_w48_l.yaml", encoding="utf-8") as f:
        cfg_lines = yaml.safe_load(f)

    print("Loading keypoint model...")
    model_kp = get_cls_net(cfg_kp)
    state_kp = torch.load(
        REPO_DIR / "SV_kp",
        map_location="cpu",
        weights_only=True,
    )
    model_kp.load_state_dict(state_kp, strict=True)
    model_kp.to(device).eval()
    del state_kp

    print("Loading line model...")
    model_lines = get_cls_net_l(cfg_lines)
    state_lines = torch.load(
        REPO_DIR / "SV_lines",
        map_location="cpu",
        weights_only=True,
    )
    model_lines.load_state_dict(state_lines, strict=True)
    model_lines.to(device).eval()
    del state_lines

    overlay_dir = args.output / "overlays"
    params_dir = args.output / "parameters"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    params_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not fps or fps <= 0:
        cap.release()
        raise RuntimeError("Video FPS is invalid.")

    duration = frame_count / fps
    print(f"Video: {args.video}")
    print(f"Dimensions: {width}x{height}, FPS: {fps:.3f}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Sampling interval: {args.interval:.2f} seconds")

    rows = []
    sample_times = np.arange(0, duration, args.interval)

    try:
        for index, seconds in enumerate(sample_times, start=1):
            frame_number = min(round(seconds * fps), frame_count - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = cap.read()

            row = {
                "frame_id": frame_number,
                "time_seconds": round(frame_number / fps, 4),
                "calibration_returned": False,
                "mode": None,
                "use_ransac": None,
                "rep_err": None,
                "pan_degrees": None,
                "tilt_degrees": None,
                "roll_degrees": None,
                "x_focal_length": None,
                "y_focal_length": None,
                "status": "candidate_unvalidated",
                "overlay_path": "",
                "parameters_path": "",
                "error": "",
            }

            if not ok:
                row["status"] = "failed_to_read_frame"
                row["error"] = "Video frame could not be read."
                rows.append(row)
                continue

            timestamp = row["time_seconds"]
            stem = f"frame_{frame_number:06d}_{timestamp:07.2f}s"

            print(
                f"[{index}/{len(sample_times)}] "
                f"frame={frame_number}, time={timestamp:.2f}s"
            )

            try:
                cam = FramebyFrameCalib(
                    iwidth=frame.shape[1],
                    iheight=frame.shape[0],
                    denormalize=True,
                )

                with torch.inference_mode():
                    params = pnl.inference(
                        cam=cam,
                        frame=frame.copy(),
                        model=model_kp,
                        model_l=model_lines,
                        kp_threshold=0.3434,
                        line_threshold=0.7867,
                        pnl_refine=args.pnl_refine,
                    )

                if params is None:
                    row["status"] = "no_calibration_returned"
                else:
                    cp = params.get("cam_params") or {}
                    row["calibration_returned"] = True
                    row["mode"] = params.get("mode")
                    row["use_ransac"] = params.get("use_ransac")
                    row["rep_err"] = params.get("rep_err")

                    for key in [
                        "pan_degrees",
                        "tilt_degrees",
                        "roll_degrees",
                        "x_focal_length",
                        "y_focal_length",
                    ]:
                        row[key] = cp.get(key)

                    params_path = params_dir / f"{stem}.json"
                    params_path.write_text(
                        json.dumps(json_safe(params), indent=2),
                        encoding="utf-8",
                    )
                    row["parameters_path"] = str(params_path)

                    projection = pnl.projection_from_cam_params(params)
                    projected = pnl.project(frame.copy(), projection)

                    if projected is not None:
                        overlay_path = overlay_dir / f"{stem}.jpg"
                        if cv2.imwrite(str(overlay_path), projected):
                            row["overlay_path"] = str(overlay_path)

                    # Intentionally do not mark calibration as validated.
                    row["status"] = "candidate_unvalidated"

                print(
                    f"    returned={row['calibration_returned']}, "
                    f"error={row['rep_err']}, status={row['status']}"
                )

            except Exception as exc:
                row["status"] = "inference_error"
                row["error"] = f"{type(exc).__name__}: {exc}"
                print("    ERROR:", row["error"])

            rows.append(row)

    finally:
        cap.release()

    summary_path = args.output / "calibration_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    returned = sum(r["calibration_returned"] for r in rows)

    print("\n=== PnLCalib summary ===")
    print("Sampled frames:", len(rows))
    print("Frames with parameters:", returned)
    print("Validated calibrations: 0 (independent validation not performed)")
    print("Summary:", summary_path)
    print("Overlays:", overlay_dir)
    print("Parameters:", params_dir)


if __name__ == "__main__":
    main()
