
#!/usr/bin/env python3
"""Diagnose video camera motion and compare it with PnLCalib estimates.

This script is read-only with respect to source video and calibration files.
It estimates frame-to-frame global motion using optical flow and robust
geometric fitting. Estimates are diagnostic, not calibrated camera poses.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_calibrations(folder: Path) -> pd.DataFrame:
    """Load PnLCalib JSON files and compute adjacent-sample changes."""
    rows = []

    for path in folder.glob("*.json"):
        match = re.search(r"_(\d+(?:\.\d+)?)s\.json$", path.name)
        if not match:
            continue

        try:
            data = json.loads(path.read_text())
            cam = data["cam_params"]
            position = np.asarray(cam["position_meters"], dtype=float)
            rotation = np.asarray(cam["rotation_matrix"], dtype=float)

            if (
                position.shape != (3,)
                or rotation.shape != (3, 3)
                or not np.isfinite(position).all()
                or not np.isfinite(rotation).all()
            ):
                continue

            rows.append({
                "time_seconds": float(match.group(1)),
                "calibration_file": path.name,
                "rep_err": float(data["rep_err"]),
                "pan_degrees": float(cam["pan_degrees"]),
                "tilt_degrees": float(cam["tilt_degrees"]),
                "roll_degrees": float(cam["roll_degrees"]),
                "fx": float(cam["x_focal_length"]),
                "fy": float(cam["y_focal_length"]),
                "camera_x": position[0],
                "camera_y": position[1],
                "camera_z": position[2],
                "_position": position,
                "_rotation": rotation,
            })
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            continue

    if not rows:
        raise RuntimeError(f"No usable calibration files found in {folder}")

    rows.sort(key=lambda r: r["time_seconds"])
    df = pd.DataFrame(rows)

    df["sample_dt_seconds"] = df["time_seconds"].diff()
    df["delta_pan_degrees"] = df["pan_degrees"].diff()
    df["delta_tilt_degrees"] = df["tilt_degrees"].diff()
    df["delta_roll_degrees"] = df["roll_degrees"].diff()
    df["delta_fx_percent"] = df["fx"].pct_change() * 100
    df["delta_camera_position_m"] = [
        np.nan,
        *[
            float(np.linalg.norm(df.iloc[i]["_position"]
                                 - df.iloc[i - 1]["_position"]))
            for i in range(1, len(df))
        ],
    ]
    df["delta_rotation_frobenius"] = [
        np.nan,
        *[
            float(np.linalg.norm(df.iloc[i]["_rotation"]
                                 - df.iloc[i - 1]["_rotation"], ord="fro"))
            for i in range(1, len(df))
        ],
    ]

    return df.drop(columns=["_position", "_rotation"])


def estimate_pair_motion(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
) -> dict:
    """Estimate robust global motion from frame t to frame t+1."""
    points0 = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=1500,
        qualityLevel=0.01,
        minDistance=8,
        blockSize=7,
    )

    empty = {
        "tracked_points": 0,
        "inliers": 0,
        "inlier_ratio": np.nan,
        "dx_px": np.nan,
        "dy_px": np.nan,
        "scale_percent": np.nan,
        "rotation_degrees": np.nan,
        "homography_inliers": 0,
        "homography_perspective": np.nan,
        "status": "insufficient_features",
    }

    if points0 is None or len(points0) < 12:
        return empty

    points1, status, errors = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        points0,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    )

    if points1 is None or status is None:
        return empty

    valid = status.ravel().astype(bool)
    p0 = points0.reshape(-1, 2)[valid]
    p1 = points1.reshape(-1, 2)[valid]

    result = dict(empty)
    result["tracked_points"] = int(len(p0))

    if len(p0) < 12:
        return result

    # Robust similarity-like transform: translation, rotation and scale.
    affine, inlier_mask = cv2.estimateAffinePartial2D(
        p0,
        p1,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=3000,
        confidence=0.99,
        refineIters=10,
    )

    if affine is None or inlier_mask is None:
        result["status"] = "affine_fit_failed"
        return result

    inliers = inlier_mask.ravel().astype(bool)
    count = int(inliers.sum())
    result["inliers"] = count
    result["inlier_ratio"] = count / max(len(p0), 1)

    a, b = affine[0, 0], affine[1, 0]
    scale = float(np.sqrt(affine[0, 0] ** 2 + affine[1, 0] ** 2))
    rotation = float(np.degrees(np.arctan2(b, a)))

    result.update({
        "dx_px": float(affine[0, 2]),
        "dy_px": float(affine[1, 2]),
        "scale_percent": (scale - 1.0) * 100.0,
        "rotation_degrees": rotation,
        "status": "ok" if count >= 12 else "low_inlier_count",
    })

    # A homography can capture perspective changes, but is more sensitive
    # to moving objects. Record it as a secondary diagnostic only.
    if count >= 12:
        H, hmask = cv2.findHomography(
            p0[inliers],
            p1[inliers],
            cv2.RANSAC,
            3.0,
        )
        if H is not None and hmask is not None:
            result["homography_inliers"] = int(hmask.sum())
            result["homography_perspective"] = float(
                np.linalg.norm(H[2, :2])
            )

    return result


def analyze_video(
    video_path: Path,
    output_dir: Path,
    start_seconds: float,
    end_seconds: float | None,
    calibration_df: pd.DataFrame,
) -> pd.DataFrame:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        cap.release()
        raise RuntimeError("Video reported an invalid FPS.")

    duration = frame_count / fps
    stop = duration if end_seconds is None else min(end_seconds, duration)
    start = max(0.0, start_seconds)

    if stop <= start:
        cap.release()
        raise ValueError("end_seconds must be greater than start_seconds")

    start_frame = max(0, int(round(start * fps)))
    end_frame = min(frame_count - 1, int(round(stop * fps)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok, previous = cap.read()

    if not ok:
        cap.release()
        raise RuntimeError("Could not read starting video frame.")

    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    previous_frame_id = start_frame
    rows = []

    for frame_id in range(start_frame + 1, end_frame + 1):
        ok, frame = cap.read()
        if not ok:
            break

        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion = estimate_pair_motion(previous_gray, current_gray)

        rows.append({
            "previous_frame_id": previous_frame_id,
            "frame_id": frame_id,
            "previous_time_seconds": previous_frame_id / fps,
            "time_seconds": frame_id / fps,
            "fps": fps,
            **motion,
        })

        previous_gray = current_gray
        previous_frame_id = frame_id

    cap.release()
    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No frame pairs were analyzed.")

    # Attach nearest calibration sample for inspection. This does not
    # imply that the sample represents the exact video frame.
    if not calibration_df.empty:
        cal = calibration_df.sort_values("time_seconds")
        times = cal["time_seconds"].to_numpy()
        indices = np.abs(
            df["time_seconds"].to_numpy()[:, None] - times[None, :]
        ).argmin(axis=1)
        nearest = cal.iloc[indices].reset_index(drop=True)

        df["nearest_calibration_time"] = nearest["time_seconds"].to_numpy()
        df["calibration_time_offset_seconds"] = (
            df["time_seconds"] - df["nearest_calibration_time"]
        )
        for column in [
            "rep_err",
            "pan_degrees",
            "roll_degrees",
            "fx",
            "delta_fx_percent",
            "delta_camera_position_m",
        ]:
            df[f"cal_{column}"] = nearest[column].to_numpy()

    # Flag suspicious image-motion samples for review, not automatic rejection.
    df["suspicious_motion"] = (
        df["inlier_ratio"].lt(0.35)
        | df["scale_percent"].abs().gt(3.0)
        | df["rotation_degrees"].abs().gt(3.0)
        | df["dx_px"].abs().gt(25.0)
        | df["dy_px"].abs().gt(25.0)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "camera_motion.csv", index=False)

    summary = {
        "video": str(video_path),
        "fps": fps,
        "frame_count": frame_count,
        "video_duration_seconds": duration,
        "analyzed_start_seconds": start_frame / fps,
        "analyzed_end_seconds": end_frame / fps,
        "frame_pairs_analyzed": len(df),
        "low_quality_pairs": int(
            df["status"].ne("ok").sum()
        ),
        "suspicious_motion_pairs": int(df["suspicious_motion"].sum()),
        "median_inlier_ratio": float(df["inlier_ratio"].median()),
        "median_abs_scale_change_percent": float(
            df["scale_percent"].abs().median()
        ),
        "warnings": [
            "Optical flow is not a metric camera-pose estimate.",
            "Moving players and changing visible background can affect fits.",
            "Calibration jumps and optical-flow jumps are evidence, not proof.",
            "No player coordinates or calibration files were modified.",
        ],
    }

    (output_dir / "camera_motion_report.json").write_text(
        json.dumps(summary, indent=2)
    )

    return df


def create_plots(
    motion_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    axes[0].plot(
        motion_df["time_seconds"],
        motion_df["dx_px"],
        label="Global horizontal translation (px/frame)",
    )
    axes[0].plot(
        motion_df["time_seconds"],
        motion_df["dy_px"],
        label="Global vertical translation (px/frame)",
    )
    axes[0].set_ylabel("Translation")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        motion_df["time_seconds"],
        motion_df["scale_percent"],
        label="Optical-flow scale change (%)",
    )
    axes[1].plot(
        motion_df["time_seconds"],
        motion_df["rotation_degrees"],
        label="Optical-flow rotation (degrees)",
    )
    axes[1].set_ylabel("Scale / rotation")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(
        calibration_df["time_seconds"],
        calibration_df["fx"],
        marker="o",
        label="Calibration focal length fx (px)",
    )
    axes[2].set_ylabel("Focal length")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("AthletiQ: camera motion vs. calibration estimates")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--params-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(
        "outputs/physical_metrics/camera_motion_diagnostic"
    ))
    parser.add_argument("--start", type=float, default=12.0)
    parser.add_argument("--end", type=float, default=17.0)
    args = parser.parse_args()

    calibration_df = parse_calibrations(args.params_dir)

    # Save the complete calibration continuity audit.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibration_df.to_csv(
        args.output_dir / "calibration_motion.csv",
        index=False,
    )

    motion_df = analyze_video(
        args.video,
        args.output_dir,
        args.start,
        args.end,
        calibration_df,
    )

    create_plots(
        motion_df,
        calibration_df,
        args.output_dir / "camera_motion_diagnostic.png",
    )

    print("\n=== Camera motion diagnostic complete ===")
    print("Output directory:", args.output_dir)
    print("Frame pairs analyzed:", len(motion_df))
    print("Low-quality fits:", int(motion_df["status"].ne("ok").sum()))
    print(
        "Suspicious motion pairs:",
        int(motion_df["suspicious_motion"].sum()),
    )
    print("\nLargest image-motion changes:")
    columns = [
        "time_seconds", "dx_px", "dy_px", "scale_percent",
        "rotation_degrees", "inlier_ratio", "status",
    ]
    print(
        motion_df.assign(
            motion_magnitude=np.hypot(
                motion_df["dx_px"], motion_df["dy_px"]
            )
        ).nlargest(15, "motion_magnitude")[columns]
        .round(4).to_string(index=False)
    )


if __name__ == "__main__":
    main()
