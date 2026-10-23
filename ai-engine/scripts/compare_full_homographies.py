
#!/usr/bin/env python3
"""
Compare calibration-predicted pitch motion against video-derived homographies.

Diagnostic only: does not modify calibration JSONs or player-track data.

Outputs:
    full_homography_comparison.csv
    full_homography_report.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def load_projector(script_path: Path):
    """Load the production ground-plane projection helper."""
    spec = importlib.util.spec_from_file_location(
        "athletiq_physical_metrics", script_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load projection script: {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "ground_homography"):
        raise RuntimeError(
            f"ground_homography() not found in {script_path}"
        )

    return module


def load_calibrations(folder: Path, projector) -> list[dict]:
    """Read calibration homographies and align projective matrix signs."""
    samples = []

    for path in folder.glob("*.json"):
        match = re.search(r"_(\d+(?:\.\d+)?)s\.json$", path.name)
        if not match:
            continue

        try:
            data = json.loads(path.read_text())
            H = np.asarray(
                projector.ground_homography(data),
                dtype=np.float64,
            )

            if H.shape != (3, 3):
                continue
            if not np.isfinite(H).all():
                continue
            if abs(np.linalg.det(H)) < 1e-12:
                continue

            samples.append({
                "time": float(match.group(1)),
                "file": path.name,
                "H": H,
                "rep_err": data.get("rep_err", np.nan),
            })

        except (
            ValueError,
            TypeError,
            KeyError,
            np.linalg.LinAlgError,
            json.JSONDecodeError,
        ) as exc:
            print(f"Skipping {path.name}: {exc}")

    samples.sort(key=lambda sample: sample["time"])

    if len(samples) < 2:
        raise RuntimeError(
            f"Fewer than two valid calibration homographies in {folder}"
        )

    for i in range(1, len(samples)):
        if np.sum(samples[i - 1]["H"] * samples[i]["H"]) < 0:
            samples[i]["H"] = -samples[i]["H"]

    return samples


def normalize_homography(H: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(H)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Invalid homography normalization")
    return H / norm


def calibration_at(
    samples: list[dict],
    timestamp: float,
    max_gap: float,
) -> np.ndarray:
    """Interpolate calibration homographies without extrapolation."""
    times = np.asarray([sample["time"] for sample in samples])
    right = int(np.searchsorted(times, timestamp, side="left"))

    if right < len(samples) and abs(times[right] - timestamp) < 1e-8:
        return normalize_homography(samples[right]["H"])

    if right == 0 or right >= len(samples):
        raise ValueError(
            f"Timestamp {timestamp:.3f}s outside calibration range"
        )

    left_sample = samples[right - 1]
    right_sample = samples[right]
    gap = right_sample["time"] - left_sample["time"]

    if gap <= 0 or gap > max_gap:
        raise ValueError(
            f"Calibration gap {gap:.3f}s exceeds limit {max_gap:.3f}s"
        )

    H0 = normalize_homography(left_sample["H"])
    H1 = normalize_homography(right_sample["H"])

    if np.sum(H0 * H1) < 0:
        H1 = -H1

    alpha = (timestamp - left_sample["time"]) / gap
    return normalize_homography((1 - alpha) * H0 + alpha * H1)


def project_points(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Project Nx2 world coordinates to image coordinates safely."""
    homogeneous = np.column_stack([points, np.ones(len(points))])
    projected = (H @ homogeneous.T).T
    denominator = projected[:, 2]

    result = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = (
        np.isfinite(projected).all(axis=1)
        & (np.abs(denominator) > 1e-8)
    )
    result[valid] = (
        projected[valid, :2] / denominator[valid, None]
    )
    return result


def points_inside_image(
    points: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Mask finite points that fall inside image bounds."""
    return (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= 0)
        & (points[:, 0] < width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < height)
    )


def observed_homography(
    video: cv2.VideoCapture,
    frame0: int,
    frame1: int,
    max_features: int = 2500,
) -> tuple[np.ndarray, dict, int, int]:
    """Estimate frame0-to-frame1 homography using optical flow and RANSAC."""
    video.set(cv2.CAP_PROP_POS_FRAMES, frame0)
    ok0, image0 = video.read()

    video.set(cv2.CAP_PROP_POS_FRAMES, frame1)
    ok1, image1 = video.read()

    if not ok0 or not ok1:
        raise ValueError("Could not read one or both video frames")

    gray0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    points0 = cv2.goodFeaturesToTrack(
        gray0,
        maxCorners=max_features,
        qualityLevel=0.01,
        minDistance=8,
        blockSize=7,
    )

    if points0 is None or len(points0) < 12:
        raise ValueError("Too few features detected in frame 0")

    points1, status, _ = cv2.calcOpticalFlowPyrLK(
        gray0,
        gray1,
        points0,
        None,
        winSize=(31, 31),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    )

    if points1 is None or status is None:
        raise ValueError("Optical flow failed")

    p0 = points0.reshape(-1, 2)
    p1 = points1.reshape(-1, 2)

    valid = status.ravel().astype(bool)
    p0, p1 = p0[valid], p1[valid]

    if len(p0) < 12:
        raise ValueError("Too few valid feature tracks")

    H, mask = cv2.findHomography(
        p0,
        p1,
        cv2.RANSAC,
        3.0,
        maxIters=5000,
        confidence=0.995,
    )

    if H is None or mask is None:
        raise ValueError("Could not estimate video homography")

    inliers = mask.ravel().astype(bool)
    if int(inliers.sum()) < 10:
        raise ValueError("Fewer than 10 homography inliers")

    predicted = cv2.perspectiveTransform(
        p0[inliers].reshape(-1, 1, 2).astype(np.float32),
        H,
    ).reshape(-1, 2)

    residuals = np.linalg.norm(predicted - p1[inliers], axis=1)

    stats = {
        "features_tracked": int(len(p0)),
        "homography_inliers": int(inliers.sum()),
        "inlier_ratio": float(inliers.mean()),
        "feature_residual_median_px": float(np.median(residuals)),
        "feature_residual_p90_px": float(np.percentile(residuals, 90)),
    }

    return H, stats, image0.shape[1], image0.shape[0]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare calibration-predicted and video-derived "
            "full projective transformations."
        )
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--optical-flow", type=Path, required=True)
    parser.add_argument("--params-dir", type=Path, required=True)

    parser.add_argument(
        "--projection-script",
        type=Path,
        default=Path("ai-engine/scripts/athletiq_physical_metrics.py"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/physical_metrics/full_homography_comparison_visible"
        ),
    )
    parser.add_argument("--max-calibration-gap", type=float, default=0.30)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.35)
    parser.add_argument("--min-visible-points", type=int, default=8)
    parser.add_argument("--review-p90-px", type=float, default=8.0)
    parser.add_argument("--pitch-length", type=float, default=105.0)
    parser.add_argument("--pitch-width", type=float, default=68.0)
    args = parser.parse_args()

    for path in (args.video, args.optical_flow, args.projection_script):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    if not args.params_dir.is_dir():
        raise NotADirectoryError(
            f"Calibration directory not found: {args.params_dir}"
        )

    projector = load_projector(args.projection_script)
    calibrations = load_calibrations(args.params_dir, projector)
    flow = pd.read_csv(args.optical_flow)

    required_columns = {
        "previous_frame_id",
        "frame_id",
        "previous_time_seconds",
        "time_seconds",
    }
    missing = required_columns - set(flow.columns)

    if missing:
        raise ValueError(
            f"Optical-flow CSV missing columns: {sorted(missing)}"
        )

    # Assumes a centered world-coordinate pitch. Validate this convention
    # against the production calibration before physical interpretation.
    xs = np.linspace(
        -args.pitch_length / 2,
        args.pitch_length / 2,
        41,
    )
    ys = np.linspace(
        -args.pitch_width / 2,
        args.pitch_width / 2,
        25,
    )
    xx, yy = np.meshgrid(xs, ys)
    pitch_grid = np.column_stack([xx.ravel(), yy.ravel()])

    video = cv2.VideoCapture(str(args.video))
    if not video.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    results = []

    try:
        for _, row in flow.iterrows():
            frame0 = int(row["previous_frame_id"])
            frame1 = int(row["frame_id"])
            time0 = float(row["previous_time_seconds"])
            time1 = float(row["time_seconds"])

            output = {
                "frame0": frame0,
                "frame1": frame1,
                "time0": time0,
                "time1": time1,
            }

            try:
                H0 = calibration_at(
                    calibrations, time0, args.max_calibration_gap
                )
                H1 = calibration_at(
                    calibrations, time1, args.max_calibration_gap
                )

                # Transform from image 0 to image 1 implied by calibration.
                H_calibration = normalize_homography(
                    H1 @ np.linalg.inv(H0)
                )

                H_video, stats, width, height = observed_homography(
                    video, frame0, frame1
                )
                H_video = normalize_homography(H_video)

                # Project pitch points into the first image.
                image0_points = project_points(H0, pitch_grid)

                visible0 = points_inside_image(
                    image0_points, width, height
                )
                image0_points = image0_points[visible0]

                if len(image0_points) < args.min_visible_points:
                    raise ValueError(
                        "Too few pitch-grid points visible in frame 0"
                    )

                points = image0_points.reshape(
                    -1, 1, 2
                ).astype(np.float32)

                predicted1 = cv2.perspectiveTransform(
                    points, H_calibration
                ).reshape(-1, 2)

                observed1 = cv2.perspectiveTransform(
                    points, H_video
                ).reshape(-1, 2)

                # Keep points visible in frame 1 under BOTH transformations.
                visible_predicted = points_inside_image(
                    predicted1, width, height
                )
                visible_observed = points_inside_image(
                    observed1, width, height
                )
                jointly_visible = visible_predicted & visible_observed

                predicted1 = predicted1[jointly_visible]
                observed1 = observed1[jointly_visible]

                if len(predicted1) < args.min_visible_points:
                    raise ValueError(
                        "Too few grid points visible in both frames under "
                        "both transformations"
                    )

                errors = np.linalg.norm(predicted1 - observed1, axis=1)
                errors = errors[np.isfinite(errors)]

                if len(errors) < args.min_visible_points:
                    raise ValueError("Too few finite point errors")

                output.update(stats)
                output.update({
                    "comparison_status": "ok",
                    "pitch_grid_points_total": int(len(pitch_grid)),
                    "grid_points_visible_frame0": int(visible0.sum()),
                    "grid_points_compared": int(len(errors)),
                    "grid_error_median_px": float(np.median(errors)),
                    "grid_error_p90_px": float(np.percentile(errors, 90)),
                    "grid_error_max_px": float(np.max(errors)),
                })

                reliable = stats["inlier_ratio"] >= args.min_inlier_ratio
                output["video_homography_reliable"] = bool(reliable)
                output["review_flag"] = bool(
                    reliable
                    and output["grid_error_p90_px"] > args.review_p90_px
                )

            except (
                ValueError,
                np.linalg.LinAlgError,
                cv2.error,
                FloatingPointError,
            ) as exc:
                output.update({
                    "comparison_status": f"unavailable: {exc}",
                    "video_homography_reliable": False,
                    "review_flag": False,
                })

            results.append(output)

    finally:
        video.release()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    result_df = pd.DataFrame(results)
    csv_path = args.output_dir / "full_homography_comparison.csv"
    report_path = args.output_dir / "full_homography_report.json"

    result_df.to_csv(csv_path, index=False)

    available = result_df[
        result_df["comparison_status"].eq("ok")
    ]
    reliable = available[
        available["video_homography_reliable"].fillna(False)
    ]
    flagged = reliable[reliable["review_flag"].fillna(False)]

    report = {
        "video": str(args.video),
        "optical_flow_csv": str(args.optical_flow),
        "calibration_directory": str(args.params_dir),
        "output_csv": str(csv_path),
        "intervals_total": int(len(result_df)),
        "comparisons_available": int(len(available)),
        "reliable_video_homographies": int(len(reliable)),
        "review_flags": int(len(flagged)),
        "median_grid_error_px": (
            float(reliable["grid_error_median_px"].median())
            if len(reliable) else None
        ),
        "p90_grid_error_p90_px": (
            float(reliable["grid_error_p90_px"].quantile(0.90))
            if len(reliable) else None
        ),
        "median_compared_grid_points": (
            float(reliable["grid_points_compared"].median())
            if len(reliable) else None
        ),
        "warnings": [
            "Only points visible in both frames under both transformations "
            "are included in the error metrics.",
            "Video homographies may be biased by moving players or other "
            "non-static features.",
            "The pitch grid assumes centered world coordinates; verify this "
            "against the production calibration convention.",
            "Review flags are triage signals, not proof of calibration failure.",
            "No calibration or player-track files were modified.",
        ],
    }

    report_path.write_text(json.dumps(report, indent=2))

    print("\n=== Full homography comparison (visible points only) ===")
    print(json.dumps(report, indent=2))

    if len(reliable):
        columns = [
            "time0",
            "time1",
            "inlier_ratio",
            "grid_points_compared",
            "grid_error_median_px",
            "grid_error_p90_px",
            "grid_error_max_px",
            "review_flag",
        ]

        print("\nLargest reliable disagreements:")
        print(
            reliable.nlargest(20, "grid_error_p90_px")[columns]
            .round(3)
            .to_string(index=False)
        )

    print("\nComparison status counts:")
    print(result_df["comparison_status"].value_counts().to_string())

    print(f"\nCSV: {csv_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
