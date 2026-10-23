
#!/usr/bin/env python3
"""Compare calibration-predicted image motion with observed optical flow."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def load_calibrations(folder: Path) -> list[dict]:
    samples = []

    for path in folder.glob("*.json"):
        match = re.search(r"_(\d+(?:\.\d+)?)s\.json$", path.name)
        if not match:
            continue

        try:
            data = json.loads(path.read_text())
            cam = data["cam_params"]

            fx = float(cam["x_focal_length"])
            fy = float(cam["y_focal_length"])
            cx, cy = np.asarray(
                cam["principal_point"], dtype=float
            ).reshape(2)

            R = np.asarray(cam["rotation_matrix"], dtype=float)
            C = np.asarray(cam["position_meters"], dtype=float).reshape(3)

            if (
                R.shape != (3, 3)
                or not np.isfinite(R).all()
                or not np.isfinite(C).all()
                or fx <= 0
                or fy <= 0
            ):
                continue

            K = np.array([
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ])

            # Match the projection convention used by AthletiQ:
            # P = K @ R @ [I | -C].
            extrinsic = np.column_stack((np.eye(3), -C))
            P = K @ R @ extrinsic
            H = P[:, [0, 1, 3]]

            if not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
                continue

            samples.append({
                "time": float(match.group(1)),
                "file": path.name,
                "H": H,
                "rep_err": float(data.get("rep_err", np.nan)),
            })

        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            continue

    samples.sort(key=lambda s: s["time"])

    # Resolve the sign ambiguity of projective matrices before interpolation.
    for i in range(1, len(samples)):
        previous = samples[i - 1]["H"]
        current = samples[i]["H"]
        if np.sum(previous * current) < 0:
            samples[i]["H"] = -current

    if len(samples) < 2:
        raise RuntimeError(f"Need at least two valid calibrations in {folder}")

    return samples


def normalize_homography(H: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(H)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Invalid homography normalization")
    return H / norm


def homography_at(samples: list[dict], time: float, max_gap: float) -> np.ndarray:
    """Interpolate normalized homographies without extrapolation."""
    times = np.array([s["time"] for s in samples])
    right = int(np.searchsorted(times, time, side="left"))

    if right < len(samples) and abs(times[right] - time) < 1e-8:
        return normalize_homography(samples[right]["H"])

    if right == 0 or right >= len(samples):
        raise ValueError("Time outside calibration range")

    left_sample = samples[right - 1]
    right_sample = samples[right]
    gap = right_sample["time"] - left_sample["time"]

    if gap <= 0 or gap > max_gap:
        raise ValueError(f"Calibration gap too large: {gap:.3f}s")

    H0 = normalize_homography(left_sample["H"])
    H1 = normalize_homography(right_sample["H"])

    # Align signs again after normalization.
    if np.sum(H0 * H1) < 0:
        H1 = -H1

    alpha = (time - left_sample["time"]) / gap
    H = (1 - alpha) * H0 + alpha * H1

    return normalize_homography(H)


def project_grid(H: np.ndarray, pitch_points: np.ndarray) -> np.ndarray:
    """Project Nx2 pitch points into image coordinates."""
    homogeneous = np.column_stack([
        pitch_points,
        np.ones(len(pitch_points)),
    ])
    projected = (H @ homogeneous.T).T
    denominator = projected[:, 2]

    valid = np.abs(denominator) > 1e-9
    result = np.full((len(pitch_points), 2), np.nan)
    result[valid] = projected[valid, :2] / denominator[valid, None]

    return result


def fit_affine(source: np.ndarray, target: np.ndarray) -> dict:
    valid = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source, target = source[valid], target[valid]

    if len(source) < 8:
        raise ValueError("Too few valid projected grid points")

    # Robust fit to the projected pitch grid.
    A, inliers = cv2.estimateAffinePartial2D(
        source.astype(np.float32),
        target.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
        maxIters=3000,
        confidence=0.99,
        refineIters=10,
    )

    if A is None or inliers is None:
        raise ValueError("Could not fit affine transform")

    a, b = A[0, 0], A[1, 0]
    scale = float(np.hypot(a, b))
    rotation = float(np.degrees(np.arctan2(b, a)))

    predicted = cv2.transform(
        source.reshape(-1, 1, 2),
        A,
    ).reshape(-1, 2)

    residual = np.linalg.norm(predicted - target, axis=1)
    inlier_mask = inliers.ravel().astype(bool)

    return {
        "dx": float(A[0, 2]),
        "dy": float(A[1, 2]),
        "scale_percent": (scale - 1) * 100,
        "rotation_degrees": rotation,
        "grid_inlier_ratio": float(inlier_mask.mean()),
        "grid_residual_median_px": float(np.median(residual[inlier_mask])),
        "grid_residual_p90_px": float(np.percentile(residual[inlier_mask], 90)),
        "grid_points": int(len(source)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optical-flow", type=Path, required=True)
    parser.add_argument("--params-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/physical_metrics/calibration_motion_comparison"),
    )
    parser.add_argument("--max-calibration-gap", type=float, default=0.30)
    parser.add_argument("--pitch-length", type=float, default=105.0)
    parser.add_argument("--pitch-width", type=float, default=68.0)
    parser.add_argument("--max-flow-offset", type=float, default=0.06)
    args = parser.parse_args()

    flow = pd.read_csv(args.optical_flow)
    required = {
        "previous_time_seconds", "time_seconds",
        "dx_px", "dy_px", "scale_percent",
        "rotation_degrees", "inlier_ratio", "status",
    }
    missing = required - set(flow.columns)
    if missing:
        raise ValueError(f"Optical-flow CSV missing columns: {sorted(missing)}")

    samples = load_calibrations(args.params_dir)

    # Ground-plane grid in the centered world-coordinate convention:
    # X in [-L/2, L/2], Y in [-W/2, W/2].
    xs = np.linspace(-args.pitch_length / 2, args.pitch_length / 2, 15)
    ys = np.linspace(-args.pitch_width / 2, args.pitch_width / 2, 9)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    rows = []

    for _, row in flow.iterrows():
        t0 = float(row["previous_time_seconds"])
        t1 = float(row["time_seconds"])

        result = {
            "previous_time_seconds": t0,
            "time_seconds": t1,
            "observed_dx_px": row["dx_px"],
            "observed_dy_px": row["dy_px"],
            "observed_scale_percent": row["scale_percent"],
            "observed_rotation_degrees": row["rotation_degrees"],
            "optical_flow_inlier_ratio": row["inlier_ratio"],
            "optical_flow_status": row["status"],
        }

        try:
            H0 = homography_at(samples, t0, args.max_calibration_gap)
            H1 = homography_at(samples, t1, args.max_calibration_gap)

            image0 = project_grid(H0, grid)
            image1 = project_grid(H1, grid)

            fit = fit_affine(image0, image1)
            result.update({
                "calibration_status": "ok",
                "predicted_dx_px": fit["dx"],
                "predicted_dy_px": fit["dy"],
                "predicted_scale_percent": fit["scale_percent"],
                "predicted_rotation_degrees": fit["rotation_degrees"],
                "grid_inlier_ratio": fit["grid_inlier_ratio"],
                "grid_residual_median_px": fit["grid_residual_median_px"],
                "grid_residual_p90_px": fit["grid_residual_p90_px"],
                "grid_points": fit["grid_points"],
            })

            # These are comparisons between two affine approximations.
            # Perspective variation across the pitch can make them differ.
            result["translation_disagreement_px"] = float(np.hypot(
                fit["dx"] - row["dx_px"],
                fit["dy"] - row["dy_px"],
            ))
            result["scale_disagreement_percentage_points"] = float(
                fit["scale_percent"] - row["scale_percent"]
            )
            result["rotation_disagreement_degrees"] = float(
                fit["rotation_degrees"] - row["rotation_degrees"]
            )

            flow_ok = (
                row["status"] == "ok"
                and np.isfinite(row["inlier_ratio"])
                and row["inlier_ratio"] >= 0.35
                and abs(t1 - t0) <= args.max_flow_offset
            )
            result["comparison_usable"] = bool(flow_ok)

            # Initial review flag only. Thresholds are diagnostics, not
            # proof of calibration failure.
            result["review_flag"] = bool(
                flow_ok
                and (
                    result["translation_disagreement_px"] > 8.0
                    or abs(result["scale_disagreement_percentage_points"]) > 1.5
                    or abs(result["rotation_disagreement_degrees"]) > 1.0
                )
            )

        except (ValueError, np.linalg.LinAlgError) as exc:
            result.update({
                "calibration_status": f"unavailable: {exc}",
                "comparison_usable": False,
                "review_flag": False,
            })

        rows.append(result)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    result_df = pd.DataFrame(rows)
    output_csv = output_dir / "calibration_vs_optical_flow.csv"
    result_df.to_csv(output_csv, index=False)

    usable = result_df[result_df["comparison_usable"]]
    flagged = result_df[result_df["review_flag"]]

    summary = {
        "optical_flow_csv": str(args.optical_flow),
        "calibration_directory": str(args.params_dir),
        "output_csv": str(output_csv),
        "intervals_total": len(result_df),
        "calibration_comparison_available": int(
            result_df["calibration_status"].eq("ok").sum()
        ),
        "usable_comparisons": len(usable),
        "review_flags": len(flagged),
        "median_translation_disagreement_px": (
            float(usable["translation_disagreement_px"].median())
            if len(usable) else None
        ),
        "p90_translation_disagreement_px": (
            float(usable["translation_disagreement_px"].quantile(0.90))
            if len(usable) else None
        ),
        "median_abs_scale_disagreement_percentage_points": (
            float(usable["scale_disagreement_percentage_points"].abs().median())
            if len(usable) else None
        ),
        "warnings": [
            "A pitch-grid affine fit approximates a projective transformation.",
            "Observed optical flow may be affected by moving players and occlusion.",
            "Camera pose estimates are only as valid as the projection convention.",
            "Review flags are not automatic rejection decisions.",
            "No source files or player coordinates were modified.",
        ],
    }

    (output_dir / "comparison_report.json").write_text(
        json.dumps(summary, indent=2)
    )

    print("\n=== Calibration vs optical flow ===")
    print(json.dumps(summary, indent=2))

    if len(usable):
        cols = [
            "time_seconds",
            "observed_dx_px", "predicted_dx_px",
            "observed_dy_px", "predicted_dy_px",
            "observed_scale_percent", "predicted_scale_percent",
            "translation_disagreement_px", "review_flag",
        ]
        print("\nLargest translation disagreements:")
        print(
            usable.nlargest(15, "translation_disagreement_px")[cols]
            .round(4).to_string(index=False)
        )


if __name__ == "__main__":
    main()
