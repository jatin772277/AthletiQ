
#!/usr/bin/env python3
"""Estimate AthletiQ physical player metrics using existing PnLCalib parameters.

Interpolates ground-plane homographies between saved camera calibration samples.
Outputs are estimates until calibration is independently validated.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parameter_time(path: Path, data: dict[str, Any]) -> float | None:
    for key in ("time_seconds", "timestamp_seconds", "time_sec", "time"):
        value = data.get(key)
        if isinstance(value, (int, float)):
            return float(value)

    match = re.search(r"_(\d+(?:\.\d+)?)s(?:\.json)?$", path.name)
    if match:
        return float(match.group(1))

    match = re.search(r"frame_(\d+)", path.name)
    return float(match.group(1)) / 25.0 if match else None


def projection_matrix(data: dict[str, Any]) -> np.ndarray:
    cam = data.get("cam_params", data)
    fx = float(cam["x_focal_length"])
    fy = float(cam["y_focal_length"])
    cx, cy = map(float, cam["principal_point"])
    camera_position = np.asarray(
        cam["position_meters"], dtype=float
    ).reshape(3)
    rotation = np.asarray(
        cam["rotation_matrix"], dtype=float
    ).reshape(3, 3)

    intrinsic = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ])
    extrinsic = np.eye(4)[:3, :]
    extrinsic[:, 3] = -camera_position
    return intrinsic @ rotation @ extrinsic


def ground_homography(data: dict[str, Any]) -> np.ndarray:
    # PnLCalib pitch coordinates are centered at (52.5 m, 34 m).
    # For ground-plane Z=0, image coordinates use H @ [X, Y, 1].
    P = projection_matrix(data)
    H = P[:, [0, 1, 3]]

    norm = np.linalg.norm(H)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Invalid ground-plane homography")

    H = H / norm

    # H and -H represent the same projective transform. Align signs
    # before interpolating to avoid artificial cancellation.
    if abs(H[2, 2]) > 1e-10 and H[2, 2] < 0:
        H = -H

    return H


def load_calibrations(param_dir: Path) -> list[dict[str, Any]]:
    records = []
    required = (
        "x_focal_length",
        "y_focal_length",
        "principal_point",
        "position_meters",
        "rotation_matrix",
    )

    for path in sorted(param_dir.rglob("*.json")):
        try:
            data = read_json(path)
            cam = data.get("cam_params", {})

            if not all(key in cam for key in required):
                continue

            timestamp = parameter_time(path, data)
            if timestamp is None:
                continue

            H = ground_homography(data)

            if not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-12:
                continue

            records.append({
                "time": timestamp,
                "path": str(path),
                "H": H,
                "rep_err": data.get("rep_err", np.nan),
            })

        except Exception as exc:
            print(f"[WARN] Skipping {path}: {exc}")

    records.sort(key=lambda item: item["time"])

    # Avoid ambiguous interpolation if duplicate timestamps exist.
    unique = {}
    for item in records:
        unique[item["time"]] = item

    return [unique[t] for t in sorted(unique)]


def homography_at_time(
    calibrations: list[dict[str, Any]],
    timestamp: float,
):
    """Interpolate neighboring homographies; do not extrapolate beyond endpoints.

    Returns inverse_H, nearest_sample_gap, effective_time, source, rep_err.
    """
    times = np.array(
        [item["time"] for item in calibrations],
        dtype=float,
    )
    right_idx = int(np.searchsorted(times, timestamp, side="left"))

    # Before the first calibration or after the last: use endpoint sample.
    if right_idx == 0 or right_idx >= len(calibrations):
        item = calibrations[0] if right_idx == 0 else calibrations[-1]
        gap = abs(timestamp - item["time"])

        try:
            return (
                np.linalg.inv(item["H"]),
                gap,
                item["time"],
                item["path"],
                float(item["rep_err"]),
            )
        except np.linalg.LinAlgError:
            return None, gap, item["time"], item["path"], float(item["rep_err"])

    left = calibrations[right_idx - 1]
    right = calibrations[right_idx]
    span = right["time"] - left["time"]

    if span <= 0:
        item = min(
            (left, right),
            key=lambda c: abs(c["time"] - timestamp),
        )
        try:
            return (
                np.linalg.inv(item["H"]),
                abs(timestamp - item["time"]),
                item["time"],
                item["path"],
                float(item["rep_err"]),
            )
        except np.linalg.LinAlgError:
            return None, abs(timestamp - item["time"]), item["time"], item["path"], float(item["rep_err"])

    alpha = float(np.clip(
        (timestamp - left["time"]) / span,
        0.0,
        1.0,
    ))

    H_left = left["H"]
    H_right = right["H"]

    if np.sum(H_left * H_right) < 0:
        H_right = -H_right

    # Interpolate the projective mappings, not the raw camera angles.
    H = (1.0 - alpha) * H_left + alpha * H_right
    norm = np.linalg.norm(H)

    if not np.isfinite(norm) or norm < 1e-12:
        return None, min(
            abs(timestamp - left["time"]),
            abs(right["time"] - timestamp),
        ), timestamp, "invalid_interpolated_homography", np.nan

    H = H / norm
    gap = min(
        abs(timestamp - left["time"]),
        abs(right["time"] - timestamp),
    )

    source = (
        f"interpolated:{Path(left['path']).name}|"
        f"{Path(right['path']).name}"
    )

    try:
        rep_err = (
            (1.0 - alpha) * float(left["rep_err"])
            + alpha * float(right["rep_err"])
        )
    except (TypeError, ValueError):
        rep_err = np.nan

    try:
        return np.linalg.inv(H), gap, timestamp, source, rep_err
    except np.linalg.LinAlgError:
        return None, gap, timestamp, source, rep_err


def image_to_pitch(
    x_px: float,
    y_px: float,
    H_inv: np.ndarray,
) -> tuple[float, float] | None:
    q = H_inv @ np.array([x_px, y_px, 1.0], dtype=float)

    if not np.isfinite(q).all() or abs(q[2]) < 1e-9:
        return None

    # Convert centered PnLCalib coordinates to corner-origin coordinates.
    x_m = q[0] / q[2] + 52.5
    y_m = q[1] / q[2] + 34.0

    if not np.isfinite([x_m, y_m]).all():
        return None

    return float(x_m), float(y_m)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--video-id", default="football2")
    parser.add_argument("--tracks", type=Path)
    parser.add_argument("--params-dir", type=Path)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--max-calibration-gap-seconds", type=float, default=2.1)
    parser.add_argument("--max-plausible-speed-mps", type=float, default=12.0)
    args = parser.parse_args()

    root = args.project_root.resolve()

    tracks_path = args.tracks or (
        root / "outputs/joint_tracking"
        / f"{args.video_id}_player_tracks_stitched_32.csv"
    )
    params_dir = args.params_dir or (
        root / "outputs/calibration/pnlcalib/parameters"
    )
    out_dir = root / "outputs/physical_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not tracks_path.exists():
        raise FileNotFoundError(f"Tracking CSV not found: {tracks_path}")
    if not params_dir.exists():
        raise FileNotFoundError(f"Calibration directory not found: {params_dir}")

    df = pd.read_csv(tracks_path)

    required = {"frame_id", "canonical_id", "x1", "y1", "x2", "y2"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Tracking CSV missing columns: {sorted(missing)}")

    if "time_seconds" not in df.columns:
        df["time_seconds"] = df["frame_id"] / args.fps

    calibrations = load_calibrations(params_dir)

    if len(calibrations) < 2:
        raise RuntimeError(
            f"Need at least two calibration samples; found {len(calibrations)}"
        )

    # Approximate ground-contact location using the bounding-box bottom-centre.
    df["foot_x_px"] = (df["x1"] + df["x2"]) / 2.0
    df["foot_y_px"] = df["y2"]

    pitch_x = []
    pitch_y = []
    calibration_gaps = []
    calibration_times = []
    calibration_sources = []
    reprojection_errors = []

    for row in df.itertuples(index=False):
        t = float(row.time_seconds)

        H_inv, gap, cal_time, source, rep_err = homography_at_time(
            calibrations, t
        )

        xy = None if H_inv is None else image_to_pitch(
            float(row.foot_x_px),
            float(row.foot_y_px),
            H_inv,
        )

        if xy is None or gap > args.max_calibration_gap_seconds:
            pitch_x.append(np.nan)
            pitch_y.append(np.nan)
        else:
            pitch_x.append(xy[0])
            pitch_y.append(xy[1])

        calibration_gaps.append(gap)
        calibration_times.append(cal_time)
        calibration_sources.append(source)
        reprojection_errors.append(rep_err)

    df["pitch_x_m"] = pitch_x
    df["pitch_y_m"] = pitch_y
    df["calibration_gap_seconds"] = calibration_gaps
    df["calibration_time_seconds"] = calibration_times
    df["calibration_source"] = calibration_sources
    df["calibration_rep_err"] = reprojection_errors

    df["physical_coordinate_valid"] = (
        df["pitch_x_m"].notna() & df["pitch_y_m"].notna()
    )
    df["within_nominal_pitch_bounds"] = (
        df["pitch_x_m"].between(0, 105)
        & df["pitch_y_m"].between(0, 68)
    )

    df = df.sort_values(
        ["canonical_id", "time_seconds", "frame_id"]
    ).reset_index(drop=True)

    df["step_distance_m"] = np.nan
    df["step_speed_mps"] = np.nan
    df["step_dt_seconds"] = np.nan

    # Only compare consecutive observations within the same canonical ID.
    for _, indices in df.groupby("canonical_id", sort=False).groups.items():
        group = df.loc[indices]
        dt = group["time_seconds"].diff()

        distance = np.sqrt(
            group["pitch_x_m"].diff() ** 2
            + group["pitch_y_m"].diff() ** 2
        )

        valid_pair = (
            group["physical_coordinate_valid"]
            & group["physical_coordinate_valid"].shift(1, fill_value=False)
            & dt.gt(0)
            & dt.le(0.5)
            & group["calibration_gap_seconds"].le(
                args.max_calibration_gap_seconds
            )
        )

        distance = distance.where(valid_pair)
        speed = (distance / dt).where(valid_pair)

        df.loc[indices, "step_distance_m"] = distance.to_numpy()
        df.loc[indices, "step_speed_mps"] = speed.to_numpy()
        df.loc[indices, "step_dt_seconds"] = dt.where(valid_pair).to_numpy()

    df["speed_outlier_flag"] = (
        df["step_speed_mps"] > args.max_plausible_speed_mps
    )

    accepted = (
        df["step_distance_m"].notna()
        & ~df["speed_outlier_flag"]
    )
    df["step_distance_accepted_m"] = df["step_distance_m"].where(accepted)

    player_rows = []

    for player_id, group in df.groupby("canonical_id", sort=True):
        distances = group["step_distance_accepted_m"].dropna()
        speeds = group.loc[
            accepted.loc[group.index], "step_speed_mps"
        ].dropna()

        valid_coords = group[group["physical_coordinate_valid"]]
        duration = (
            float(
                valid_coords["time_seconds"].max()
                - valid_coords["time_seconds"].min()
            )
            if len(valid_coords) >= 2 else 0.0
        )

        total_distance = (
            float(distances.sum()) if len(distances) else np.nan
        )
        outlier_count = int(group["speed_outlier_flag"].sum())
        coverage = float(group["physical_coordinate_valid"].mean())

        player_rows.append({
            "canonical_id": player_id,
            "original_track_ids": (
                ";".join(map(
                    str,
                    sorted(group["mixsort_id"].dropna().unique())
                ))
                if "mixsort_id" in group.columns else ""
            ),
            "observations": int(len(group)),
            "valid_physical_coordinate_observations": int(
                group["physical_coordinate_valid"].sum()
            ),
            "physical_coordinate_coverage": coverage,
            "within_pitch_bounds_fraction": float(
                group["within_nominal_pitch_bounds"].mean()
            ),
            "duration_seconds": duration,
            "distance_travelled_m_estimate": total_distance,
            "average_speed_mps_estimate": (
                total_distance / duration
                if duration > 0 and np.isfinite(total_distance)
                else np.nan
            ),
            "mean_step_speed_mps_estimate": (
                float(speeds.mean()) if len(speeds) else np.nan
            ),
            "median_step_speed_mps_estimate": (
                float(speeds.median()) if len(speeds) else np.nan
            ),
            "maximum_step_speed_mps_estimate": (
                float(speeds.max()) if len(speeds) else np.nan
            ),
            "p95_step_speed_mps_estimate": (
                float(speeds.quantile(0.95)) if len(speeds) else np.nan
            ),
            "accepted_movement_steps": int(len(distances)),
            "speed_outlier_steps_flagged": outlier_count,
            "mean_detector_confidence": (
                float(group["confidence"].mean())
                if "confidence" in group.columns else np.nan
            ),
            "calibration_gap_median_seconds": float(
                group["calibration_gap_seconds"].median()
            ),
            "calibration_reprojection_error_median": float(
                group["calibration_rep_err"].median()
            ),
            "metrics_status": (
                "estimate_low_coverage" if coverage < 0.5
                else "estimate_review_outliers" if outlier_count > 0
                else "estimate_unvalidated_calibration"
            ),
        })

    player_metrics = pd.DataFrame(player_rows)

    point_path = out_dir / (
        f"{args.video_id}_stitched_physical_tracks.csv"
    )
    metrics_path = out_dir / (
        f"{args.video_id}_player_physical_metrics.csv"
    )
    audit_path = out_dir / (
        f"{args.video_id}_physical_metrics_audit.json"
    )

    df.to_csv(point_path, index=False)
    player_metrics.to_csv(metrics_path, index=False)

    audit = {
        "video_id": args.video_id,
        "input_tracks": str(tracks_path),
        "calibration_parameter_directory": str(params_dir),
        "point_output": str(point_path),
        "player_metrics_output": str(metrics_path),
        "input_rows": int(len(df)),
        "canonical_players": int(df["canonical_id"].nunique()),
        "calibration_samples_loaded": int(len(calibrations)),
        "calibration_time_range_seconds": [
            float(calibrations[0]["time"]),
            float(calibrations[-1]["time"]),
        ],
        "calibration_method": (
            "linear interpolation of normalized ground-plane homographies"
        ),
        "valid_coordinate_rows": int(
            df["physical_coordinate_valid"].sum()
        ),
        "valid_coordinate_fraction": float(
            df["physical_coordinate_valid"].mean()
        ),
        "rows_within_nominal_pitch_bounds": int(
            df["within_nominal_pitch_bounds"].sum()
        ),
        "speed_outlier_steps_flagged": int(
            df["speed_outlier_flag"].sum()
        ),
        "units": {
            "pitch_x_m": "metres from a pitch corner",
            "pitch_y_m": "metres from a pitch corner",
            "step_distance_m": "metres",
            "step_speed_mps": "metres per second",
        },
        "warnings": [
            "Metrics are estimates; calibration is not independently validated.",
            "Interpolation may reduce jumps but cannot correct inaccurate calibration.",
            "Player ground contact is approximated by bounding-box bottom-centre.",
            "Only consecutive observations separated by at most 0.5 seconds contribute to movement metrics.",
            f"Speeds above {args.max_plausible_speed_mps} m/s are flagged and excluded from aggregate distance, but retained in point-level output.",
            "Coordinates outside the 105 x 68 m nominal pitch are retained and flagged.",
            "Assumes calibration and tracking coordinates use the same image coordinate system.",
        ],
    }

    audit_path.write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    print("\n=== PHYSICAL METRICS COMPLETE ===")
    print(
        f"Rows: {len(df)} | canonical IDs: "
        f"{df['canonical_id'].nunique()}"
    )
    print(f"Calibration samples: {len(calibrations)}")
    print("Calibration method: interpolated ground-plane homographies")
    print(
        f"Valid projected coordinates: "
        f"{audit['valid_coordinate_rows']} "
        f"({audit['valid_coordinate_fraction']:.1%})"
    )
    print(
        f"Rows within pitch bounds: "
        f"{audit['rows_within_nominal_pitch_bounds']}"
    )
    print(
        f"Speed outlier steps flagged: "
        f"{audit['speed_outlier_steps_flagged']}"
    )
    print(f"Point-level CSV: {point_path}")
    print(f"Player metrics:  {metrics_path}")
    print(f"Audit JSON:      {audit_path}")
    print("Reminder: estimated metres/m/s, not validated ground truth.")


if __name__ == "__main__":
    main()
