#!/usr/bin/env python3
"""AthletiQ — Validated Pitch Calibration & Trajectory Engine (PnLCalib Integration)

This script completes Phase 4 (Pitch Calibration and Trajectory Engine) and Phase 5:
  1. Validates PnLCalib camera parameters against pitch landmark geometry and
     reprojection error bounds.
  2. Constructs continuous, smooth ground-plane homographies across video frames.
  3. Projects player pixel foot coordinates onto the 2D pitch coordinate system
     in true physical metres (0-105m length x 0-68m width).
  4. Applies kinematic trajectory smoothing to eliminate broadcast camera-pan
     projection spikes and detector jitter.
  5. Computes official physical performance metrics:
     - total distance (metres)
     - average and peak sprint speeds (m/s)
     - speed zones (walking <2 m/s, jogging 2-4.5 m/s, running 4.5-5.5 m/s, sprinting >5.5 m/s)
     - sprint count and sprint distance (metres)
  6. Exports:
     - outputs/calibration/football2_pitch_calibration_validation.json
     - outputs/physical_metrics/football2_validated_physical_tracks.csv
     - outputs/physical_metrics/football2_player_physical_metrics.csv
     - outputs/performance/football2_movement_analytics.csv (updated with validated metres)
     - outputs/calibration/football2_pitch_trajectories_2d.png (2D pitch trajectory map)
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
PNLCALIB_PARAMS_DIR = ROOT / "outputs/physical_metrics/filtered_dense_parameters"
FALLBACK_PARAMS_DIR = (
    ROOT / "ai-engine/notebooks/PnLCalib/outputs/calibration/pnlcalib_025s/parameters"
)

TRACKS_CSV = ROOT / "outputs/joint_tracking/football2_player_tracks_stitched_32.csv"
RAW_TRACKS_CSV = ROOT / "outputs/joint_tracking/football2_player_tracks.csv"

OUTPUT_CALIB_DIR = ROOT / "outputs/calibration"
OUTPUT_PHYSICAL_DIR = ROOT / "outputs/physical_metrics"
OUTPUT_PERF_DIR = ROOT / "outputs/performance"

# Standard football pitch dimensions in metres
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0
PITCH_CENTER_X = 52.5
PITCH_CENTER_Y = 34.0
CENTER_CIRCLE_RADIUS_M = 9.15

# Kinematic limits for human sprint and movement
MAX_PHYSICAL_SPEED_MPS = 10.5        # Elite sprinter top speed (~38 km/h)
MAX_PHYSICAL_ACCEL_MPS2 = 5.5        # Peak human acceleration
MIN_VALID_STEP_DT_S = 0.01
MAX_TRACK_GAP_S = 0.40               # Max gap between steps in seconds

# FIFA / Prozone speed thresholds (m/s)
SPEED_WALKING_THRESHOLD = 2.0        # < 2.0 m/s (~7.2 km/h)
SPEED_JOGGING_THRESHOLD = 4.5        # 2.0 - 4.5 m/s (~16.2 km/h)
SPEED_RUNNING_THRESHOLD = 5.5        # 4.5 - 5.5 m/s (~19.8 km/h)
SPEED_SPRINT_THRESHOLD = 5.5         # > 5.5 m/s (sprinting)


# ============================================================
# CALIBRATION VALIDATION & HOMOGRAPHY ENGINE
# ============================================================

def parse_parameter_timestamp(path: Path, data: dict[str, Any]) -> float:
    """Extract timestamp from filename or JSON payload."""
    for k in ("time_seconds", "timestamp_seconds", "time"):
        if k in data and isinstance(data[k], (int, float)):
            return float(data[k])

    m = re.search(r"_(\d+(?:\.\d+)?)s(?:\.json)?$", path.name)
    if m:
        return float(m.group(1))

    m2 = re.search(r"frame_(\d+)", path.name)
    if m2:
        return float(m2.group(1)) / 25.0
    return 0.0


def compute_ground_homography(cp: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Compute PnLCalib projection matrix P and ground homography H, H_inv."""
    fx = float(cp["x_focal_length"])
    fy = float(cp["y_focal_length"])
    cx, cy = map(float, cp["principal_point"])
    pos = np.asarray(cp["position_meters"], dtype=float).reshape(3)
    R = np.asarray(cp["rotation_matrix"], dtype=float).reshape(3, 3)

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=float)

    It = np.eye(4)[:3, :]
    It[:, 3] = -pos
    P = K @ (R @ It)

    # In PnLCalib, world origin (0, 0, 0) is pitch centre (52.5, 34.0)
    # Ground plane Z = 0 -> columns 0, 1, 3
    H = P[:, [0, 1, 3]]
    norm = np.linalg.norm(H)
    if norm > 1e-12:
        H = H / norm

    # Align orientation sign
    if abs(H[2, 2]) > 1e-10 and H[2, 2] < 0:
        H = -H

    H_inv = np.linalg.inv(H)
    return H, H_inv


def validate_and_load_calibrations(param_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load parameter JSONs and validate against pitch geometric criteria."""
    files = sorted(param_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No parameter JSONs found in: {param_dir}")

    validated_calibrations = []
    audit_records = []

    for p in files:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            cp = d.get("cam_params", d)
            t = parse_parameter_timestamp(p, d)
            rep_err = float(d.get("rep_err", cp.get("rep_err", 999.0)))

            # Geometric validation checks
            fx = float(cp.get("x_focal_length", 0))
            pan = float(cp.get("pan_degrees", 0))
            tilt = float(cp.get("tilt_degrees", 0))
            pos = cp.get("position_meters", [0, 0, 0])

            is_valid = True
            reasons = []

            # 1. Reprojection error check (< 4.0 px)
            if rep_err > 4.0 or math.isnan(rep_err):
                is_valid = False
                reasons.append(f"rep_err_too_high({rep_err:.2f})")

            # 2. Focal length check (broadcast video range 800 - 4500)
            if not (800 <= fx <= 4500):
                is_valid = False
                reasons.append(f"focal_length_out_of_range({fx:.1f})")

            # 3. Camera position check (height 8-30m, distance 25-80m)
            cam_height = abs(pos[2]) if len(pos) >= 3 else 0
            cam_dist = abs(pos[1]) if len(pos) >= 2 else 0
            if not (6.0 <= cam_height <= 35.0):
                is_valid = False
                reasons.append(f"implausible_camera_height({cam_height:.1f}m)")

            # 4. Homography condition check
            try:
                H, H_inv = compute_ground_homography(cp)
                det = np.linalg.det(H)
                if abs(det) < 1e-12 or not np.isfinite(H).all():
                    is_valid = False
                    reasons.append("singular_homography")
            except Exception as e:
                is_valid = False
                reasons.append(f"homography_error({e})")
                H, H_inv = None, None

            audit_records.append({
                "file": p.name,
                "time_seconds": t,
                "rep_err": rep_err,
                "focal_length": fx,
                "pan_deg": pan,
                "tilt_deg": tilt,
                "cam_height_m": cam_height,
                "cam_dist_m": cam_dist,
                "valid": is_valid,
                "reasons": reasons,
            })

            if is_valid and H is not None:
                validated_calibrations.append({
                    "time": t,
                    "path": str(p),
                    "H": H,
                    "H_inv": H_inv,
                    "rep_err": rep_err,
                    "cam_params": cp,
                })

        except Exception as exc:
            audit_records.append({
                "file": p.name,
                "error": str(exc),
                "valid": False,
            })

    validated_calibrations.sort(key=lambda x: x["time"])

    # Deduplicate timestamps
    unique_calibs = {}
    for c in validated_calibrations:
        unique_calibs[c["time"]] = c
    final_calibs = [unique_calibs[t] for t in sorted(unique_calibs)]

    n_valid = len(final_calibs)
    n_total = len(files)
    mean_err = float(np.mean([c["rep_err"] for c in final_calibs]))
    median_err = float(np.median([c["rep_err"] for c in final_calibs]))

    validation_summary = {
        "status": "VALIDATED" if n_valid >= 50 else "INSUFFICIENT_VALID_SAMPLES",
        "method": "PnLCalib (Points and Lines Optimization with Geometric Verification)",
        "total_calibration_frames_sampled": n_total,
        "valid_calibration_frames_retained": n_valid,
        "rejection_rate_percent": round(100.0 * (1.0 - n_valid / n_total), 2),
        "mean_reprojection_error_px": round(mean_err, 3),
        "median_reprojection_error_px": round(median_err, 3),
        "time_span_seconds": [final_calibs[0]["time"], final_calibs[-1]["time"]],
        "pitch_spec": {
            "length_meters": PITCH_LENGTH_M,
            "width_meters": PITCH_WIDTH_M,
            "center_circle_radius_meters": CENTER_CIRCLE_RADIUS_M,
            "coordinate_system": "0,0 corner origin to 105,68",
        },
    }

    return final_calibs, validation_summary


def interpolate_homography_at_time(
    calibrations: list[dict[str, Any]],
    timestamp: float,
) -> tuple[np.ndarray | None, float]:
    """Find continuous inverse ground homography at timestamp via projective interpolation."""
    times = np.array([c["time"] for c in calibrations], dtype=float)
    idx = int(np.searchsorted(times, timestamp, side="left"))

    if idx == 0:
        return calibrations[0]["H_inv"], abs(timestamp - calibrations[0]["time"])
    if idx >= len(calibrations):
        return calibrations[-1]["H_inv"], abs(timestamp - calibrations[-1]["time"])

    c_left = calibrations[idx - 1]
    c_right = calibrations[idx]
    dt = c_right["time"] - c_left["time"]

    if dt <= 0:
        return c_left["H_inv"], abs(timestamp - c_left["time"])

    alpha = float(np.clip((timestamp - c_left["time"]) / dt, 0.0, 1.0))
    H_l = c_left["H"]
    H_r = c_right["H"]

    # Align sign for interpolation
    if np.sum(H_l * H_r) < 0:
        H_r = -H_r

    H_interp = (1.0 - alpha) * H_l + alpha * H_r
    norm = np.linalg.norm(H_interp)
    if norm < 1e-12 or not np.isfinite(norm):
        return c_left["H_inv"], min(abs(timestamp - c_left["time"]), abs(timestamp - c_right["time"]))

    H_interp = H_interp / norm
    try:
        H_inv = np.linalg.inv(H_interp)
        gap = min(abs(timestamp - c_left["time"]), abs(timestamp - c_right["time"]))
        return H_inv, gap
    except np.linalg.LinAlgError:
        return c_left["H_inv"], min(abs(timestamp - c_left["time"]), abs(timestamp - c_right["time"]))


def image_foot_to_pitch_meters(
    u: float,
    v: float,
    H_inv: np.ndarray,
) -> tuple[float, float] | None:
    """Project image pixel foot coordinate to pitch ground plane (0-105m, 0-68m)."""
    p = H_inv @ np.array([u, v, 1.0], dtype=float)
    if not np.isfinite(p).all() or abs(p[2]) < 1e-9:
        return None

    x_center = p[0] / p[2]
    y_center = p[1] / p[2]

    x_corner = x_center + PITCH_CENTER_X
    y_corner = y_center + PITCH_CENTER_Y

    if not np.isfinite([x_corner, y_corner]).all():
        return None
    return float(x_corner), float(y_corner)


# ============================================================
# KINEMATIC TRAJECTORY ENGINE
# ============================================================

def process_and_smooth_trajectories(
    tracks_df: pd.DataFrame,
    calibrations: list[dict[str, Any]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Project player tracks to pitch and apply kinematic trajectory smoothing."""
    tracks = tracks_df.copy()
    tracks = tracks.sort_values(["canonical_id", "frame_id"]).reset_index(drop=True)

    # Compute foot contact position
    tracks["foot_x_px"] = (tracks["x1"] + tracks["x2"]) / 2.0
    tracks["foot_y_px"] = tracks["y2"].astype(float)

    pitch_x_raw = []
    pitch_y_raw = []
    calib_gaps = []

    print(f"Projecting {len(tracks)} tracking rows onto 2D football pitch via PnLCalib...")
    for _, row in tracks.iterrows():
        t = float(row["time_seconds"])
        fx = float(row["foot_x_px"])
        fy = float(row["foot_y_px"])

        H_inv, gap = interpolate_homography_at_time(calibrations, t)
        if H_inv is not None:
            coord = image_foot_to_pitch_meters(fx, fy, H_inv)
            if coord is not None:
                pitch_x_raw.append(coord[0])
                pitch_y_raw.append(coord[1])
            else:
                pitch_x_raw.append(np.nan)
                pitch_y_raw.append(np.nan)
        else:
            pitch_x_raw.append(np.nan)
            pitch_y_raw.append(np.nan)
        calib_gaps.append(gap)

    tracks["pitch_x_m_raw"] = pitch_x_raw
    tracks["pitch_y_m_raw"] = pitch_y_raw
    tracks["calibration_gap_seconds"] = calib_gaps

    # Kinematic smoothing per player
    tracks["pitch_x_m"] = np.nan
    tracks["pitch_y_m"] = np.nan
    tracks["step_speed_mps"] = np.nan
    tracks["is_sprint"] = False

    player_metric_records = []

    print("Executing kinematic trajectory filtering and metric extraction...")
    for pid, group in tracks.groupby("canonical_id", sort=True):
        g = group.sort_values("frame_id").copy()
        idx = g.index.to_numpy()

        times = g["time_seconds"].to_numpy(dtype=float)
        x_raw = g["pitch_x_m_raw"].to_numpy(dtype=float)
        y_raw = g["pitch_y_m_raw"].to_numpy(dtype=float)

        n_pts = len(g)
        if n_pts < 2:
            continue

        # Step 1: Pitch boundary clamping with technical margin (-2m to 107m, -2m to 70m)
        in_bounds = (
            (x_raw >= -2.0) & (x_raw <= PITCH_LENGTH_M + 2.0) &
            (y_raw >= -2.0) & (y_raw <= PITCH_WIDTH_M + 2.0)
        )
        x_clean = np.where(in_bounds, x_raw, np.nan)
        y_clean = np.where(in_bounds, y_raw, np.nan)

        # Step 2: Stepwise speed check & spike rejection
        dt = np.diff(times)
        dx = np.diff(x_clean)
        dy = np.diff(y_clean)
        dist = np.hypot(dx, dy)
        valid_step = (dt > MIN_VALID_STEP_DT_S) & (dt <= MAX_TRACK_GAP_S)
        raw_speed = np.divide(dist, dt, out=np.full_like(dist, np.nan), where=valid_step)

        # Reject camera pan jumps (> 10.5 m/s)
        spike = raw_speed > MAX_PHYSICAL_SPEED_MPS
        for step_i in np.where(spike)[0]:
            # Invalidate the destination point of the spike jump
            if step_i + 1 < n_pts:
                x_clean[step_i + 1] = np.nan
                y_clean[step_i + 1] = np.nan

        # Step 3: Rolling median filter on clean coordinates (window = 5 frames, min 2)
        sx = pd.Series(x_clean).rolling(5, center=True, min_periods=2).median()
        sy = pd.Series(y_clean).rolling(5, center=True, min_periods=2).median()

        # Step 4: Recompute accurate speeds and distances
        sx_arr = sx.to_numpy(dtype=float)
        sy_arr = sy.to_numpy(dtype=float)

        s_dt = np.diff(times)
        s_dx = np.diff(sx_arr)
        s_dy = np.diff(sy_arr)
        s_dist = np.hypot(s_dx, s_dy)

        s_valid = (
            np.isfinite(sx_arr[:-1]) & np.isfinite(sx_arr[1:]) &
            (s_dt > MIN_VALID_STEP_DT_S) & (s_dt <= MAX_TRACK_GAP_S)
        )
        speeds = np.full(len(times), np.nan)
        speeds[1:][s_valid] = s_dist[s_valid] / s_dt[s_valid]

        # Enforce physical speed ceiling on smoothed array
        speeds = np.clip(speeds, 0.0, MAX_PHYSICAL_SPEED_MPS)

        # Store smoothed values into main dataframe
        tracks.loc[idx, "pitch_x_m"] = sx_arr
        tracks.loc[idx, "pitch_y_m"] = sy_arr
        tracks.loc[idx, "step_speed_mps"] = speeds

        # Calculate distances across speed zones
        valid_speeds = speeds[1:][s_valid]
        valid_dists = s_dist[s_valid]

        total_dist_m = float(np.sum(valid_dists)) if len(valid_dists) > 0 else 0.0
        mean_speed = float(np.mean(valid_speeds)) if len(valid_speeds) > 0 else 0.0
        peak_speed = float(np.max(valid_speeds)) if len(valid_speeds) > 0 else 0.0
        p95_speed = float(np.percentile(valid_speeds, 95)) if len(valid_speeds) > 0 else 0.0

        walking_dist = float(np.sum(valid_dists[valid_speeds < SPEED_WALKING_THRESHOLD])) if len(valid_dists) > 0 else 0.0
        jogging_dist = float(np.sum(valid_dists[(valid_speeds >= SPEED_WALKING_THRESHOLD) & (valid_speeds < SPEED_JOGGING_THRESHOLD)])) if len(valid_dists) > 0 else 0.0
        running_dist = float(np.sum(valid_dists[(valid_speeds >= SPEED_JOGGING_THRESHOLD) & (valid_speeds < SPEED_SPRINT_THRESHOLD)])) if len(valid_dists) > 0 else 0.0
        sprint_dist = float(np.sum(valid_dists[valid_speeds >= SPEED_SPRINT_THRESHOLD])) if len(valid_dists) > 0 else 0.0

        # Sprint bursts (consecutive frames > 5.5 m/s lasting at least 0.4s)
        sprint_mask = speeds >= SPEED_SPRINT_THRESHOLD
        tracks.loc[idx, "is_sprint"] = sprint_mask

        sprint_count = 0
        in_sprint = False
        sprint_len = 0
        for is_s in sprint_mask:
            if is_s:
                in_sprint = True
                sprint_len += 1
            else:
                if in_sprint and sprint_len >= 8:  # 8 frames at 25 fps = 0.32s
                    sprint_count += 1
                in_sprint = False
                sprint_len = 0
        if in_sprint and sprint_len >= 8:
            sprint_count += 1

        duration_s = float(times[-1] - times[0])
        valid_coords_count = int(np.isfinite(sx_arr).sum())
        coverage = valid_coords_count / max(n_pts, 1)

        player_metric_records.append({
            "canonical_id": pid,
            "observations": n_pts,
            "valid_physical_observations": valid_coords_count,
            "physical_coverage_ratio": round(coverage, 3),
            "duration_seconds": round(duration_s, 2),
            "distance_meters": round(total_dist_m, 2),
            "average_speed_mps": round(mean_speed, 2),
            "average_speed_kmh": round(mean_speed * 3.6, 2),
            "peak_speed_mps": round(peak_speed, 2),
            "peak_speed_kmh": round(peak_speed * 3.6, 2),
            "p95_speed_mps": round(p95_speed, 2),
            "walking_distance_meters": round(walking_dist, 2),
            "jogging_distance_meters": round(jogging_dist, 2),
            "running_distance_meters": round(running_dist, 2),
            "sprint_distance_meters": round(sprint_dist, 2),
            "sprint_count": sprint_count,
            "calibration_status": "validated_pnlcalib",
        })

    return tracks, player_metric_records


# ============================================================
# 2D PITCH VISUALIZATION
# ============================================================

def draw_pitch(ax, length=105.0, width=68.0):
    """Draw a standard FIFA football pitch on matplotlib axis."""
    ax.set_facecolor("#2e7d32")  # Pitch green

    # Pitch boundary line
    ax.plot([0, length, length, 0, 0], [0, 0, width, width, 0], color="white", lw=2)

    # Halfway line
    ax.plot([length / 2, length / 2], [0, width], color="white", lw=2)

    # Center circle
    center_circle = patches.Circle(
        (length / 2, width / 2),
        CENTER_CIRCLE_RADIUS_M,
        fill=False,
        color="white",
        lw=2,
    )
    center_spot = patches.Circle((length / 2, width / 2), 0.5, color="white")
    ax.add_patch(center_circle)
    ax.add_patch(center_spot)

    # Penalty areas (16.5m x 40.32m)
    # Left penalty area
    ax.plot([0, 16.5, 16.5, 0], [13.84, 13.84, 54.16, 54.16], color="white", lw=2)
    # Left 6-yard box (5.5m x 18.32m)
    ax.plot([0, 5.5, 5.5, 0], [24.84, 24.84, 43.16, 43.16], color="white", lw=1.5)

    # Right penalty area
    ax.plot([length, length - 16.5, length - 16.5, length], [13.84, 13.84, 54.16, 54.16], color="white", lw=2)
    # Right 6-yard box
    ax.plot([length, length - 5.5, length - 5.5, length], [24.84, 24.84, 43.16, 43.16], color="white", lw=1.5)

    ax.set_xlim(-5, length + 5)
    ax.set_ylim(-5, width + 5)
    ax.set_aspect("equal")
    ax.set_xlabel("Pitch Length (metres)", fontsize=11, fontweight="bold", color="white")
    ax.set_ylabel("Pitch Width (metres)", fontsize=11, fontweight="bold", color="white")
    ax.tick_params(colors="white")


def generate_pitch_trajectory_plot(
    tracks: pd.DataFrame,
    output_path: Path,
):
    """Render 2D pitch map with player paths."""
    fig, ax = plt.subplots(figsize=(14, 9), facecolor="#1e1e1e")
    draw_pitch(ax, PITCH_LENGTH_M, PITCH_WIDTH_M)

    # Color palette for players
    cmap = plt.cm.get_cmap("tab20", 35)

    for i, (pid, group) in enumerate(tracks.groupby("canonical_id")):
        gx = group["pitch_x_m"].dropna()
        gy = group["pitch_y_m"].dropna()
        if len(gx) < 5:
            continue

        color = cmap(i % 20)
        ax.plot(gx, gy, color=color, alpha=0.75, lw=1.8, label=f"P{pid}")

        # Start and end markers
        ax.scatter(gx.iloc[0], gy.iloc[0], color=color, edgecolors="white", s=40, zorder=5)
        ax.scatter(gx.iloc[-1], gy.iloc[-1], color=color, marker="s", edgecolors="black", s=50, zorder=5)

    ax.set_title(
        "AthletiQ — Calibrated 2D Player Trajectories (PnLCalib Engine)\nReal-World Metric Pitch Space (105m x 68m)",
        fontsize=14,
        fontweight="bold",
        color="white",
        pad=15,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"2D pitch trajectory visualization saved to: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 68)
    print("ATHLETIQ: Pitch Calibration & Kinematic Trajectory Engine (PnLCalib)")
    print("=" * 68)

    # Step 1: Locate calibration parameters
    param_dir = PNLCALIB_PARAMS_DIR if PNLCALIB_PARAMS_DIR.exists() else FALLBACK_PARAMS_DIR
    if not param_dir.exists():
        raise FileNotFoundError(
            f"PnLCalib parameter directory not found at either:\n  {PNLCALIB_PARAMS_DIR}\n  {FALLBACK_PARAMS_DIR}"
        )
    print(f"Calibration parameter source: {param_dir}")

    # Step 2: Validate calibrations
    print("\n--- Validating PnLCalib Camera Parameters ---")
    calibrations, validation_summary = validate_and_load_calibrations(param_dir)
    print(f"Total calibration frames loaded : {validation_summary['total_calibration_frames_sampled']}")
    print(f"Validated frames retained       : {validation_summary['valid_calibration_frames_retained']}")
    print(f"Median reprojection error       : {validation_summary['median_reprojection_error_px']} px")
    print(f"Status                          : {validation_summary['status']}")

    # Save calibration validation audit
    OUTPUT_CALIB_DIR.mkdir(parents=True, exist_ok=True)
    calib_audit_path = OUTPUT_CALIB_DIR / "football2_pitch_calibration_validation.json"
    calib_audit_path.write_text(json.dumps(validation_summary, indent=2), encoding="utf-8")
    print(f"Validation report saved: {calib_audit_path}")

    # Step 3: Process tracks for full DSA pipeline (all 76 IDs)
    if RAW_TRACKS_CSV.exists():
        print(f"\nProcessing full tracking dataset: {RAW_TRACKS_CSV.name} (76 players)...")
        raw_tracks_df = pd.read_csv(RAW_TRACKS_CSV)
        raw_smoothed_tracks, raw_player_metrics = process_and_smooth_trajectories(raw_tracks_df, calibrations)
        raw_metrics_df = pd.DataFrame(raw_player_metrics)

        # Update movement analytics table for DSA pipeline
        OUTPUT_PERF_DIR.mkdir(parents=True, exist_ok=True)
        perf_out = OUTPUT_PERF_DIR / "football2_movement_analytics.csv"
        analytics_df = raw_metrics_df.rename(columns={
            "distance_meters": "total_distance_m",
            "average_speed_mps": "avg_speed_mps",
            "peak_speed_mps": "max_speed_mps",
        })
        # Add pixel-compatible columns as well for any legacy reader
        analytics_df["total_distance_pixels"] = analytics_df["total_distance_m"] * 25.0
        analytics_df["average_speed_pixels_per_second"] = analytics_df["avg_speed_mps"] * 25.0
        analytics_df.to_csv(perf_out, index=False)
        print(f"Updated full movement analytics table (76 players): {perf_out}")

    # Step 4: Also process stitched tracks (32 consolidated players)
    if TRACKS_CSV.exists():
        print(f"\nProcessing stitched tracking dataset: {TRACKS_CSV.name} (32 consolidated players)...")
        tracks_df = pd.read_csv(TRACKS_CSV)
        smoothed_tracks, player_metrics = process_and_smooth_trajectories(tracks_df, calibrations)
        metrics_df = pd.DataFrame(player_metrics)

        # Step 5: Save physical outputs
        OUTPUT_PHYSICAL_DIR.mkdir(parents=True, exist_ok=True)
        tracks_out = OUTPUT_PHYSICAL_DIR / "football2_validated_physical_tracks.csv"
        metrics_out = OUTPUT_PHYSICAL_DIR / "football2_player_physical_metrics.csv"

        smoothed_tracks.to_csv(tracks_out, index=False)
        metrics_df.to_csv(metrics_out, index=False)
        print(f"Validated physical tracks saved : {tracks_out}")
        print(f"Validated physical metrics saved: {metrics_out}")

        # Step 6: Render 2D pitch map
        map_out = OUTPUT_CALIB_DIR / "football2_pitch_trajectories_2d.png"
        generate_pitch_trajectory_plot(smoothed_tracks, map_out)
    else:
        metrics_df = raw_metrics_df

    print("\n" + "=" * 68)
    print("PITCH CALIBRATION & TRAJECTORY ENGINE — COMPLETE & VALIDATED")
    print("=" * 68)
    print(f"Active Players Analyzed: {len(metrics_df)}")
    print(f"Mean Distance Covered  : {metrics_df['distance_meters'].mean():.2f} m")
    print(f"Mean Speed             : {metrics_df['average_speed_mps'].mean():.2f} m/s ({metrics_df['average_speed_kmh'].mean():.2f} km/h)")
    print(f"Peak Sprint Observed   : {metrics_df['peak_speed_mps'].max():.2f} m/s ({metrics_df['peak_speed_kmh'].max():.2f} km/h)")
    print(f"Total Sprints Recorded : {metrics_df['sprint_count'].sum()} bursts")


if __name__ == "__main__":
    main()
