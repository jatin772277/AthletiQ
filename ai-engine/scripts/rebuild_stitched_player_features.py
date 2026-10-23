#!/usr/bin/env python3
"""Rebuild per-player movement features from AthletiQ stitched tracking CSVs.

Example:
  python ai-engine/scripts/rebuild_stitched_player_features.py --video-id football2

Writes a new feature CSV and audit JSON. Does not overwrite existing DSA files.
This computes movement/trajectory features only; possession, team, KD-tree, and
interaction-graph features must be recomputed by their corresponding pipelines.
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = {
    "frame_id", "time_seconds", "canonical_id", "x1", "y1", "x2", "y2",
    "center_x", "center_y", "confidence",
}

def args_parse():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--video-id", default="football2")
    p.add_argument("--input", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--max-gap-seconds", type=float, default=0.5)
    p.add_argument("--sprint-threshold-px-s", type=float, default=180.0)
    return p.parse_args()

def numeric(series):
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)

def agg(group, video_id, fps, max_gap, sprint_threshold):
    g = group.sort_values(["frame_id", "time_seconds"]).drop_duplicates("frame_id").copy()
    frames = numeric(g.frame_id).to_numpy(float)
    times = numeric(g.time_seconds).to_numpy(float)
    times = np.where(np.isfinite(times), times, frames / fps)
    x, y = numeric(g.center_x).to_numpy(float), numeric(g.center_y).to_numpy(float)
    valid = np.isfinite(frames) & np.isfinite(times) & np.isfinite(x) & np.isfinite(y)
    frames, times, x, y = frames[valid], times[valid], x[valid], y[valid]
    if not len(frames):
        raise ValueError("Track group has no valid observations.")
    order = np.argsort(frames)
    frames, times, x, y = frames[order], times[order], x[order], y[order]
    dt = np.diff(times)
    dist = np.hypot(np.diff(x), np.diff(y))
    valid_step = np.isfinite(dt) & (dt > 0) & (dt <= max_gap) & np.isfinite(dist)
    speeds = np.full(len(dt), np.nan)
    speeds[valid_step] = dist[valid_step] / dt[valid_step]
    valid_speeds = speeds[np.isfinite(speeds)]
    valid_dist = float(dist[valid_step].sum())
    valid_time = float(dt[valid_step].sum())
    acceleration = np.diff(valid_speeds) * fps if len(valid_speeds) > 1 else np.array([])
    span_frames = max(float(frames[-1] - frames[0] + 1), 1.0)
    coverage = min(len(np.unique(frames)) / span_frames, 1.0)
    sprint_mask = valid_speeds >= sprint_threshold
    original_ids = []
    if "original_track_id" in g:
        original_ids = sorted(numeric(g.original_track_id).dropna().astype(int).unique().tolist())
    row = {
        "video_id": video_id,
        "canonical_id": int(numeric(g.canonical_id).iloc[0]),
        "first_frame": int(frames.min()),
        "last_frame": int(frames.max()),
        "observed_frames": int(len(np.unique(frames))),
        "track_span_seconds": float(times.max() - times.min()),
        "movement_coverage_ratio": float(coverage),
        "movement_distance_pixels": valid_dist,
        "movement_valid_movement_time_seconds": valid_time,
        "movement_valid_movement_steps": int(valid_step.sum()),
        "movement_excluded_gap_steps": int((~valid_step).sum()),
        "movement_average_speed_pixels_per_second": valid_dist / valid_time if valid_time > 0 else np.nan,
        "movement_median_step_speed_pixels_per_second": float(np.median(valid_speeds)) if len(valid_speeds) else np.nan,
        "movement_maximum_step_speed_pixels_per_second": float(np.max(valid_speeds)) if len(valid_speeds) else np.nan,
        "movement_maximum_positive_acceleration_pixels_per_second_squared": float(np.max(acceleration[acceleration > 0])) if np.any(acceleration > 0) else np.nan,
        "movement_maximum_deceleration_magnitude_pixels_per_second_squared": float(abs(np.min(acceleration[acceleration < 0]))) if np.any(acceleration < 0) else np.nan,
        "movement_experimental_sprint_threshold_pixels_per_second": sprint_threshold,
        "movement_experimental_sprint_count": int(sprint_mask.sum()),
        "movement_experimental_sprint_distance_pixels": float(dist[valid_step][sprint_mask].sum()) if len(sprint_mask) == int(valid_step.sum()) else np.nan,
        "mean_detection_confidence": float(numeric(g.confidence).mean()),
        "mean_bbox_width_pixels": float((numeric(g.x2) - numeric(g.x1)).mean()),
        "mean_bbox_height_pixels": float((numeric(g.y2) - numeric(g.y1)).mean()),
        "mean_center_x_pixels": float(np.mean(x)),
        "mean_center_y_pixels": float(np.mean(y)),
        "start_center_x_pixels": float(x[0]),
        "start_center_y_pixels": float(y[0]),
        "end_center_x_pixels": float(x[-1]),
        "end_center_y_pixels": float(y[-1]),
    }
    if "original_track_id" in g:
        row["original_track_ids"] = ",".join(map(str, original_ids))
        row["original_track_count"] = len(original_ids)
    return row

def main():
    args = args_parse()
    root = args.project_root.resolve()
    input_path = args.input or root / "outputs/joint_tracking" / f"{args.video_id}_player_tracks_stitched_32.csv"
    if not input_path.exists() and args.input is None and args.video_id == "football2":
        input_path = root / "outputs/joint_tracking/football2_player_tracks_stitched_32.csv"
    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")
    output_path = args.output or root / "outputs/dsa" / f"{args.video_id}_stitched_movement_features.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")
    if df.empty or df.canonical_id.isna().any() or df.frame_id.isna().any():
        raise SystemExit("Input is empty or has missing canonical_id/frame_id values.")
    df["canonical_id"] = pd.to_numeric(df.canonical_id, errors="raise").astype(int)
    df["frame_id"] = pd.to_numeric(df.frame_id, errors="raise").astype(int)
    features = pd.DataFrame([
        agg(g, args.video_id, args.fps, args.max_gap_seconds, args.sprint_threshold_px_s)
        for _, g in df.groupby("canonical_id", sort=True)
    ]).sort_values(["video_id", "canonical_id"]).reset_index(drop=True)
    if features.duplicated(["video_id", "canonical_id"]).any():
        raise RuntimeError("Duplicate (video_id, canonical_id) feature keys.")
    features.to_csv(output_path, index=False)
    audit = {
        "project": "AthletiQ",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(input_path),
        "output_csv": str(output_path),
        "video_id": args.video_id,
        "fps": args.fps,
        "max_gap_seconds_for_movement": args.max_gap_seconds,
        "experimental_sprint_threshold_pixels_per_second": args.sprint_threshold_px_s,
        "input_rows": int(len(df)),
        "unique_canonical_ids": int(df.canonical_id.nunique()),
        "feature_rows": int(len(features)),
        "unique_key": ["video_id", "canonical_id"],
        "limitations": [
            "Image-space distances and speeds are not calibrated physical metrics.",
            "Sprint threshold is experimental and requires calibration.",
            "Possession, team, KD-tree and interaction-graph features are not recomputed.",
            "Automatically stitched identities are not ground-truth verified."
        ]
    }
    audit_path = output_path.with_name(output_path.stem + "_audit.json")
    audit_path.write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    print("ATHLETIQ STITCHED MOVEMENT FEATURES")
    print("Input:", input_path)
    print("Output:", output_path)
    print("Audit:", audit_path)
    print("Input rows:", len(df))
    print("Canonical IDs:", df.canonical_id.nunique())
    print("Feature rows:", len(features))
    print("Unique (video_id, canonical_id):", not features.duplicated(["video_id", "canonical_id"]).any())
    print("\nPreview:")
    print(features.head(10).to_string(index=False))
    print("\nSupervised performance model: NOT TRAINED (reviewed scores required).")

if __name__ == "__main__":
    main()
