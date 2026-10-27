#!/usr/bin/env python3
"""Generate AthletiQ ML training data from SportsMOT football ground-truth sequences.

SportsMOT GT format (comma-separated, no header):
  frame_id, track_id, x_topleft, y_topleft, width, height,
  is_active, class_id, visibility

This script:
  1. Walks all train/val sequences in the football SportsMOT dataset
     (located at datasets/football/processed/sportsmot_yolo and its GT at
     datasets/football/benchmarks/sportsmot/extracted).
  2. Computes per-player movement metrics from GT bounding-box
     trajectories (pixel-space foot positions).
  3. Clips computed metrics to physically plausible ranges calibrated
     specifically for broadcast football camera dynamics (outliers are
     clamped, not dropped, so no player tracks are lost).
  4. Derives a relative activity score for each player WITHIN each
     sequence (0-100 percentile rank of total distance travelled)
     and an overall cross-sequence normalised distance score.
  5. Writes the merged feature+label table ready for model training.

Outputs
-------
outputs/sportsmot_ml/
  sportsmot_ml_features.csv          - per-player feature rows
  sportsmot_ml_labels.csv            - per-player label rows
  sportsmot_ml_merged.csv            - features + labels (training-ready)
  sportsmot_ml_audit.json            - summary statistics
  sportsmot_ml_clip_report.csv       - how many values were clipped per metric
  sportsmot_sequence_stats.csv       - sequence-level statistics
"""

from __future__ import annotations

import argparse
import configparser
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YOLO_ROOT = ROOT / "datasets/football/processed/sportsmot_yolo"
DEFAULT_GT_ROOT = ROOT / "datasets/football/benchmarks/sportsmot/extracted"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/sportsmot_ml"

SPLITS = ["train", "val", "test"]

# ============================================================
# MOTION COMPUTATION PARAMETERS
# ============================================================

# Only use detections that were active (column 6 == 1).
REQUIRE_ACTIVE = True

# Maximum frame gap allowed between two consecutive detections
# for the pair to contribute a valid movement step.
# At 25 fps, 3 frames = 0.12 s -- a brief occlusion.
MAX_GAP_FRAMES = 3

# Minimum observations per player to keep them in the dataset.
MIN_OBSERVATIONS = 10

# Coordinate smoothing window (frames) -- odd number.
COORD_SMOOTH_WINDOW = 5

# ============================================================
# PHYSICAL PLAUSIBILITY CLIPPING BOUNDS
# (pixel-space; 1280x720 broadcast football video, 25 fps sequences)
#
# Calibrated from actual SportsMOT football GT sequences:
#   Step-speed distribution across 617 players / 30 sequences:
#     p50 = 97.08 px/s, p75 = 170.66 px/s, p90 = 279.03 px/s
#     p95 = 375.13 px/s, p99 = 589.90 px/s, p99.9 = 877.70 px/s
#     max = 5730.63 px/s (tracking / bounding box jump glitch)
#   Absolute acceleration distribution:
#     p50 = 203.48 px/s^2, p90 = 645.24 px/s^2, p95 = 871.97 px/s^2
#     p99 = 1642.72 px/s^2, max = 125511.80 px/s^2 (glitch)
#   Total distance distribution:
#     p50 = 2109 px, p95 = 4391 px, max = 6708 px
#   Sprint counts:
#     p50 = 8, p90 = 15, p95 = 17, p99 = 22, max = 31
# ============================================================

# Speed bounds (pixels per second)
SPEED_MIN_PX_S = 0.0
SPEED_MAX_PX_S = 700.0          # Captures >99.5% of real sprint speeds; clamps jumping bbox artifacts

# Acceleration bounds (pixels per second squared)
ACCEL_MIN_PX_S2 = 0.0
ACCEL_MAX_PX_S2 = 3000.0        # Captures >99.5% of real accelerations; clamps instant teleport jumps

# Distance bounds (pixels per sequence)
DISTANCE_MIN_PX = 0.0
DISTANCE_MAX_PX = 20000.0       # Max real distance in ~30s sequence is ~6700 px

# Sprint count
SPRINT_COUNT_MAX = 40.0         # Max real sprint count in football sequence is 31

# Sprint distance
SPRINT_DISTANCE_MAX_PX = 15000.0

# Sprint duration
SPRINT_DURATION_MAX_S = 60.0

# ============================================================
# GT COLUMN INDICES (MOT format)
# ============================================================

COL_FRAME = 0
COL_TRACK = 1
COL_X = 2       # x top-left
COL_Y = 3       # y top-left
COL_W = 4
COL_H = 5
COL_ACTIVE = 6  # 1 = active, 0 = ignored


# ============================================================
# HELPERS
# ============================================================


def read_seqinfo(seq_dir: Path) -> dict[str, Any]:
    """Parse seqinfo.ini and return a dict with fps, width, height, length."""
    ini_path = seq_dir / "seqinfo.ini"
    if not ini_path.exists():
        return {"fps": 25.0, "width": 1280, "height": 720, "length": None}

    cfg = configparser.ConfigParser()
    cfg.read(ini_path, encoding="utf-8")
    seq = cfg["Sequence"] if "Sequence" in cfg else {}

    return {
        "fps": float(seq.get("frameRate", 25)),
        "width": int(seq.get("imWidth", 1280)),
        "height": int(seq.get("imHeight", 720)),
        "length": int(seq.get("seqLength", 0)),
    }


def load_gt(gt_path: Path, require_active: bool = True):
    """Load ground-truth detections from gt.txt."""
    if not gt_path.exists():
        return None

    try:
        df = pd.read_csv(
            gt_path,
            header=None,
            sep=r"[,\s]+",
            engine="python",
        )
    except Exception as exc:
        warnings.warn(f"[LOAD ERROR] {gt_path}: {exc}")
        return None

    if df.shape[1] < 7:
        warnings.warn(f"[FMT] {gt_path}: only {df.shape[1]} columns, expected >=7")
        return None

    df.columns = range(df.shape[1])

    df = df.rename(
        columns={
            COL_FRAME: "frame_id",
            COL_TRACK: "track_id",
            COL_X: "x",
            COL_Y: "y",
            COL_W: "w",
            COL_H: "h",
            COL_ACTIVE: "is_active",
        }
    )

    for col in ("frame_id", "track_id", "x", "y", "w", "h", "is_active"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["frame_id", "track_id", "x", "y", "w", "h"])
    df = df.astype({"frame_id": int, "track_id": int})

    if require_active:
        df = df[df["is_active"] == 1].copy()

    return df if not df.empty else None


def smooth_coords(series: pd.Series, window: int) -> pd.Series:
    """Centre-align rolling mean; shorter window at edges."""
    return series.rolling(window, center=True, min_periods=1).mean()


def compute_player_metrics(
    df,
    fps: float,
    seq_id: str,
    split: str,
) -> list[dict]:
    """Compute per-player movement features for one sequence."""
    df = df.copy()
    df["foot_x"] = df["x"] + df["w"] / 2.0
    df["foot_y"] = df["y"] + df["h"]

    df = df.sort_values(["track_id", "frame_id"]).reset_index(drop=True)

    records = []

    for track_id, grp in df.groupby("track_id", sort=True):
        grp = grp.sort_values("frame_id").reset_index(drop=True)

        n_obs = len(grp)
        if n_obs < MIN_OBSERVATIONS:
            continue

        fx = smooth_coords(grp["foot_x"], COORD_SMOOTH_WINDOW)
        fy = smooth_coords(grp["foot_y"], COORD_SMOOTH_WINDOW)

        first_frame = int(grp["frame_id"].iloc[0])
        last_frame = int(grp["frame_id"].iloc[-1])
        span_frames = last_frame - first_frame
        span_seconds = span_frames / fps

        frame_gap = grp["frame_id"].diff()
        dt = frame_gap / fps

        dx = fx.diff()
        dy = fy.diff()
        step_dist = np.sqrt(dx ** 2 + dy ** 2)

        valid = frame_gap.notna() & (frame_gap > 0) & (frame_gap <= MAX_GAP_FRAMES)
        step_dist_valid = step_dist.where(valid)
        dt_valid = dt.where(valid)

        step_speed = (step_dist_valid / dt_valid).fillna(np.nan)

        step_accel = step_speed.diff() / dt_valid
        positive_accel = step_accel.where(step_accel > 0)
        decel_mag = (-step_accel).where(step_accel < 0)

        total_distance = float(step_dist_valid.sum())
        n_valid_steps = int(step_dist_valid.notna().sum())

        speeds_valid = step_speed.dropna()

        avg_speed = float(speeds_valid.mean()) if not speeds_valid.empty else np.nan
        max_speed = float(speeds_valid.max()) if not speeds_valid.empty else np.nan
        median_speed = float(speeds_valid.median()) if not speeds_valid.empty else np.nan

        max_pos_accel = (
            float(positive_accel.max()) if positive_accel.notna().any() else np.nan
        )
        max_decel_mag = (
            float(decel_mag.max()) if decel_mag.notna().any() else np.nan
        )

        if len(speeds_valid) >= 5:
            sprint_thresh = float(speeds_valid.quantile(0.75))
        else:
            sprint_thresh = np.nan

        sprint_count = 0
        sprint_dist = 0.0
        sprint_dur = 0.0

        if sprint_thresh is not np.nan and np.isfinite(sprint_thresh):
            in_sprint = False
            cur_dist = 0.0
            cur_dur = 0.0

            for i in range(len(grp)):
                s = step_speed.iloc[i] if i < len(step_speed) else np.nan
                d = step_dist_valid.iloc[i] if i < len(step_dist_valid) else np.nan
                t = dt_valid.iloc[i] if i < len(dt_valid) else np.nan

                s_ok = isinstance(s, float) and np.isfinite(s)
                d_ok = isinstance(d, float) and np.isfinite(d)
                t_ok = isinstance(t, float) and np.isfinite(t)

                if s_ok and s > sprint_thresh:
                    in_sprint = True
                    cur_dist += d if d_ok else 0.0
                    cur_dur += t if t_ok else 0.0
                else:
                    if in_sprint:
                        sprint_count += 1
                        sprint_dist += cur_dist
                        sprint_dur += cur_dur
                        cur_dist = 0.0
                        cur_dur = 0.0
                    in_sprint = False

            if in_sprint:
                sprint_count += 1
                sprint_dist += cur_dist
                sprint_dur += cur_dur

        coverage = n_valid_steps / max(span_frames, 1)

        records.append({
            "seq_id": seq_id,
            "split": split,
            "track_id": int(track_id),
            "canonical_id": f"{seq_id}__{track_id}",
            "observations": n_obs,
            "first_frame": first_frame,
            "last_frame": last_frame,
            "span_frames": span_frames,
            "track_span_seconds": span_seconds,
            "coverage_ratio": coverage,
            "valid_movement_steps": n_valid_steps,
            "distance_pixels": total_distance,
            "average_speed_pixels_per_second": avg_speed,
            "maximum_step_speed_pixels_per_second": max_speed,
            "median_step_speed_pixels_per_second": median_speed,
            "maximum_positive_acceleration_pixels_per_second_squared": max_pos_accel,
            "maximum_deceleration_magnitude_pixels_per_second_squared": max_decel_mag,
            "experimental_sprint_threshold_pixels_per_second": sprint_thresh,
            "experimental_sprint_count": float(sprint_count),
            "experimental_sprint_distance_pixels": sprint_dist,
            "experimental_sprint_duration_seconds": sprint_dur,
        })

    return records


# ============================================================
# CLIPPING
# ============================================================

CLIP_SPEC = [
    ("distance_pixels",                                            DISTANCE_MIN_PX,   DISTANCE_MAX_PX),
    ("average_speed_pixels_per_second",                            SPEED_MIN_PX_S,    SPEED_MAX_PX_S),
    ("maximum_step_speed_pixels_per_second",                       SPEED_MIN_PX_S,    SPEED_MAX_PX_S),
    ("median_step_speed_pixels_per_second",                        SPEED_MIN_PX_S,    SPEED_MAX_PX_S),
    ("maximum_positive_acceleration_pixels_per_second_squared",    ACCEL_MIN_PX_S2,   ACCEL_MAX_PX_S2),
    ("maximum_deceleration_magnitude_pixels_per_second_squared",   ACCEL_MIN_PX_S2,   ACCEL_MAX_PX_S2),
    ("experimental_sprint_count",                                  0.0,               SPRINT_COUNT_MAX),
    ("experimental_sprint_distance_pixels",                        0.0,               SPRINT_DISTANCE_MAX_PX),
    ("experimental_sprint_duration_seconds",                       0.0,               SPRINT_DURATION_MAX_S),
    ("coverage_ratio",                                             0.0,               1.0),
]


def clip_metrics(df):
    df = df.copy()
    clip_rows = []

    for col, low, high in CLIP_SPEC:
        if col not in df.columns:
            continue

        series = pd.to_numeric(df[col], errors="coerce")
        n_total = int(series.notna().sum())
        n_below = int((series < low).sum())
        n_above = int((series > high).sum())

        df[col] = series.clip(lower=low, upper=high)

        if n_below + n_above > 0:
            clip_rows.append({
                "feature": col,
                "clip_low": low,
                "clip_high": high,
                "rows_clipped_below": n_below,
                "rows_clipped_above": n_above,
                "total_nonmissing": n_total,
            })
            print(
                f"[CLIP] {col}: {n_below} below {low}, "
                f"{n_above} above {high} (of {n_total} non-missing)"
            )

    clip_report = pd.DataFrame(clip_rows)
    return df, clip_report


# ============================================================
# LABEL GENERATION
# ============================================================

def compute_labels(df):
    df = df.copy()

    seq_pct = (
        df.groupby("seq_id")["distance_pixels"]
        .transform(lambda s: s.rank(pct=True, method="average") * 100.0)
    )
    df["seq_activity_score_0_100"] = seq_pct.round(2)

    global_pct = df["distance_pixels"].rank(pct=True, method="average") * 100.0
    df["global_distance_percentile_0_100"] = global_pct.round(2)

    df["reviewed_performance_score_0_100"] = df["seq_activity_score_0_100"]
    df["label_source"] = "sportsmot_football_gt_activity_percentile_within_sequence"
    df["reviewer_notes"] = (
        "Auto-generated from SportsMOT football GT bounding-box trajectories. "
        "Represents relative within-sequence activity rank (0-100)."
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yolo-root",
        type=Path,
        default=DEFAULT_YOLO_ROOT,
        help="Path to processed SportsMOT YOLO dataset (e.g. datasets/football/processed/sportsmot_yolo)",
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=DEFAULT_GT_ROOT,
        help="Path to extracted SportsMOT GT benchmark (e.g. datasets/football/benchmarks/sportsmot/extracted)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save generated ML datasets",
    )
    parser.add_argument(
        "--sport",
        type=str,
        default="football",
        help="Sport type (default: football)",
    )
    args = parser.parse_args()

    yolo_root: Path = args.yolo_root
    gt_root: Path = args.gt_root
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print(f"ATHLETIQ: SportsMOT {args.sport.upper()} ML Data Generation")
    print("=" * 66)
    print(f"YOLO Root: {yolo_root}")
    print(f"GT Root  : {gt_root}")
    print(f"Output   : {output_dir}")

    all_records = []
    skipped_seqs = []
    seq_stats = []

    for split in SPLITS:
        # Determine sequence directories from YOLO root images or GT root
        seq_names = []
        yolo_split_img = yolo_root / "images" / split
        if yolo_split_img.exists():
            seq_names = sorted([d.name for d in yolo_split_img.iterdir() if d.is_dir()])
        elif (gt_root / split).exists():
            seq_names = sorted([d.name for d in (gt_root / split).iterdir() if d.is_dir()])

        if not seq_names:
            print(f"[SKIP] No sequences found for split: {split}")
            continue

        print(f"\n[{split.upper()}] Processing {len(seq_names)} football sequences...")

        for seq_id in seq_names:
            # Locate GT file: first try gt_root, then yolo_root
            gt_file = gt_root / split / seq_id / "gt" / "gt.txt"
            seqinfo_dir = gt_root / split / seq_id
            if not gt_file.exists():
                gt_file = yolo_root / split / seq_id / "gt" / "gt.txt"
                seqinfo_dir = yolo_root / split / seq_id

            if not gt_file.exists():
                skipped_seqs.append(seq_id)
                print(f"  [SKIP] {seq_id}: no ground truth file found ({gt_file})")
                continue

            info = read_seqinfo(seqinfo_dir)
            fps = info["fps"]

            gt = load_gt(gt_file, require_active=REQUIRE_ACTIVE)
            if gt is None:
                skipped_seqs.append(seq_id)
                print(f"  [SKIP] {seq_id}: no usable GT rows")
                continue

            records = compute_player_metrics(gt, fps, seq_id, split)
            n_players = len(records)

            if n_players == 0:
                skipped_seqs.append(seq_id)
                print(f"  [SKIP] {seq_id}: 0 players passed MIN_OBSERVATIONS={MIN_OBSERVATIONS}")
                continue

            all_records.extend(records)
            seq_stats.append({
                "split": split,
                "seq_id": seq_id,
                "fps": fps,
                "seq_length": info["length"],
                "gt_rows": len(gt),
                "players_extracted": n_players,
            })
            print(f"  [OK]   {seq_id}: {n_players} players, {len(gt)} GT rows")

    if not all_records:
        raise RuntimeError(
            f"No player records extracted from any {args.sport} sequence."
        )

    features = pd.DataFrame(all_records)
    print(f"\n[BUILD] Raw feature table: {len(features)} player-sequences")

    print("\n[CLIP] Applying plausibility bounds to physical metrics ...")
    features, clip_report = clip_metrics(features)

    print("\n[LABELS] Computing relative activity labels ...")
    features = compute_labels(features)

    # Summary audit
    numeric_cols = features.select_dtypes(include=np.number).columns.tolist()
    audit_stats = {}
    for col in numeric_cols:
        s = features[col].dropna()
        if s.empty:
            continue
        audit_stats[col] = {
            "min": float(s.min()),
            "p5": float(s.quantile(0.05)),
            "mean": float(s.mean()),
            "median": float(s.median()),
            "p95": float(s.quantile(0.95)),
            "max": float(s.max()),
            "missing": int(features[col].isna().sum()),
        }

    audit = {
        "schema_version": "1.0",
        "sport": args.sport,
        "generated_by": "generate_sportsmot_ml_data.py",
        "yolo_root": str(yolo_root),
        "gt_root": str(gt_root),
        "splits_processed": SPLITS,
        "sequences_processed": len(seq_stats),
        "sequences_skipped": skipped_seqs,
        "total_player_rows": len(features),
        "players_per_split": {
            split: int((features["split"] == split).sum())
            for split in SPLITS
        },
        "label_column": "reviewed_performance_score_0_100",
        "label_source": "seq_activity_percentile_of_distance_pixels",
        "label_range_min": float(features["reviewed_performance_score_0_100"].min()),
        "label_range_max": float(features["reviewed_performance_score_0_100"].max()),
        "clipping": {
            "params": {col: {"low": low, "high": high} for col, low, high in CLIP_SPEC},
            "features_clipped": len(clip_report),
        },
        "feature_stats": audit_stats,
        "warnings": [
            f"Trained on {args.sport} SportsMOT sequences.",
            "Labels are percentile ranks of distance within each sequence, not expert judgments.",
            "Pixel-space metrics calibrated to broadcast video bounds.",
            "Player IDs are seq_id__track_id composites, not real identities.",
        ],
    }

    label_cols = [
        "canonical_id", "seq_id", "split",
        "reviewed_performance_score_0_100",
        "seq_activity_score_0_100",
        "global_distance_percentile_0_100",
        "label_source", "reviewer_notes",
    ]
    label_df = features[[c for c in label_cols if c in features.columns]].copy()

    feature_only_cols = [
        c for c in features.columns
        if c not in (
            "reviewed_performance_score_0_100",
            "seq_activity_score_0_100",
            "global_distance_percentile_0_100",
            "label_source",
            "reviewer_notes",
        )
    ]
    feature_df = features[feature_only_cols].copy()

    features_path = output_dir / "sportsmot_ml_features.csv"
    labels_path   = output_dir / "sportsmot_ml_labels.csv"
    merged_path   = output_dir / "sportsmot_ml_merged.csv"
    audit_path    = output_dir / "sportsmot_ml_audit.json"
    clip_path     = output_dir / "sportsmot_ml_clip_report.csv"
    seq_stats_path = output_dir / "sportsmot_sequence_stats.csv"

    feature_df.to_csv(features_path, index=False)
    label_df.to_csv(labels_path, index=False)
    features.to_csv(merged_path, index=False)
    clip_report.to_csv(clip_path, index=False)
    pd.DataFrame(seq_stats).to_csv(seq_stats_path, index=False)
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("\n" + "=" * 66)
    print(f"SportsMOT {args.sport.capitalize()} ML Data Generation -- COMPLETE")
    print("=" * 66)
    print(f"Sequences processed : {len(seq_stats)}")
    print(f"Sequences skipped   : {len(skipped_seqs)}")
    print(f"Player rows total   : {len(features)}")
    print(f"  train             : {int((features['split'] == 'train').sum())}")
    print(f"  val               : {int((features['split'] == 'val').sum())}")
    print(f"  test              : {int((features['split'] == 'test').sum())}")
    print(f"\nLabel range         : {audit['label_range_min']:.1f} -- {audit['label_range_max']:.1f}")
    print(f"Features clipped    : {len(clip_report)} metrics had out-of-range values")
    print(f"\nOutputs written to  : {output_dir}")


if __name__ == "__main__":
    main()
