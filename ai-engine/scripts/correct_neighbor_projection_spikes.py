#!/usr/bin/env python3
"""Conservative neighbor-guided correction of football pitch trajectories.

Original pitch_x_m / pitch_y_m are never overwritten.
Outputs corrected coordinates, correction provenance, and an audit JSON.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def robust_vector_median(vectors):
    """Component-wise median of a collection of 2D vectors."""
    return np.median(np.asarray(vectors, dtype=float), axis=0)


def unit_cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return -1.0
    return float(np.dot(a, b) / (na * nb))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="outputs/physical_metrics/"
                "football2_stitched_physical_tracks.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/physical_metrics/neighbor_correction",
    )
    parser.add_argument("--neighbor-radius-px", type=float, default=120.0)
    parser.add_argument("--max-interpolation-gap", type=float, default=0.30)
    parser.add_argument("--spike-deviation-m", type=float, default=1.0)
    parser.add_argument("--spike-speed-mps", type=float, default=12.0)
    parser.add_argument("--min-neighbors", type=int, default=2)
    parser.add_argument("--min-neighbor-agreement", type=float, default=0.65)
    parser.add_argument("--max-neighbor-residual-m", type=float, default=8.0)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    required = {
        "frame_id", "time_seconds", "canonical_id",
        "foot_x_px", "foot_y_px", "pitch_x_m", "pitch_y_m",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")

    df = df.sort_values(
        ["canonical_id", "time_seconds", "frame_id"]
    ).reset_index(drop=True)

    # Work in new columns; never overwrite original pitch coordinates.
    df["pitch_x_m_corrected"] = df["pitch_x_m"]
    df["pitch_y_m_corrected"] = df["pitch_y_m"]
    df["correction_method"] = "none"
    df["correction_confidence"] = 0.0
    df["correction_applied"] = False
    df["trajectory_deviation_m"] = np.nan
    df["neighbor_support_count"] = 0

    # Keep only finite projected points for prediction/correction.
    x = df["pitch_x_m"].to_numpy(dtype=float)
    y = df["pitch_y_m"].to_numpy(dtype=float)
    t = df["time_seconds"].to_numpy(dtype=float)
    fx = df["foot_x_px"].to_numpy(dtype=float)
    fy = df["foot_y_px"].to_numpy(dtype=float)
    frames = df["frame_id"].to_numpy()
    ids = df["canonical_id"].to_numpy()

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)
    xy = np.column_stack([x, y])

    # For each observation, predict its position by interpolation between
    # its previous and next observation. Large residuals are spike candidates.
    predictions = np.full_like(xy, np.nan, dtype=float)
    deviations = np.full(len(df), np.nan)
    groups = df.groupby("canonical_id", sort=False).indices

    for _, indices in groups.items():
        idx = np.asarray(indices, dtype=int)
        idx = idx[np.argsort(t[idx])]
        for k in range(1, len(idx) - 1):
            prev_i, i, next_i = idx[k - 1], idx[k], idx[k + 1]
            if not (valid[prev_i] and valid[i] and valid[next_i]):
                continue

            left_dt = t[i] - t[prev_i]
            right_dt = t[next_i] - t[i]
            total_dt = t[next_i] - t[prev_i]

            if (
                left_dt <= 0 or right_dt <= 0 or total_dt <= 0
                or left_dt > args.max_interpolation_gap
                or right_dt > args.max_interpolation_gap
            ):
                continue

            alpha = left_dt / total_dt
            pred = xy[prev_i] + alpha * (xy[next_i] - xy[prev_i])
            predictions[i] = pred
            deviations[i] = float(np.linalg.norm(xy[i] - pred))

    df["trajectory_deviation_m"] = deviations

    # Candidate must be a large deviation from its own local trajectory and
    # also have an implausible step speed. This is intentionally conservative.
    speed = (
        df["step_speed_mps"].to_numpy(dtype=float)
        if "step_speed_mps" in df.columns
        else np.full(len(df), np.nan)
    )
    candidate = (
        np.isfinite(deviations)
        & (deviations >= args.spike_deviation_m)
        & (np.isfinite(speed) & (speed > args.spike_speed_mps))
    )

    # Build exact-frame index for finding spatially nearby players.
    frame_to_indices = {}
    for i, frame in enumerate(frames):
        frame_to_indices.setdefault(frame, []).append(i)

    # First pass: calculate candidate corrections without changing coordinates.
    proposals = {}

    for i in np.flatnonzero(candidate):
        frame_candidates = frame_to_indices.get(frames[i], [])
        neighbors = []

        for j in frame_candidates:
            if j == i or ids[j] == ids[i]:
                continue
            if not np.isfinite(fx[j]) or not np.isfinite(fy[j]):
                continue
            if not np.all(np.isfinite(predictions[j])):
                continue

            image_distance = float(
                np.hypot(fx[j] - fx[i], fy[j] - fy[i])
            )
            if image_distance > args.neighbor_radius_px:
                continue

            residual = xy[j] - predictions[j]
            residual_magnitude = float(np.linalg.norm(residual))

            # Ignore enormous neighbor residuals: they are not reliable
            # evidence for a shared camera/projection shift.
            if (
                residual_magnitude > args.max_neighbor_residual_m
                or residual_magnitude < 0.25
            ):
                continue

            # Prefer high-confidence detections when this metadata exists.
            detection_conf = (
                float(df.iloc[j]["confidence"])
                if "confidence" in df.columns
                and pd.notna(df.iloc[j]["confidence"])
                else 1.0
            )
            if detection_conf < 0.15:
                continue

            neighbors.append((j, residual, image_distance))

        own_residual = xy[i] - predictions[i]
        agreeing = []

        for j, residual, image_distance in neighbors:
            if unit_cosine(residual, own_residual) >= (
                args.min_neighbor_agreement
            ):
                agreeing.append((j, residual, image_distance))

        # Shared-shift correction only if at least the requested number of
        # independent nearby players show a similarly directed residual.
        if len(agreeing) >= args.min_neighbors:
            shared_shift = robust_vector_median(
                [item[1] for item in agreeing]
            )

            # Correct only the component supported by the shared shift.
            corrected = xy[i] - shared_shift
            correction_size = float(np.linalg.norm(xy[i] - corrected))

            # A guard against excessively large automatic changes.
            if (
                np.all(np.isfinite(corrected))
                and correction_size >= 0.25
                and correction_size <= args.max_neighbor_residual_m
            ):
                agreement = float(np.mean([
                    max(0.0, unit_cosine(item[1], own_residual))
                    for item in agreeing
                ]))
                proposals[i] = {
                    "xy": corrected,
                    "method": "neighbor_shared_shift",
                    "confidence": min(
                        0.95,
                        0.50 + 0.45 * agreement
                        * min(1.0, len(agreeing) / 4.0),
                    ),
                    "support": len(agreeing),
                }
                continue

        # Fallback: if no shared shift is supported, interpolate this
        # isolated spike from its own surrounding observations.
        pred = predictions[i]
        correction_size = float(np.linalg.norm(xy[i] - pred))
        if (
            np.all(np.isfinite(pred))
            and correction_size >= args.spike_deviation_m
        ):
            proposals[i] = {
                "xy": pred,
                "method": "temporal_interpolation_fallback",
                "confidence": 0.45,
                "support": len(agreeing),
            }

    # Apply proposals only to corrected columns.
    for i, proposal in proposals.items():
        df.at[i, "pitch_x_m_corrected"] = proposal["xy"][0]
        df.at[i, "pitch_y_m_corrected"] = proposal["xy"][1]
        df.at[i, "correction_method"] = proposal["method"]
        df.at[i, "correction_confidence"] = proposal["confidence"]
        df.at[i, "correction_applied"] = True
        df.at[i, "neighbor_support_count"] = proposal["support"]

    # Recalculate step metrics from corrected coordinates.
    df = df.sort_values(
        ["canonical_id", "time_seconds", "frame_id"]
    ).reset_index(drop=True)
    cg = df.groupby("canonical_id", sort=False)
    dt = cg["time_seconds"].diff()
    dx = cg["pitch_x_m_corrected"].diff()
    dy = cg["pitch_y_m_corrected"].diff()
    distance = np.hypot(dx, dy)

    df["corrected_step_dt_seconds"] = dt
    df["corrected_step_distance_m"] = distance
    df["corrected_step_speed_mps"] = distance / dt.where(dt > 0)

    # Mark rather than silently count implausible or long-gap steps.
    df["corrected_speed_outlier_flag"] = (
        (df["corrected_step_dt_seconds"] <= 0)
        | (df["corrected_step_dt_seconds"] > 0.5)
        | (df["corrected_step_speed_mps"] > args.spike_speed_mps)
    )
    df.loc[
        df["corrected_step_dt_seconds"].isna(),
        "corrected_speed_outlier_flag"
    ] = False

    points_path = output_dir / "football2_neighbor_corrected_tracks.csv"
    df.to_csv(points_path, index=False)

    corrected = df[df["correction_applied"]]
    methods = corrected["correction_method"].value_counts().to_dict()
    audit = {
        "input": str(input_path),
        "output": str(points_path),
        "input_rows": int(len(df)),
        "original_coordinate_columns_preserved": True,
        "candidate_spikes": int(candidate.sum()),
        "corrections_applied": int(len(corrected)),
        "correction_methods": methods,
        "neighbor_radius_px": args.neighbor_radius_px,
        "minimum_neighbor_support": args.min_neighbors,
        "spike_deviation_threshold_m": args.spike_deviation_m,
        "speed_threshold_mps": args.spike_speed_mps,
        "corrected_speed_outliers": int(
            df["corrected_speed_outlier_flag"].sum()
        ),
        "mean_correction_m": (
            float(np.mean(np.linalg.norm(
                corrected[["pitch_x_m_corrected", "pitch_y_m_corrected"]]
                .to_numpy()
                - corrected[["pitch_x_m", "pitch_y_m"]].to_numpy(),
                axis=1,
            ))) if len(corrected) else 0.0
        ),
        "warnings": [
            "Corrections are heuristic and are not ground-truth validated.",
            "Neighbor support is based on image-space proximity and temporal residuals.",
            "Temporal interpolation can suppress genuine rapid changes.",
            "Inspect correction provenance and overlay corrected tracks on video.",
        ],
    }

    audit_path = output_dir / "football2_neighbor_correction_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2))

    print(json.dumps(audit, indent=2))
    print("\nOutputs:")
    print(points_path)
    print(audit_path)


if __name__ == "__main__":
    main()
