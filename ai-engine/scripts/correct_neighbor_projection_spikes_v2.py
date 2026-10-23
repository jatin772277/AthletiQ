#!/usr/bin/env python3
"""V2: conservative neighbor-guided trajectory spike correction."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return -1.0
    return float(np.dot(a, b) / (na * nb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="outputs/physical_metrics/"
                "football2_stitched_physical_tracks.csv",
    )
    ap.add_argument(
        "--output-dir",
        default="outputs/physical_metrics/neighbor_correction_v2",
    )
    ap.add_argument("--neighbor-radius-px", type=float, default=120.0)
    ap.add_argument("--min-neighbors", type=int, default=2)
    ap.add_argument("--min-agreement", type=float, default=0.65)
    ap.add_argument("--min-residual-m", type=float, default=0.75)
    ap.add_argument("--max-correction-m", type=float, default=5.0)
    ap.add_argument("--max-interpolation-gap", type=float, default=0.30)
    ap.add_argument("--speed-limit", type=float, default=12.0)
    ap.add_argument("--max-neighbor-residual-m", type=float, default=8.0)
    args = ap.parse_args()

    src = Path(args.input)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src).sort_values(
        ["canonical_id", "time_seconds", "frame_id"]
    ).reset_index(drop=True)

    required = [
        "canonical_id", "frame_id", "time_seconds",
        "foot_x_px", "foot_y_px", "pitch_x_m", "pitch_y_m",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    n = len(df)
    ids = df["canonical_id"].to_numpy()
    frames = df["frame_id"].to_numpy()
    times = df["time_seconds"].to_numpy(float)
    ix = df["foot_x_px"].to_numpy(float)
    iy = df["foot_y_px"].to_numpy(float)
    xy = df[["pitch_x_m", "pitch_y_m"]].to_numpy(float)
    valid = np.isfinite(xy).all(axis=1) & np.isfinite(times)

    # Prediction from the observations immediately before and after a point.
    # We only predict when both temporal gaps are short.
    pred = np.full((n, 2), np.nan)
    residual = np.full((n, 2), np.nan)
    residual_size = np.full(n, np.nan)
    prev_idx = np.full(n, -1, dtype=int)
    next_idx = np.full(n, -1, dtype=int)

    for _, raw_indices in df.groupby("canonical_id", sort=False).indices.items():
        inds = np.asarray(raw_indices, dtype=int)
        inds = inds[np.argsort(times[inds])]
        for k in range(1, len(inds) - 1):
            a, i, b = inds[k - 1], inds[k], inds[k + 1]
            if not (valid[a] and valid[i] and valid[b]):
                continue
            da, db = times[i] - times[a], times[b] - times[i]
            total = times[b] - times[a]
            if min(da, db, total) <= 0:
                continue
            if max(da, db) > args.max_interpolation_gap:
                continue

            pred[i] = xy[a] + (da / total) * (xy[b] - xy[a])
            residual[i] = xy[i] - pred[i]
            residual_size[i] = np.linalg.norm(residual[i])
            prev_idx[i], next_idx[i] = a, b

    # Calculate original speed per consecutive observation.
    original_speed = np.full(n, np.nan)
    for _, raw_indices in df.groupby("canonical_id", sort=False).indices.items():
        inds = np.asarray(raw_indices, dtype=int)
        inds = inds[np.argsort(times[inds])]
        for k in range(1, len(inds)):
            a, b = inds[k - 1], inds[k]
            dt = times[b] - times[a]
            if dt > 0 and dt <= 0.5 and valid[a] and valid[b]:
                original_speed[b] = np.linalg.norm(xy[b] - xy[a]) / dt

    # Candidate if either the source speed flag is set, speed exceeds limit,
    # or the local trajectory residual is large.
    source_flag = (
        df["speed_outlier_flag"].fillna(False).to_numpy(bool)
        if "speed_outlier_flag" in df.columns
        else np.zeros(n, dtype=bool)
    )
    candidate = (
        source_flag
        | (np.isfinite(original_speed) & (original_speed > args.speed_limit))
        | (np.isfinite(residual_size) & (residual_size >= args.min_residual_m))
    )

    frame_map = {}
    for i, frame in enumerate(frames):
        frame_map.setdefault(frame, []).append(i)

    # Find a shared residual for each candidate. Neighbors need not be
    # flagged as outliers themselves; they only need valid local predictions.
    proposals = {}
    for i in np.flatnonzero(candidate):
        if not np.isfinite(residual[i]).all():
            continue

        support = []
        for j in frame_map.get(frames[i], []):
            if j == i or ids[j] == ids[i]:
                continue
            if not np.isfinite(residual[j]).all():
                continue
            if residual_size[j] < args.min_residual_m:
                continue
            if residual_size[j] > args.max_neighbor_residual_m:
                continue
            if not (np.isfinite(ix[j]) and np.isfinite(iy[j])
                    and np.isfinite(ix[i]) and np.isfinite(iy[i])):
                continue

            image_dist = np.hypot(ix[j] - ix[i], iy[j] - iy[i])
            if image_dist > args.neighbor_radius_px:
                continue

            agree = cosine(residual[j], residual[i])
            if agree >= args.min_agreement:
                support.append((j, residual[j], agree, image_dist))

        if len(support) >= args.min_neighbors:
            shift = np.median(
                np.asarray([s[1] for s in support]), axis=0
            )
            # Require that the common residual actually explains a
            # meaningful portion of the candidate's own residual.
            aligned = float(np.dot(shift, residual[i]) /
                            (np.dot(residual[i], residual[i]) + 1e-9))
            aligned = float(np.clip(aligned, 0.0, 1.0))
            correction = shift * aligned
            magnitude = float(np.linalg.norm(correction))

            if 0.25 <= magnitude <= args.max_correction_m:
                proposals[i] = {
                    "xy": xy[i] - correction,
                    "method": "neighbor_shared_residual",
                    "support": len(support),
                    "confidence": min(
                        0.95,
                        float(np.mean([s[2] for s in support]))
                        * min(1.0, len(support) / 3.0),
                    ),
                }

    # Temporal fallback: try the between-neighbor interpolation only when
    # it reduces the worst adjacent speed and does not create new speed spikes.
    def adjacent_speeds(i, point):
        a, b = prev_idx[i], next_idx[i]
        if a < 0 or b < 0:
            return None
        left_dt, right_dt = times[i] - times[a], times[b] - times[i]
        if left_dt <= 0 or right_dt <= 0:
            return None
        return (
            np.linalg.norm(point - xy[a]) / left_dt,
            np.linalg.norm(xy[b] - point) / right_dt,
        )

    for i in np.flatnonzero(candidate):
        if i in proposals or not np.isfinite(pred[i]).all():
            continue
        before = adjacent_speeds(i, xy[i])
        after = adjacent_speeds(i, pred[i])
        if before is None or after is None:
            continue

        before_max, after_max = max(before), max(after)
        before_sum, after_sum = sum(before), sum(after)
        if (
            after_max < before_max
            and after_sum < before_sum
            and after_max <= args.speed_limit
            and np.linalg.norm(xy[i] - pred[i]) <= args.max_correction_m
        ):
            proposals[i] = {
                "xy": pred[i],
                "method": "validated_temporal_interpolation",
                "support": 0,
                "confidence": 0.45,
            }

    # Preserve originals; accepted corrections live in separate columns.
    df["pitch_x_m_corrected"] = xy[:, 0]
    df["pitch_y_m_corrected"] = xy[:, 1]
    df["correction_candidate"] = candidate
    df["trajectory_deviation_m"] = residual_size
    df["correction_applied"] = False
    df["correction_method"] = "unchanged"
    df["correction_confidence"] = 0.0
    df["neighbor_support_count"] = 0
    df["correction_delta_m"] = 0.0

    for i, p in proposals.items():
        df.at[i, "pitch_x_m_corrected"] = p["xy"][0]
        df.at[i, "pitch_y_m_corrected"] = p["xy"][1]
        df.at[i, "correction_applied"] = True
        df.at[i, "correction_method"] = p["method"]
        df.at[i, "correction_confidence"] = p["confidence"]
        df.at[i, "neighbor_support_count"] = p["support"]
        df.at[i, "correction_delta_m"] = float(
            np.linalg.norm(p["xy"] - xy[i])
        )

    # Compare original/corrected speeds and flag suspicious corrected steps.
    def calc_speeds(xcol, ycol):
        result = np.full(n, np.nan)
        coords = df[[xcol, ycol]].to_numpy(float)
        for _, raw_indices in df.groupby("canonical_id", sort=False).indices.items():
            inds = np.asarray(raw_indices, dtype=int)
            inds = inds[np.argsort(times[inds])]
            for k in range(1, len(inds)):
                a, b = inds[k - 1], inds[k]
                dt = times[b] - times[a]
                if 0 < dt <= 0.5 and np.isfinite(coords[[a, b]]).all():
                    result[b] = np.linalg.norm(coords[b] - coords[a]) / dt
        return result

    corrected_speed = calc_speeds(
        "pitch_x_m_corrected", "pitch_y_m_corrected"
    )
    df["original_speed_recomputed_mps"] = original_speed
    df["corrected_speed_mps"] = corrected_speed
    df["corrected_speed_outlier_flag"] = (
        np.isfinite(corrected_speed) & (corrected_speed > args.speed_limit)
    )

    out_csv = outdir / "football2_neighbor_corrected_tracks_v2.csv"
    df.to_csv(out_csv, index=False)

    method_counts = (
        df.loc[df["correction_applied"], "correction_method"]
        .value_counts().to_dict()
    )
    audit = {
        "input": str(src),
        "output": str(out_csv),
        "rows": int(n),
        "candidate_points": int(candidate.sum()),
        "corrections_applied": int(len(proposals)),
        "correction_methods": method_counts,
        "original_speed_outliers_recomputed": int(
            np.nansum(original_speed > args.speed_limit)
        ),
        "corrected_speed_outliers": int(
            np.nansum(corrected_speed > args.speed_limit)
        ),
        "mean_correction_m": float(
            df.loc[df["correction_applied"], "correction_delta_m"].mean()
        ) if proposals else 0.0,
        "max_correction_m": float(
            df.loc[df["correction_applied"], "correction_delta_m"].max()
        ) if proposals else 0.0,
        "neighbor_radius_px": args.neighbor_radius_px,
        "minimum_neighbor_support": args.min_neighbors,
        "warnings": [
            "Heuristic correction; no ground-truth positions are available.",
            "A shared residual is evidence, not proof, of projection error.",
            "Review corrected observations against the source video before production use.",
        ],
    }
    out_json = outdir / "football2_neighbor_correction_audit_v2.json"
    out_json.write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))
    print(f"\nCSV: {out_csv}\nAudit: {out_json}")


if __name__ == "__main__":
    main()
