
#!/usr/bin/env python3
"""Independently validate the V2 trajectory correction candidate."""

from pathlib import Path
import json

import numpy as np
import pandas as pd

INPUT = Path(
    "outputs/physical_metrics/neighbor_correction_v2/"
    "football2_neighbor_corrected_tracks_v2.csv"
)
OUTPUT_DIR = Path(
    "outputs/physical_metrics/neighbor_correction_v2/validation"
)
LIMIT = 12.0
MAX_GAP = 0.5


def calculate_speeds(df, xcol, ycol):
    result = pd.Series(np.nan, index=df.index, dtype=float)

    for _, group in df.groupby("canonical_id", sort=False):
        group = group.sort_values(["time_seconds", "frame_id"])
        idx = group.index.to_numpy()

        xy = group[[xcol, ycol]].to_numpy(dtype=float)
        times = group["time_seconds"].to_numpy(dtype=float)

        if len(group) < 2:
            continue

        dt = np.diff(times)
        distance = np.linalg.norm(np.diff(xy, axis=0), axis=1)

        valid = (
            np.isfinite(xy[:-1]).all(axis=1)
            & np.isfinite(xy[1:]).all(axis=1)
            & np.isfinite(dt)
            & (dt > 0)
            & (dt <= MAX_GAP)
        )

        speeds = np.full(len(dt), np.nan)
        speeds[valid] = distance[valid] / dt[valid]
        result.loc[idx[1:]] = speeds

    return result


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing candidate file: {INPUT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT)

    required = {
        "canonical_id", "frame_id", "time_seconds",
        "pitch_x_m", "pitch_y_m",
        "pitch_x_m_corrected", "pitch_y_m_corrected",
        "correction_applied", "correction_method",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.sort_values(
        ["canonical_id", "time_seconds", "frame_id"]
    ).reset_index(drop=True)

    df["original_speed_recomputed_mps"] = calculate_speeds(
        df, "pitch_x_m", "pitch_y_m"
    )
    df["corrected_speed_recomputed_mps"] = calculate_speeds(
        df, "pitch_x_m_corrected", "pitch_y_m_corrected"
    )

    original = df["original_speed_recomputed_mps"]
    corrected = df["corrected_speed_recomputed_mps"]

    df["original_outlier"] = original > LIMIT
    df["corrected_outlier"] = corrected > LIMIT
    df["new_outlier_after_correction"] = (
        ~df["original_outlier"].fillna(False)
        & df["corrected_outlier"].fillna(False)
    )
    df["resolved_outlier"] = (
        df["original_outlier"].fillna(False)
        & ~df["corrected_outlier"].fillna(False)
    )

    # Summarize each correction method.
    method_rows = []
    for method, group in df[df["correction_applied"]].groupby(
        "correction_method", dropna=False
    ):
        idx = group.index
        method_rows.append({
            "method": method,
            "corrected_points": len(group),
            "points_still_over_12_mps": int(
                df.loc[idx, "corrected_outlier"].sum()
            ),
            "new_outliers_at_corrected_points": int(
                df.loc[idx, "new_outlier_after_correction"].sum()
            ),
            "resolved_outliers_at_corrected_points": int(
                df.loc[idx, "resolved_outlier"].sum()
            ),
            "median_correction_m": float(
                group["correction_delta_m"].median()
            ),
            "max_correction_m": float(
                group["correction_delta_m"].max()
            ),
        })

    method_summary = pd.DataFrame(method_rows)
    method_summary.to_csv(
        OUTPUT_DIR / "correction_method_validation.csv", index=False
    )

    # Save spikes with adjacent observations for review.
    spike_indices = set(
        df.index[
            df["original_outlier"].fillna(False)
            | df["corrected_outlier"].fillna(False)
        ]
    )

    review_indices = set(spike_indices)
    for _, group in df.groupby("canonical_id", sort=False):
        ordered = group.sort_values(["time_seconds", "frame_id"])
        indices = ordered.index.to_list()
        positions = {idx: i for i, idx in enumerate(indices)}

        for idx in list(spike_indices.intersection(indices)):
            pos = positions[idx]
            for neighbor_pos in (pos - 1, pos + 1):
                if 0 <= neighbor_pos < len(indices):
                    review_indices.add(indices[neighbor_pos])

    review_cols = [
        "canonical_id", "frame_id", "time_seconds",
        "pitch_x_m", "pitch_y_m",
        "pitch_x_m_corrected", "pitch_y_m_corrected",
        "original_speed_recomputed_mps",
        "corrected_speed_recomputed_mps",
        "original_outlier", "corrected_outlier",
        "new_outlier_after_correction", "resolved_outlier",
        "correction_applied", "correction_method",
        "correction_delta_m", "correction_confidence",
    ]
    review = df.loc[sorted(review_indices), review_cols].copy()
    review.to_csv(
        OUTPUT_DIR / "spikes_with_neighbors_review.csv", index=False
    )

    report = {
        "rows": int(len(df)),
        "players": int(df["canonical_id"].nunique()),
        "original_outliers_recomputed": int(
            df["original_outlier"].sum()
        ),
        "corrected_outliers_recomputed": int(
            df["corrected_outlier"].sum()
        ),
        "resolved_outliers": int(df["resolved_outlier"].sum()),
        "new_outliers_after_correction": int(
            df["new_outlier_after_correction"].sum()
        ),
        "corrections_applied": int(df["correction_applied"].sum()),
        "max_corrected_speed_mps": (
            float(corrected.max()) if corrected.notna().any() else None
        ),
        "p99_original_speed_mps": float(original.quantile(0.99)),
        "p99_corrected_speed_mps": float(corrected.quantile(0.99)),
        "method_summary": method_rows,
        "warning": (
            "A reduced outlier count does not by itself prove that "
            "corrected coordinates are physically accurate."
        ),
    }

    (OUTPUT_DIR / "validation_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print("=== INDEPENDENT CORRECTION VALIDATION ===")
    print(json.dumps(report, indent=2))
    print("\nMethod summary:")
    if not method_summary.empty:
        print(method_summary.round(3).to_string(index=False))
    print("\nSaved:")
    print(OUTPUT_DIR / "validation_summary.json")
    print(OUTPUT_DIR / "correction_method_validation.csv")
    print(OUTPUT_DIR / "spikes_with_neighbors_review.csv")


if __name__ == "__main__":
    main()
