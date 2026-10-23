
#!/usr/bin/env python3
"""Compare original and provisional corrected player trajectories."""

from pathlib import Path
import json

import numpy as np
import pandas as pd

ORIGINAL = Path(
    "outputs/physical_metrics/football2_stitched_physical_tracks.csv"
)
CORRECTED = Path(
    "outputs/physical_metrics/neighbor_correction_v2/"
    "football2_neighbor_corrected_tracks_v2.csv"
)
OUTPUT_DIR = Path(
    "outputs/physical_metrics/trajectory_correction_comparison"
)

SPEED_LIMIT_MPS = 12.0
MAX_GAP_SECONDS = 0.5


def load_tracks(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")

    df = pd.read_csv(path)

    required = {
        "canonical_id",
        "time_seconds",
        "pitch_x_m",
        "pitch_y_m",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df["time_seconds"] = pd.to_numeric(
        df["time_seconds"], errors="coerce"
    )
    df["pitch_x_m"] = pd.to_numeric(df["pitch_x_m"], errors="coerce")
    df["pitch_y_m"] = pd.to_numeric(df["pitch_y_m"], errors="coerce")

    df = df.sort_values(
        ["canonical_id", "time_seconds"]
    ).reset_index(drop=True)

    # Recompute speed from coordinates rather than trusting stored speeds.
    grouped = df.groupby("canonical_id", sort=False)

    df["previous_time_seconds"] = grouped["time_seconds"].shift(1)
    df["previous_pitch_x_m"] = grouped["pitch_x_m"].shift(1)
    df["previous_pitch_y_m"] = grouped["pitch_y_m"].shift(1)

    df["step_dt_recomputed_s"] = (
        df["time_seconds"] - df["previous_time_seconds"]
    )

    dx = df["pitch_x_m"] - df["previous_pitch_x_m"]
    dy = df["pitch_y_m"] - df["previous_pitch_y_m"]

    df["step_distance_recomputed_m"] = np.hypot(dx, dy)

    valid = (
        df["pitch_x_m"].notna()
        & df["pitch_y_m"].notna()
        & df["previous_pitch_x_m"].notna()
        & df["previous_pitch_y_m"].notna()
        & df["step_dt_recomputed_s"].gt(0)
        & df["step_dt_recomputed_s"].le(MAX_GAP_SECONDS)
    )

    df["speed_recomputed_mps"] = np.nan
    df.loc[valid, "speed_recomputed_mps"] = (
        df.loc[valid, "step_distance_recomputed_m"]
        / df.loc[valid, "step_dt_recomputed_s"]
    )

    df["speed_outlier_recomputed"] = (
        df["speed_recomputed_mps"] > SPEED_LIMIT_MPS
    )

    df["source"] = label
    return df


def summarize(df: pd.DataFrame) -> dict:
    speed = df["speed_recomputed_mps"].dropna()
    outliers = speed[speed > SPEED_LIMIT_MPS]

    # Exclude outlier steps from this diagnostic distance total.
    accepted = df.loc[
        df["speed_recomputed_mps"].notna()
        & (df["speed_recomputed_mps"] <= SPEED_LIMIT_MPS),
        "step_distance_recomputed_m",
    ]

    return {
        "source": str(df["source"].iloc[0]),
        "rows": int(len(df)),
        "players": int(df["canonical_id"].nunique()),
        "valid_steps": int(speed.size),
        "steps_over_12_mps": int(len(outliers)),
        "percentage_steps_over_12": (
            round(100 * len(outliers) / len(speed), 3)
            if len(speed)
            else None
        ),
        "maximum_speed_mps": (
            float(speed.max()) if len(speed) else None
        ),
        "p95_speed_mps": (
            float(speed.quantile(0.95)) if len(speed) else None
        ),
        "p99_speed_mps": (
            float(speed.quantile(0.99)) if len(speed) else None
        ),
        "accepted_step_distance_sum_m": (
            float(accepted.sum()) if len(accepted) else None
        ),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    original = load_tracks(ORIGINAL, "original")
    corrected = load_tracks(CORRECTED, "corrected")

    summaries = pd.DataFrame([
        summarize(original),
        summarize(corrected),
    ])

    print("\n=== TRAJECTORY COMPARISON ===")
    print(summaries.round(3).to_string(index=False))

    # Compare coordinates for matching player/frame observations.
    keys = ["canonical_id", "time_seconds"]

    original_coords = original[
        keys + ["pitch_x_m", "pitch_y_m"]
    ].rename(columns={
        "pitch_x_m": "original_x_m",
        "pitch_y_m": "original_y_m",
    })

    corrected_coords = corrected[
        keys + ["pitch_x_m", "pitch_y_m"]
    ].rename(columns={
        "pitch_x_m": "corrected_x_m",
        "pitch_y_m": "corrected_y_m",
    })

    comparison = original_coords.merge(
        corrected_coords,
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    comparison["coordinate_change_m"] = np.hypot(
        comparison["corrected_x_m"] - comparison["original_x_m"],
        comparison["corrected_y_m"] - comparison["original_y_m"],
    )

    changed = comparison["coordinate_change_m"] > 1e-6

    coordinate_report = {
        "matching_observations": int(len(comparison)),
        "observations_with_changed_coordinates": int(changed.sum()),
        "mean_coordinate_change_m": (
            float(comparison.loc[changed, "coordinate_change_m"].mean())
            if changed.any() else 0.0
        ),
        "median_coordinate_change_m": (
            float(comparison.loc[changed, "coordinate_change_m"].median())
            if changed.any() else 0.0
        ),
        "maximum_coordinate_change_m": (
            float(comparison["coordinate_change_m"].max())
            if len(comparison) else None
        ),
    }

    # Save all recomputed speed outliers for manual review.
    outlier_rows = []
    for tracks in (original, corrected):
        cols = [
            "source",
            "canonical_id",
            "time_seconds",
            "pitch_x_m",
            "pitch_y_m",
            "step_dt_recomputed_s",
            "step_distance_recomputed_m",
            "speed_recomputed_mps",
        ]
        outlier_rows.append(
            tracks.loc[
                tracks["speed_outlier_recomputed"], cols
            ]
        )

    outliers = pd.concat(outlier_rows, ignore_index=True)
    outlier_path = OUTPUT_DIR / "recomputed_speed_outliers.csv"
    outliers.to_csv(outlier_path, index=False)

    summary_path = OUTPUT_DIR / "trajectory_comparison_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "speed_threshold_mps": SPEED_LIMIT_MPS,
                "maximum_step_gap_seconds": MAX_GAP_SECONDS,
                "summaries": summaries.to_dict(orient="records"),
                "coordinate_comparison": coordinate_report,
                "warning": (
                    "This is a diagnostic comparison, not proof that "
                    "the corrected coordinates are physically accurate."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== COORDINATE CHANGE AUDIT ===")
    print(json.dumps(coordinate_report, indent=2))
    print(f"\nOutlier details: {outlier_path}")
    print(f"Summary report:  {summary_path}")


if __name__ == "__main__":
    main()
