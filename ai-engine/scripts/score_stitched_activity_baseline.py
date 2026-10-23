
#!/usr/bin/env python3
"""Create a provisional activity baseline for stitched AthletiQ tracks."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "outputs/dsa/football2_stitched_movement_features.csv"
OUTPUT_DIR = ROOT / "outputs/dsa/stitched_baseline"
OUTPUT = OUTPUT_DIR / "football2_stitched_activity_baseline.csv"
RANKING = OUTPUT_DIR / "football2_stitched_activity_ranking.csv"
SUMMARY = OUTPUT_DIR / "football2_stitched_activity_summary.json"

MIN_OBSERVED_FRAMES = 30
MIN_VALID_STEPS = 1


def percentile(series):
    """Rank non-missing values among eligible tracks only."""
    values = pd.to_numeric(series, errors="coerce")
    return values.rank(pct=True, method="average") * 100


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT}")

    df = pd.read_csv(INPUT)

    required = [
        "video_id",
        "canonical_id",
        "observed_frames",
        "movement_valid_movement_steps",
        "movement_coverage_ratio",
        "movement_distance_pixels",
        "movement_average_speed_pixels_per_second",
        "movement_maximum_step_speed_pixels_per_second",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.empty:
        raise ValueError("Input feature table is empty.")

    if df.duplicated(["video_id", "canonical_id"]).any():
        raise ValueError("Duplicate (video_id, canonical_id) keys found.")

    for col in required[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df["canonical_id"].isna().any():
        raise ValueError("Missing canonical IDs found.")

    df["canonical_id"] = df["canonical_id"].astype(int)

    eligible = (
        df["observed_frames"].ge(MIN_OBSERVED_FRAMES)
        & df["movement_valid_movement_steps"].ge(MIN_VALID_STEPS)
    )

    # Percentiles are computed only within eligible stitched tracks.
    distance_pct = pd.Series(np.nan, index=df.index, dtype=float)
    speed_pct = pd.Series(np.nan, index=df.index, dtype=float)

    eligible_distance = pd.to_numeric(
        df.loc[eligible, "movement_distance_pixels"], errors="coerce"
    )
    eligible_speed = pd.to_numeric(
        df.loc[eligible, "movement_average_speed_pixels_per_second"],
        errors="coerce",
    )

    distance_pct.loc[eligible] = percentile(eligible_distance)
    speed_pct.loc[eligible] = percentile(eligible_speed)

    distance_component = distance_pct * 0.70
    speed_component = speed_pct * 0.30

    component_values = pd.concat(
        [distance_component, speed_component], axis=1
    )
    component_weights = pd.concat(
        [
            distance_pct.notna().astype(float) * 0.70,
            speed_pct.notna().astype(float) * 0.30,
        ],
        axis=1,
    )

    weighted_sum = component_values.sum(axis=1, min_count=1)
    available_weight = component_weights.sum(axis=1)

    activity = weighted_sum.div(
        available_weight.where(available_weight > 0)
    )
    activity.loc[~eligible] = np.nan

    df["baseline_distance_percentile"] = distance_pct.round(2)
    df["baseline_speed_percentile"] = speed_pct.round(2)
    df["baseline_activity_index"] = activity.round(2)
    df["activity_score_status"] = np.select(
        [
            ~eligible,
            eligible & activity.notna(),
        ],
        [
            "INSUFFICIENT_OBSERVATIONS",
            "OK",
        ],
        default="INSUFFICIENT_MOVEMENT_FEATURES",
    )

    # Flag extreme image-space steps for review without altering raw values.
    df["extreme_step_speed_review"] = (
        df["movement_maximum_step_speed_pixels_per_second"] > 200
    )
    df["score_type"] = "provisional_stitched_within_video_activity_percentile"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    ranking = df[
        [
            "video_id",
            "canonical_id",
            "baseline_activity_index",
            "activity_score_status",
            "observed_frames",
            "movement_valid_movement_steps",
            "movement_coverage_ratio",
            "movement_maximum_step_speed_pixels_per_second",
            "extreme_step_speed_review",
        ]
    ].copy()

    ranking["_eligible"] = ranking["activity_score_status"].eq("OK")
    ranking = ranking.sort_values(
        ["_eligible", "baseline_activity_index"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)

    ranking.insert(
        0,
        "rank",
        pd.Series(pd.array([pd.NA] * len(ranking), dtype="Int64")),
    )

    eligible_rows = ranking["activity_score_status"].eq("OK")
    ranking.loc[eligible_rows, "rank"] = range(
        1, int(eligible_rows.sum()) + 1
    )
    ranking = ranking.drop(columns="_eligible")
    ranking.to_csv(RANKING, index=False)

    scored = int(activity.notna().sum())
    summary = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(INPUT.relative_to(ROOT)),
        "activity_definition": (
            "70% distance percentile + 30% average-speed percentile "
            "among eligible stitched tracks."
        ),
        "score_type": "provisional_within_video_activity_percentile_not_ML",
        "minimum_observed_frames": MIN_OBSERVED_FRAMES,
        "minimum_valid_movement_steps": MIN_VALID_STEPS,
        "total_tracks": int(len(df)),
        "unique_canonical_ids": int(df["canonical_id"].nunique()),
        "eligible_tracks": int(eligible.sum()),
        "ineligible_tracks": int((~eligible).sum()),
        "scored_tracks": scored,
        "unscored_tracks": int(len(df) - scored),
        "tracks_flagged_for_extreme_step_speed": int(
            df["extreme_step_speed_review"].sum()
        ),
        "maximum_step_speed_px_per_second": (
            float(df["movement_maximum_step_speed_pixels_per_second"].max())
        ),
        "outputs": [
            str(OUTPUT.relative_to(ROOT)),
            str(RANKING.relative_to(ROOT)),
        ],
        "limitations": [
            "Scores describe relative image-space activity, not validated football performance.",
            "Pixel-space movement is not calibrated physical distance or speed.",
            "Extreme step speeds are flagged for review but are not corrected.",
            "Eligibility does not prove correct player identity or accurate trajectories.",
            "No supervised performance model is trained.",
        ],
    }

    SUMMARY.write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("STITCHED ACTIVITY BASELINE GENERATED")
    print("Tracks:", len(df))
    print("Eligible:", int(eligible.sum()))
    print("Ineligible:", int((~eligible).sum()))
    print("Scored:", scored)
    print("Extreme-speed review flags:", int(df["extreme_step_speed_review"].sum()))
    print("Baseline:", OUTPUT)
    print("Ranking:", RANKING)
    print("Summary:", SUMMARY)


if __name__ == "__main__":
    main()
