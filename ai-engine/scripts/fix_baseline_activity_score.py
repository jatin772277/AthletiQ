
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

FEATURES_PATH = ROOT / "outputs/dsa/football2_unified_player_features.csv"
RANKING_PATH = ROOT / "outputs/dsa/football2_priority_queue_ranking.csv"
SUMMARY_PATH = ROOT / "outputs/dsa/football2_activity_score_summary.json"

# Provisional eligibility requirements for this video.
MIN_OBSERVED_FRAMES = 30
MIN_VALID_MOVEMENT_STEPS = 1

WEIGHTS = {
    "movement_distance_pixels": 0.70,
    "movement_average_speed_pixels_per_second": 0.30,
}


def percentile_score(series):
    """Calculate percentile scores while preserving missing values."""
    numeric = pd.to_numeric(series, errors="coerce")
    result = numeric.rank(pct=True, method="average") * 100
    return result.where(numeric.notna())


def main():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Features file not found: {FEATURES_PATH}")

    df = pd.read_csv(FEATURES_PATH)

    required = [
        "canonical_id",
        "movement_observed_rows",
        "movement_valid_movement_steps",
        "movement_coverage_ratio",
        "movement_distance_pixels",
        "movement_average_speed_pixels_per_second",
    ]

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["canonical_id"] = pd.to_numeric(
        df["canonical_id"], errors="raise"
    ).astype(int)

    if df["canonical_id"].duplicated().any():
        raise ValueError("canonical_id values must be unique.")

    numeric_columns = [
        "movement_observed_rows",
        "movement_valid_movement_steps",
        "movement_coverage_ratio",
        "movement_distance_pixels",
        "movement_average_speed_pixels_per_second",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Eligibility is independent of the activity score.
    eligible = (
        df["movement_observed_rows"].ge(MIN_OBSERVED_FRAMES)
        & df["movement_valid_movement_steps"].ge(
            MIN_VALID_MOVEMENT_STEPS
        )
    )

    # Calculate percentiles only among eligible tracks.
    distance_pct = pd.Series(np.nan, index=df.index, dtype=float)
    speed_pct = pd.Series(np.nan, index=df.index, dtype=float)

    distance_pct.loc[eligible] = percentile_score(
        df.loc[eligible, "movement_distance_pixels"]
    )
    speed_pct.loc[eligible] = percentile_score(
        df.loc[eligible, "movement_average_speed_pixels_per_second"]
    )

    distance_component = (
        distance_pct * WEIGHTS["movement_distance_pixels"]
    )
    speed_component = (
        speed_pct * WEIGHTS["movement_average_speed_pixels_per_second"]
    )

    components = pd.concat(
        [distance_component, speed_component], axis=1
    )

    # Reweight only when one component is unavailable.
    component_weights = pd.DataFrame(
        {
            "distance": distance_pct.notna()
            * WEIGHTS["movement_distance_pixels"],
            "speed": speed_pct.notna()
            * WEIGHTS["movement_average_speed_pixels_per_second"],
        },
        index=df.index,
    )

    weighted_sum = components.sum(axis=1, min_count=1)
    available_weight = component_weights.sum(axis=1)

    activity = weighted_sum.div(
        available_weight.where(available_weight > 0)
    )

    # Never assign an activity score to an ineligible track.
    activity.loc[~eligible] = np.nan

    df["baseline_distance_percentile"] = distance_pct.round(2)
    df["baseline_speed_percentile"] = speed_pct.round(2)
    df["baseline_activity_index"] = activity.round(2)

    # Reliability remains separate from activity.
    coverage = df["movement_coverage_ratio"].fillna(0).clip(0, 1)
    df["tracking_reliability_index"] = (coverage * 100).round(2)

    df["score_type"] = "within_video_activity_percentile_not_ml"
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

    # Preserve the unified table and all its existing feature columns.
    df.to_csv(FEATURES_PATH, index=False)

    # Rank eligible tracks only. Keep all tracks in the output for auditability.
    ranking_columns = [
        "canonical_id",
        "baseline_activity_index",
        "tracking_reliability_index",
        "activity_score_status",
        "movement_observed_rows",
        "movement_valid_movement_steps",
        "movement_distance_pixels",
        "movement_average_speed_pixels_per_second",
        "movement_coverage_ratio",
    ]

    ranking = df[ranking_columns].copy()
    ranking["_eligible_sort"] = ranking[
        "activity_score_status"
    ].eq("OK")

    ranking = ranking.sort_values(
        ["_eligible_sort", "baseline_activity_index",
         "tracking_reliability_index"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    ranking.insert(
        0,
        "rank",
        pd.Series(
            pd.array([pd.NA] * len(ranking), dtype="Int64")
        ),
    )

    eligible_rows = ranking["activity_score_status"].eq("OK")
    ranking.loc[eligible_rows, "rank"] = range(
        1, int(eligible_rows.sum()) + 1
    )

    ranking = ranking.drop(columns=["_eligible_sort"])
    ranking.to_csv(RANKING_PATH, index=False)

    scored = int(activity.notna().sum())
    unscored = int(activity.isna().sum())
    eligible_count = int(eligible.sum())
    ineligible_count = int((~eligible).sum())

    summary = {
        "schema_version": "1.1",
        "score_type": "within_video_activity_percentile_not_ml",
        "method": {
            "movement_distance_weight": 0.70,
            "average_speed_weight": 0.30,
            "percentile_population": "eligible_tracks_only",
            "minimum_observed_frames": MIN_OBSERVED_FRAMES,
            "minimum_valid_movement_steps": MIN_VALID_MOVEMENT_STEPS,
        },
        "player_ids": int(len(df)),
        "eligible_ids": eligible_count,
        "ineligible_ids": ineligible_count,
        "scored_ids": scored,
        "unscored_ids": unscored,
        "unique_activity_scores": int(activity.nunique()),
        "activity_score_min": (
            float(activity.min()) if activity.notna().any() else None
        ),
        "activity_score_max": (
            float(activity.max()) if activity.notna().any() else None
        ),
        "limitations": [
            "Activity scores are heuristic within-video percentiles, not ML predictions.",
            "Pixel-space movement is not calibrated physical distance or speed.",
            "Tracks below the eligibility thresholds are not activity-ranked.",
            "Eligibility does not guarantee correct identity tracking or accurate movement.",
            "Tracking reliability is reported separately from activity.",
            "No validated football performance score is produced.",
        ],
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("AthletiQ baseline scoring updated.")
    print(f"Total tracks: {len(df)}")
    print(f"Eligible tracks: {eligible_count}")
    print(f"Ineligible tracks: {ineligible_count}")
    print(f"Scored tracks: {scored}")
    print(f"Unscored tracks: {unscored}")
    print(f"Updated features: {FEATURES_PATH}")
    print(f"Updated ranking: {RANKING_PATH}")
    print(f"Updated summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
