"""Provisional, observation-normalized activity scoring for AthletiQ."""

from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "outputs/dsa/football2_ml_features.csv"
OUTPUT = ROOT / "outputs/performance/football2_activity_scores.json"

# Explicit proxy weights, not learned performance weights.
WEIGHTS = {
    "movement_distance_rate": 0.25,
    "movement_average_speed_pixels_per_second": 0.25,
    "sprint_count_per_observed_minute": 0.15,
    "sprint_distance_rate": 0.15,
    "possession_share": 0.10,
    "graph_weighted_degree": 0.10,
}


def numeric(df, column):
    if column not in df:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def percentile(series):
    """Percentile rank among available tracks; missing stays missing."""
    x = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=x.index, dtype=float)
    valid = x.notna()
    if valid.sum() == 1:
        out.loc[valid] = 50.0
    elif valid.sum() > 1:
        out.loc[valid] = x.loc[valid].rank(pct=True, method="average") * 100
    return out


def main():
    if not FEATURES.exists():
        raise SystemExit(f"Missing feature table: {FEATURES}")

    df = pd.read_csv(FEATURES)
    required = {"canonical_id", "movement_distance_pixels",
                "movement_observed_rows", "movement_coverage_ratio"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")
    if df.empty or df["canonical_id"].isna().any():
        raise SystemExit("Empty dataset or missing IDs.")
    if df["canonical_id"].duplicated().any():
        raise SystemExit("Expected one row per temporary track ID.")

    observed = numeric(df, "spatial_observed_seconds")
    if observed.isna().all():
        observed = numeric(df, "movement_track_span_seconds")

    # Rates reduce (but do not eliminate) bias from tracks of different lengths.
    safe_seconds = observed.where(observed > 0)
    df["movement_distance_rate"] = (
        numeric(df, "movement_distance_pixels") / safe_seconds
    )
    df["sprint_count_per_observed_minute"] = (
        numeric(df, "movement_experimental_sprint_count")
        / safe_seconds * 60
    )
    df["sprint_distance_rate"] = (
        numeric(df, "movement_experimental_sprint_distance_pixels")
        / safe_seconds
    )
    df["possession_share"] = (
        numeric(df, "possession_estimated_possession_seconds")
        / safe_seconds
    ).clip(0, 1)

    df["movement_average_speed_pixels_per_second"] = numeric(
        df, "movement_average_speed_pixels_per_second"
    )
    df["graph_weighted_degree"] = numeric(df, "graph_weighted_degree")

    component_scores = {
        name: percentile(df[name]) for name in WEIGHTS
    }

    numerator = pd.Series(0.0, index=df.index)
    denominator = pd.Series(0.0, index=df.index)
    for name, weight in WEIGHTS.items():
        score = component_scores[name]
        valid = score.notna()
        numerator += score.fillna(0) * weight
        denominator += valid.astype(float) * weight

    df["observed_activity_index"] = (
        numerator.div(denominator.where(denominator > 0))
    ).round(2)

    coverage = numeric(df, "movement_coverage_ratio").clip(0, 1)
    team_conf = numeric(df, "team_confidence").clip(0, 1)

    players = []
    for idx, row in df.sort_values("canonical_id").iterrows():
        components = {
            name: (
                None if pd.isna(score.loc[idx])
                else round(float(score.loc[idx]), 2)
            )
            for name, score in component_scores.items()
        }
        players.append({
            "player_id": int(row["canonical_id"]),
            "identity_status": "temporary_track_id",
            "observed_activity_index": (
                None if pd.isna(row["observed_activity_index"])
                else float(row["observed_activity_index"])
            ),
            "observed_seconds": (
                None if pd.isna(observed.loc[idx])
                else round(float(observed.loc[idx]), 3)
            ),
            "movement_coverage_ratio": (
                None if pd.isna(coverage.loc[idx])
                else round(float(coverage.loc[idx]), 4)
            ),
            "team_classifier_confidence": (
                None if pd.isna(team_conf.loc[idx])
                else round(float(team_conf.loc[idx]), 4)
            ),
            "tracking_reliability_score": None,
            "activity_components_percentile": components,
            "supervised_performance_score": None,
            "supervised_model_status": "not_trained_no_reviewed_labels",
        })

    report = {
        "schema_version": "1.1",
        "project": "AthletiQ",
        "status": "provisional_track_level_scores",
        "player_count": len(players),
        "score_definitions": {
            "observed_activity_index": (
                "Weighted percentile index using observation-normalized "
                "movement and sprint proxies, estimated possession share, "
                "and graph interaction. Not a validated football rating."
            ),
            "movement_coverage_ratio": (
                "Tracking coverage feature from the input table; not a "
                "detector-confidence estimate."
            ),
            "team_classifier_confidence": (
                "Team-classification confidence from the input table; "
                "reported separately and not treated as tracking reliability."
            ),
            "supervised_performance_score": (
                "Unavailable until reviewed labels and defensible evaluation exist."
            ),
        },
        "weights": WEIGHTS,
        "limitations": [
            "Tracks are temporary IDs, not confirmed player identities.",
            "Pixel-space movement remains affected by camera perspective.",
            "Rate normalization does not correct occlusion or tracking fragmentation.",
            "Possession, sprint, and graph features are estimates/proxies.",
            "Scores are relative to this video and must be recalculated after stitching.",
            "No supervised performance model is claimed.",
        ],
        "players": players,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"Saved: {OUTPUT}")
    print(f"Tracks: {len(players)}")
    print("Supervised model: not trained")
    print("Tracking reliability: deliberately not combined into an unvalidated score")


if __name__ == "__main__":
    main()
