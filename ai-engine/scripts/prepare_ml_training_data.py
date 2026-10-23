
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd


# ============================================================
# ATHLETIQ ML DATA PREPARATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DSA_DIR = OUTPUT_DIR / "dsa"
TEAM_DIR = OUTPUT_DIR / "team_identification"

UNIFIED_FEATURES_PATH = (
    DSA_DIR / "football2_unified_player_features.csv"
)
TEAM_PREDICTIONS_PATH = (
    TEAM_DIR / "football2_team_predictions.csv"
)

ML_FEATURES_PATH = DSA_DIR / "football2_ml_features.csv"
LABEL_TEMPLATE_PATH = DSA_DIR / "player_performance_labels_template.csv"
AUDIT_PATH = DSA_DIR / "football2_ml_dataset_audit.csv"
SUMMARY_PATH = DSA_DIR / "football2_ml_dataset_summary.json"

VALID_TEAMS = {"team_a", "team_b"}

CANDIDATE_FEATURES = [
    "movement_distance_pixels",
    "movement_track_span_seconds",
    "movement_observed_rows",
    "movement_coverage_ratio",
    "movement_valid_movement_steps",
    "movement_average_speed_pixels_per_second",
    "movement_maximum_step_speed_pixels_per_second",
    "movement_median_step_speed_pixels_per_second",
    "movement_maximum_positive_acceleration_pixels_per_second_squared",
    "movement_maximum_deceleration_magnitude_pixels_per_second_squared",
    "movement_experimental_sprint_count",
    "movement_experimental_sprint_distance_pixels",
    "movement_experimental_sprint_duration_seconds",
    "spatial_observed_seconds",
    "spatial_zone_row",
    "spatial_zone_col",
    "possession_estimated_possession_frames",
    "possession_estimated_possession_seconds",
    "possession_mean_ball_distance_px",
    "possession_mean_heuristic_confidence",
    "kdtree_observed_frames",
    "kdtree_mean_nearest_distance_px",
    "kdtree_median_nearest_distance_px",
    "kdtree_mean_neighbors_within_radius",
    "graph_degree",
    "graph_weighted_degree",
    "graph_same_team_edges",
    "graph_opponent_edges",
    "graph_component_size",
]

LABEL_TEMPLATE_COLUMNS = [
    "canonical_id",
    "team_label",
    "reviewed_player_identity",
    "reviewed_performance_score_0_100",
    "label_source",
    "reviewer_notes",
]


# ============================================================
# HELPERS
# ============================================================

def normalize_id(series):
    return series.astype("string").str.strip()


def normalize_team(series):
    values = series.astype("string").str.strip().str.lower()
    return values.where(values.isin(VALID_TEAMS), pd.NA)


def mode_or_missing(series):
    values = series.dropna()
    if values.empty:
        return pd.NA

    modes = values.mode()
    return modes.iloc[0] if not modes.empty else pd.NA


def numeric_mean_or_missing(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else np.nan


# ============================================================
# TEAM MAPPING
# ============================================================

def load_team_labels(player_ids):
    empty = pd.DataFrame(
        columns=["canonical_id", "team_label", "team_confidence"]
    )

    if not TEAM_PREDICTIONS_PATH.exists():
        print(f"[WARNING] Team file not found: {TEAM_PREDICTIONS_PATH}")
        return empty

    df = pd.read_csv(TEAM_PREDICTIONS_PATH)

    required = {"canonical_id", "team_prediction"}
    if not required.issubset(df.columns):
        print(
            "[WARNING] Team file is missing required columns:",
            sorted(required - set(df.columns)),
        )
        return empty

    df = df.copy()
    df["canonical_id"] = normalize_id(df["canonical_id"])
    df["_frame_team"] = normalize_team(df["team_prediction"])

    if "aggregated_team" in df.columns:
        df["_aggregated_team"] = normalize_team(df["aggregated_team"])
    else:
        df["_aggregated_team"] = pd.NA

    # Prefer the aggregated team prediction for each track.
    aggregated = (
        df.groupby("canonical_id")["_aggregated_team"]
        .agg(mode_or_missing)
    )

    # Fallback: majority vote of frame-level predictions.
    majority = (
        df.groupby("canonical_id")["_frame_team"]
        .agg(mode_or_missing)
    )

    labels = pd.DataFrame({
        "canonical_id": aggregated.index,
        "team_label": aggregated.values,
    })

    labels["team_label"] = labels["team_label"].astype("string")

    invalid = (
        labels["team_label"].isna()
        | ~labels["team_label"].isin(VALID_TEAMS)
    )

    fallback_map = majority.to_dict()
    labels.loc[invalid, "team_label"] = (
        labels.loc[invalid, "canonical_id"].map(fallback_map)
    )

    confidence_col = None
    if "aggregated_team_confidence" in df.columns:
        confidence_col = "aggregated_team_confidence"
    elif "team_confidence" in df.columns:
        confidence_col = "team_confidence"

    if confidence_col:
        df[confidence_col] = pd.to_numeric(
            df[confidence_col], errors="coerce"
        )
        confidence = (
            df.groupby("canonical_id")[confidence_col]
            .agg(numeric_mean_or_missing)
            .rename("team_confidence")
            .reset_index()
        )

        labels = labels.merge(
            confidence,
            on="canonical_id",
            how="left",
            validate="one_to_one",
        )
    else:
        labels["team_confidence"] = np.nan

    labels = labels.drop_duplicates("canonical_id")

    valid_ids = set(player_ids.dropna().astype(str))
    labels = labels[labels["canonical_id"].isin(valid_ids)].copy()

    labels["team_label"] = normalize_team(labels["team_label"])

    print(f"[TEAM] Player IDs with team labels: {labels['team_label'].notna().sum()}")
    print("[TEAM] Team-label distribution:")
    print(labels["team_label"].value_counts(dropna=False).to_string())

    return labels


# ============================================================
# REVIEW TEMPLATE
# ============================================================

def prepare_label_template(player_ids, team_labels):
    """
    Preserve existing human-entered annotations while refreshing
    current team predictions. Never generate performance scores.
    """
    current = pd.DataFrame({
        "canonical_id": normalize_id(player_ids).drop_duplicates()
    })

    team_lookup = team_labels.set_index("canonical_id")["team_label"].to_dict()
    current["team_label"] = current["canonical_id"].map(team_lookup)

    # Ensure the output schema exists.
    for column in LABEL_TEMPLATE_COLUMNS:
        if column not in current.columns:
            current[column] = pd.NA

    if LABEL_TEMPLATE_PATH.exists():
        try:
            old = pd.read_csv(
                LABEL_TEMPLATE_PATH,
                dtype={"canonical_id": "string"},
            )

            # Remove duplicate column names before accessing columns.
            old = old.loc[:, ~old.columns.duplicated()].copy()

            if "canonical_id" in old.columns:
                old["canonical_id"] = normalize_id(old["canonical_id"])
                old = old.dropna(subset=["canonical_id"])

                # If duplicate IDs exist, keep the last non-empty
                # reviewed value for each annotation field.
                annotation_columns = [
                    "reviewed_player_identity",
                    "reviewed_performance_score_0_100",
                    "label_source",
                    "reviewer_notes",
                ]

                for column in annotation_columns:
                    if column not in old.columns:
                        old[column] = pd.NA

                old = old[
                    ["canonical_id"] + annotation_columns
                ].copy()

                def last_nonempty(series):
                    values = series.dropna().astype(str).str.strip()
                    values = values[values != ""]
                    return values.iloc[-1] if not values.empty else pd.NA

                old = (
                    old.groupby("canonical_id", as_index=False)
                    .agg({
                        column: last_nonempty
                        for column in annotation_columns
                    })
                )

                # IMPORTANT FIX:
                # canonical_id is the join key and must not be included
                # again in the list of columns merged from the old file.
                current = current.merge(
                    old[
                        ["canonical_id"] + annotation_columns
                    ],
                    on="canonical_id",
                    how="left",
                    validate="one_to_one",
                )

                # Refresh team labels from current predictions.
                current["team_label"] = current["canonical_id"].map(
                    team_lookup
                )

        except Exception as exc:
            print(
                "[WARNING] Could not preserve the existing label template:",
                str(exc),
            )
            print(
                "[WARNING] Creating a new template. "
                "Check the old file if it contains manual annotations."
            )

            current = pd.DataFrame({
                "canonical_id": normalize_id(player_ids).drop_duplicates()
            })
            current["team_label"] = current["canonical_id"].map(team_lookup)

            for column in LABEL_TEMPLATE_COLUMNS:
                if column not in current.columns:
                    current[column] = pd.NA

    # Add any missing schema columns, then enforce the exact order.
    for column in LABEL_TEMPLATE_COLUMNS:
        if column not in current.columns:
            current[column] = pd.NA

    current = current[LABEL_TEMPLATE_COLUMNS].copy()
    current.to_csv(LABEL_TEMPLATE_PATH, index=False)

    print(f"[LABELS] Review template: {LABEL_TEMPLATE_PATH}")


# ============================================================
# MAIN
# ============================================================

def main():
    DSA_DIR.mkdir(parents=True, exist_ok=True)

    if not UNIFIED_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Unified feature file not found: {UNIFIED_FEATURES_PATH}"
        )

    print("=" * 58)
    print("ATHLETIQ ML DATA PREPARATION")
    print("=" * 58)

    features = pd.read_csv(UNIFIED_FEATURES_PATH)

    if "canonical_id" not in features.columns:
        raise ValueError(
            "canonical_id column missing from unified features. "
            f"Columns found: {features.columns.tolist()}"
        )

    features = features.copy()
    features["canonical_id"] = normalize_id(features["canonical_id"])
    features = features.dropna(subset=["canonical_id"])

    if features["canonical_id"].duplicated().any():
        warnings.warn(
            "Duplicate canonical IDs found in unified features. "
            "Keeping the first row for each ID."
        )
        features = features.drop_duplicates(
            "canonical_id", keep="first"
        )

    player_ids = features["canonical_id"].copy()

    print(f"[LOAD] Player IDs: {len(features)}")

    available = [
        column for column in CANDIDATE_FEATURES
        if column in features.columns
    ]
    missing = [
        column for column in CANDIDATE_FEATURES
        if column not in features.columns
    ]

    print(f"[FEATURES] Candidate features available: {len(available)}")
    print(f"[FEATURES] Candidate features missing: {len(missing)}")

    if missing:
        print("[FEATURES] Missing columns:")
        for column in missing:
            print(f"  - {column}")

    for column in available:
        features[column] = pd.to_numeric(
            features[column], errors="coerce"
        )
        features[column] = features[column].replace(
            [np.inf, -np.inf], np.nan
        )

    team_labels = load_team_labels(player_ids)

    ml_df = features[["canonical_id"] + available].merge(
        team_labels,
        on="canonical_id",
        how="left",
        validate="one_to_one",
    )

    ml_df.to_csv(ML_FEATURES_PATH, index=False)

    # --------------------------------------------------------
    # Feature audit
    # --------------------------------------------------------
    audit_rows = []

    for column in available:
        series = ml_df[column]
        nonmissing = series.dropna()
        missing_count = int(series.isna().sum())
        total = len(series)

        audit_rows.append({
            "feature": column,
            "missing_count": missing_count,
            "missing_percent": round(
                100 * missing_count / total, 2
            ) if total else 0.0,
            "unique_nonmissing_values": int(nonmissing.nunique()),
            "candidate_for_training": bool(nonmissing.nunique() > 1),
        })

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(AUDIT_PATH, index=False)

    # --------------------------------------------------------
    # Review-label template
    # --------------------------------------------------------
    prepare_label_template(player_ids, team_labels)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    team_counts = ml_df["team_label"].value_counts(dropna=False)

    summary = {
        "schema_version": "1.2",
        "player_track_count": int(len(ml_df)),
        "candidate_feature_count": len(available),
        "missing_candidate_feature_columns": missing,
        "team_labels_available": int(ml_df["team_label"].notna().sum()),
        "team_label_distribution": {
            str(key) if pd.notna(key) else "missing": int(value)
            for key, value in team_counts.items()
        },
        "team_labels_are_predictions_not_ground_truth": True,
        "performance_labels_invented": False,
        "model_trained": False,
        "candidate_features": available,
        "output_files": {
            "ml_features": str(ML_FEATURES_PATH),
            "label_template": str(LABEL_TEMPLATE_PATH),
            "feature_audit": str(AUDIT_PATH),
        },
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------
    print("\n" + "=" * 58)
    print("ATHLETIQ ML DATA PREPARATION COMPLETE")
    print("=" * 58)
    print(f"Player IDs: {len(ml_df)}")
    print(f"Candidate features available: {len(available)}")
    print(f"Candidate features missing: {len(missing)}")
    print(
        f"Team labels available: {ml_df['team_label'].notna().sum()}"
    )

    print(f"\nML feature table: {ML_FEATURES_PATH}")
    print(f"Review label template: {LABEL_TEMPLATE_PATH}")
    print(f"Feature audit: {AUDIT_PATH}")
    print(f"Dataset summary: {SUMMARY_PATH}")

    print("\nFeature missingness:")
    print(audit.to_string(index=False) if not audit.empty else "No features found.")

    print(
        "\n[IMPORTANT] Team assignments are predictions, not verified "
        "ground truth."
    )
    print(
        "[IMPORTANT] Player IDs are temporary tracks. Verify identity "
        "before assigning reviewed performance scores."
    )
    print(
        "[IMPORTANT] No performance labels were invented and no ML "
        "model was trained."
    )


if __name__ == "__main__":
    main()
