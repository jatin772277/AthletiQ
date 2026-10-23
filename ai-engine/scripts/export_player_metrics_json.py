#!/usr/bin/env python3
"""Export the stitched 32-player AthletiQ report with validated baseline scores."""

import json
import math
import os
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

MOVEMENT_INPUT = ROOT / "outputs/dsa/football2_stitched_movement_features.csv"
BASELINE_INPUT = (
    ROOT / "outputs/dsa/stitched_baseline/"
    "football2_stitched_activity_baseline.csv"
)
RANKING_INPUT = (
    ROOT / "outputs/dsa/stitched_baseline/"
    "football2_stitched_activity_ranking.csv"
)
SUMMARY_INPUT = (
    ROOT / "outputs/dsa/stitched_baseline/"
    "football2_stitched_activity_summary.json"
)
OUTPUT = ROOT / "outputs/performance/football2_player_report.json"

MIN_OBSERVED_FRAMES = 30
MIN_VALID_STEPS = 1


def fail(message):
    raise SystemExit(f"EXPORT ABORTED: {message}")


def read_csv(path):
    if not path.is_file():
        fail(f"Required input does not exist: {path}")
    try:
        return pd.read_csv(path)
    except Exception as exc:
        fail(f"Could not read {path}: {exc}")


def require_columns(df, columns, label):
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        fail(f"{label} is missing required columns: {missing}")


def validate_ids(df, label):
    if df["canonical_id"].isna().any():
        fail(f"{label} contains missing canonical IDs.")

    numeric_ids = pd.to_numeric(df["canonical_id"], errors="coerce")
    if numeric_ids.isna().any():
        fail(f"{label} contains non-numeric canonical IDs.")

    if (numeric_ids % 1 != 0).any():
        fail(f"{label} contains non-integer canonical IDs.")

    df["canonical_id"] = numeric_ids.astype(int)

    if df["canonical_id"].duplicated().any():
        duplicates = sorted(
            df.loc[
                df["canonical_id"].duplicated(keep=False), "canonical_id"
            ].unique().tolist()
        )
        fail(f"{label} contains duplicate canonical IDs: {duplicates}")


def number(value, digits=3):
    if value is None or pd.isna(value):
        return None
    result = float(value)
    if not math.isfinite(result):
        return None
    return round(result, digits)


def integer(value):
    if value is None or pd.isna(value):
        return None
    result = float(value)
    if not math.isfinite(result) or not result.is_integer():
        fail(f"Expected an integer value, got {value!r}")
    return int(result)


movement = read_csv(MOVEMENT_INPUT)
baseline = read_csv(BASELINE_INPUT)
ranking = read_csv(RANKING_INPUT)

movement_required = [
    "video_id",
    "canonical_id",
    "first_frame",
    "last_frame",
    "observed_frames",
    "track_span_seconds",
    "movement_coverage_ratio",
    "movement_distance_pixels",
    "movement_valid_movement_time_seconds",
    "movement_valid_movement_steps",
    "movement_excluded_gap_steps",
    "movement_average_speed_pixels_per_second",
    "movement_median_step_speed_pixels_per_second",
    "movement_maximum_step_speed_pixels_per_second",
    "mean_detection_confidence",
    "mean_bbox_width_pixels",
    "mean_bbox_height_pixels",
    "original_track_count",
]
baseline_required = [
    "canonical_id",
    "baseline_distance_percentile",
    "baseline_speed_percentile",
    "baseline_activity_index",
    "activity_score_status",
    "extreme_step_speed_review",
    "score_type",
    "observed_frames",
    "movement_valid_movement_steps",
    "movement_distance_pixels",
    "movement_average_speed_pixels_per_second",
]
ranking_required = [
    "canonical_id",
    "rank",
    "activity_score_status",
]

require_columns(movement, movement_required, "Movement features")
require_columns(baseline, baseline_required, "Activity baseline")
require_columns(ranking, ranking_required, "Activity ranking")

validate_ids(movement, "Movement features")
validate_ids(baseline, "Activity baseline")
validate_ids(ranking, "Activity ranking")

movement_ids = set(movement["canonical_id"])
baseline_ids = set(baseline["canonical_id"])
ranking_ids = set(ranking["canonical_id"])

if movement_ids != baseline_ids:
    fail(
        "Movement and baseline ID sets differ. "
        f"Only movement: {sorted(movement_ids - baseline_ids)}; "
        f"only baseline: {sorted(baseline_ids - movement_ids)}"
    )

if baseline_ids != ranking_ids:
    fail(
        "Baseline and ranking ID sets differ. "
        f"Only baseline: {sorted(baseline_ids - ranking_ids)}; "
        f"only ranking: {sorted(ranking_ids - baseline_ids)}"
    )

# The score file should preserve the movement features from the same build.
check_columns = [
    "observed_frames",
    "movement_valid_movement_steps",
    "movement_distance_pixels",
    "movement_average_speed_pixels_per_second",
]
check = movement[
    ["canonical_id"] + check_columns
].merge(
    baseline[["canonical_id"] + check_columns],
    on="canonical_id",
    suffixes=("_movement", "_baseline"),
    validate="one_to_one",
)

for column in check_columns:
    left = pd.to_numeric(check[f"{column}_movement"], errors="coerce")
    right = pd.to_numeric(check[f"{column}_baseline"], errors="coerce")
    equal = (left.isna() & right.isna()) | (
        left.notna() & right.notna()
        & ((left - right).abs() <= 1e-6)
    )
    if not equal.all():
        bad_ids = check.loc[~equal, "canonical_id"].tolist()
        fail(f"Movement/baseline mismatch in {column}, IDs: {bad_ids}")

# Validate eligibility independently instead of trusting only a stored status.
baseline["observed_frames"] = pd.to_numeric(
    baseline["observed_frames"], errors="coerce"
)
baseline["movement_valid_movement_steps"] = pd.to_numeric(
    baseline["movement_valid_movement_steps"], errors="coerce"
)
expected_eligible = (
    baseline["observed_frames"].ge(MIN_OBSERVED_FRAMES)
    & baseline["movement_valid_movement_steps"].ge(MIN_VALID_STEPS)
)
status = baseline["activity_score_status"].astype(str)
expected_status = expected_eligible.map({True: "OK", False: "INSUFFICIENT_OBSERVATIONS"})

if not status.equals(expected_status):
    mismatched = baseline.loc[
        status != expected_status, "canonical_id"
    ].tolist()
    fail(f"Eligibility/status mismatch for IDs: {mismatched}")

scores = pd.to_numeric(baseline["baseline_activity_index"], errors="coerce")
if scores[expected_eligible].isna().any():
    fail("An eligible player is missing its activity score.")
if scores[~expected_eligible].notna().any():
    fail("An ineligible player has a numeric activity score.")
if (~scores[expected_eligible].map(math.isfinite)).any():
    fail("An eligible player has a non-finite activity score.")

# Rank values are allowed only for eligible players.
rank_status = ranking["activity_score_status"].astype(str)
rank_values = pd.to_numeric(ranking["rank"], errors="coerce")
if rank_values[rank_status == "OK"].isna().any():
    fail("An eligible player is missing a ranking.")
if rank_values[rank_status != "OK"].notna().any():
    fail("An ineligible player has a ranking.")

# Join only after all identity and consistency checks pass.
df = movement.merge(
    baseline[
        [
            "canonical_id",
            "baseline_distance_percentile",
            "baseline_speed_percentile",
            "baseline_activity_index",
            "activity_score_status",
            "extreme_step_speed_review",
            "score_type",
        ]
    ],
    on="canonical_id",
    how="inner",
    validate="one_to_one",
)
df = df.merge(
    ranking[["canonical_id", "rank"]],
    on="canonical_id",
    how="inner",
    validate="one_to_one",
)
df = df.sort_values("canonical_id").reset_index(drop=True)

players = []
for _, row in df.iterrows():
    eligible = row["activity_score_status"] == "OK"
    rank = number(row["rank"], 0) if eligible else None
    if rank is not None:
        rank = int(rank)

    players.append({
        "player_id": int(row["canonical_id"]),
        "tracking": {
            "first_frame": integer(row["first_frame"]),
            "last_frame": integer(row["last_frame"]),
            "observed_frames": integer(row["observed_frames"]),
            "track_span_seconds": number(row["track_span_seconds"]),
            "track_coverage_ratio": number(row["movement_coverage_ratio"]),
            "valid_movement_steps": integer(
                row["movement_valid_movement_steps"]
            ),
            "valid_movement_time_seconds": number(
                row["movement_valid_movement_time_seconds"]
            ),
            "excluded_gap_steps": integer(
                row["movement_excluded_gap_steps"]
            ),
        },
        "detection_quality": {
            "mean_confidence": number(row["mean_detection_confidence"]),
        },
        "movement": {
            "distance": number(row["movement_distance_pixels"]),
            "distance_unit": "pixels",
            "average_speed": number(
                row["movement_average_speed_pixels_per_second"]
            ),
            "median_step_speed": number(
                row["movement_median_step_speed_pixels_per_second"]
            ),
            "maximum_step_speed": number(
                row["movement_maximum_step_speed_pixels_per_second"]
            ),
            "speed_unit": "pixels_per_second",
            "extreme_step_speed_review": bool(
                row["extreme_step_speed_review"]
            ),
        },
        "activity": {
            "score": number(row["baseline_activity_index"]),
            "distance_percentile": number(
                row["baseline_distance_percentile"]
            ),
            "speed_percentile": number(row["baseline_speed_percentile"]),
            "rank": rank,
            "eligible": bool(eligible),
            "status": str(row["activity_score_status"]),
            "score_type": str(row["score_type"]),
        },
        "identity": {
            "type": "stitched_canonical_tracking_id",
            "real_player_identified": False,
            "original_track_ids": (
                str(row["original_track_ids"]).split(",")
                if pd.notna(row.get("original_track_ids"))
                else []
            ),
            "original_track_count": integer(row["original_track_count"]),
        },
    })

try:
    summary_source = json.loads(SUMMARY_INPUT.read_text(encoding="utf-8"))
except Exception as exc:
    fail(f"Could not read activity summary JSON: {exc}")

eligible_count = sum(p["activity"]["eligible"] for p in players)
ineligible_count = len(players) - eligible_count
coverage = pd.to_numeric(df["movement_coverage_ratio"], errors="coerce")
confidence = pd.to_numeric(df["mean_detection_confidence"], errors="coerce")

report = {
    "schema_version": "2.0",
    "project": "AthletiQ",
    "video_id": "football2",
    "source": {
        "movement_features": str(MOVEMENT_INPUT.relative_to(ROOT)),
        "activity_baseline": str(BASELINE_INPUT.relative_to(ROOT)),
        "activity_ranking": str(RANKING_INPUT.relative_to(ROOT)),
        "activity_summary": str(SUMMARY_INPUT.relative_to(ROOT)),
    },
    "status": "stitched_tracking_and_provisional_activity_available",
    "physical_calibration": {
        "validated": False,
        "physical_metrics_enabled": False,
        "reason": (
            "Pitch calibration has not passed independent validation. "
            "Movement distance and speed remain image-space pixel metrics."
        ),
    },
    "scoring": {
        "type": "provisional_within_video_activity_percentile",
        "supervised_performance_score": None,
        "model_trained": False,
        "eligibility": {
            "minimum_observed_frames": MIN_OBSERVED_FRAMES,
            "minimum_valid_movement_steps": MIN_VALID_STEPS,
            "eligible_players": eligible_count,
            "ineligible_players": ineligible_count,
        },
        "source_summary_schema_version": summary_source.get("schema_version"),
        "warning": (
            "Activity score is a provisional within-video ranking proxy, "
            "not a validated measure of football skill or performance."
        ),
    },
    "units": {
        "distance": "pixels",
        "speed": "pixels_per_second",
        "time": "seconds",
        "confidence": "0_to_1",
        "activity_score": "within_video_percentile_index_0_to_100",
    },
    "summary": {
        "tracked_player_ids": len(players),
        "eligible_players": eligible_count,
        "ineligible_players": ineligible_count,
        "extreme_step_speed_review_count": sum(
            p["movement"]["extreme_step_speed_review"] for p in players
        ),
        "mean_track_coverage_ratio": number(coverage.mean()),
        "mean_detection_confidence": number(confidence.mean()),
        "total_observed_frames": int(df["observed_frames"].sum()),
        "interpretation_warning": (
            "Pixel movement depends on camera perspective and image motion. "
            "It is not physical distance or running speed. Extreme step "
            "speeds are flagged for review, not automatically corrected."
        ),
    },
    "players": players,
}

# Write to a temporary file, then atomically replace the report only on success.
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
temp_path = None
try:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=OUTPUT.parent,
        prefix=OUTPUT.name + ".",
        suffix=".tmp",
        delete=False,
    ) as temp:
        temp_path = Path(temp.name)
        json.dump(report, temp, indent=2, allow_nan=False)
        temp.write("\n")
        temp.flush()
        os.fsync(temp.fileno())

    # Validate the exact serialized JSON before replacing the live report.
    json.loads(temp_path.read_text(encoding="utf-8"))
    temp_path.replace(OUTPUT)
except Exception:
    if temp_path is not None:
        temp_path.unlink(missing_ok=True)
    raise

print("AthletiQ stitched JSON export complete")
print(f"Player IDs: {len(players)}")
print(f"Eligible: {eligible_count}")
print(f"Ineligible: {ineligible_count}")
print(
    "Extreme-speed review flags: "
    f"{report['summary']['extreme_step_speed_review_count']}"
)
print(f"Output: {OUTPUT}")
print("Physical metrics: DISABLED")
print("Supervised performance score: NOT AVAILABLE")
