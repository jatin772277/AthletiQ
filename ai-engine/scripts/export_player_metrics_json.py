#!/usr/bin/env python3
"""Export AthletiQ player metrics to dashboard-ready JSON."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "outputs/performance/football2_player_metrics.csv"
OUTPUT = ROOT / "outputs/performance/football2_player_report.json"

df = pd.read_csv(INPUT)

required = [
    "canonical_id",
    "first_frame",
    "last_frame",
    "observed_frames",
    "observed_time_seconds",
    "track_span_seconds",
    "track_coverage_ratio",
    "mean_detection_confidence",
    "median_detection_confidence",
    "movement_distance_pixels",
    "average_speed_pixels_per_second",
    "maximum_step_speed_pixels_per_second",
    "valid_movement_steps",
    "excluded_gap_steps",
]

missing = sorted(set(required) - set(df.columns))
if missing:
    raise SystemExit(f"Missing required columns: {missing}")

if df["canonical_id"].duplicated().any():
    raise SystemExit("Duplicate canonical_id values found.")

def number(value, digits=3):
    if pd.isna(value):
        return None
    return round(float(value), digits)

players = []

for _, row in df.iterrows():
    players.append({
        "player_id": int(row["canonical_id"]),
        "tracking": {
            "first_frame": int(row["first_frame"]),
            "last_frame": int(row["last_frame"]),
            "observed_frames": int(row["observed_frames"]),
            "observed_time_seconds": number(
                row["observed_time_seconds"]
            ),
            "track_span_seconds": number(
                row["track_span_seconds"]
            ),
            "track_coverage_ratio": number(
                row["track_coverage_ratio"]
            ),
            "valid_movement_steps": int(
                row["valid_movement_steps"]
            ),
            "excluded_gap_steps": int(
                row["excluded_gap_steps"]
            ),
        },
        "detection_quality": {
            "mean_confidence": number(
                row["mean_detection_confidence"]
            ),
            "median_confidence": number(
                row["median_detection_confidence"]
            ),
        },
        "movement": {
            "distance": number(
                row["movement_distance_pixels"]
            ),
            "distance_unit": "pixels",
            "average_speed": number(
                row["average_speed_pixels_per_second"]
            ),
            "maximum_step_speed": number(
                row["maximum_step_speed_pixels_per_second"]
            ),
            "speed_unit": "pixels_per_second",
        },
        "identity": {
            "type": "canonical_tracking_id",
            "real_player_identified": False,
        },
    })

players.sort(
    key=lambda player: player["player_id"]
)

coverage = df["track_coverage_ratio"].dropna()
confidence = df["mean_detection_confidence"].dropna()

report = {
    "schema_version": "1.0",
    "project": "AthletiQ",
    "source": str(INPUT.relative_to(ROOT)),
    "status": "tracking_metrics_available",
    "physical_calibration": {
        "validated": False,
        "physical_metrics_enabled": False,
        "reason": (
            "Pitch calibration has not passed independent validation."
        ),
    },
    "units": {
        "distance": "pixels",
        "speed": "pixels_per_second",
        "time": "seconds",
        "confidence": "0_to_1",
    },
    "summary": {
        "tracked_player_ids": len(players),
        "mean_track_coverage_ratio": number(coverage.mean()),
        "mean_detection_confidence": number(confidence.mean()),
        "total_observed_frames": int(
            df["observed_frames"].sum()
        ),
        "interpretation_warning": (
            "Pixel movement depends on camera perspective and movement "
            "in the image. It is not physical distance or running speed."
        ),
    },
    "players": players,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(
    json.dumps(report, indent=2, allow_nan=False),
    encoding="utf-8",
)

print("AthletiQ JSON export complete")
print(f"Player IDs: {len(players)}")
print(f"Output: {OUTPUT}")
print("Physical metrics: DISABLED")
