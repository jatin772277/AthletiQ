#!/usr/bin/env python3
"""Transparent activity and tracking-quality baselines; not an ML model."""

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/performance/football2_player_metrics.csv"
OUTPUT = ROOT / "outputs/performance/football2_scoring_baseline.json"
LABELS = ROOT / "outputs/performance/player_score_labels_template.csv"

def main():
    if not SOURCE.exists():
        raise SystemExit(f"Missing metrics file: {SOURCE}")

    df = pd.read_csv(SOURCE)
    required = [
        "canonical_id", "track_coverage_ratio",
        "mean_detection_confidence", "movement_distance_pixels",
        "average_speed_pixels_per_second", "observed_frames",
        "excluded_gap_steps",
    ]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise SystemExit(f"Missing columns: {missing}")
    if df.empty or df["canonical_id"].isna().any():
        raise SystemExit("Empty data or missing player IDs.")
    if df["canonical_id"].duplicated().any():
        raise SystemExit("Player IDs must be unique.")

    for col in required[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    def percentile(col):
        return df[col].rank(pct=True, method="average").fillna(0) * 100

    # Relative image activity within this video, not physical performance.
    df["movement_activity_index"] = (
        0.7 * percentile("movement_distance_pixels")
        + 0.3 * percentile("average_speed_pixels_per_second")
    ).round(2)

    # Track reliability is not a football performance score.
    coverage = df["track_coverage_ratio"].fillna(0).clip(0, 1)
    confidence = df["mean_detection_confidence"].fillna(0).clip(0, 1)
    df["tracking_reliability_score"] = (
        60 * coverage + 40 * confidence
    ).round(2)

    def number(value, digits=3):
        return None if pd.isna(value) else round(float(value), digits)

    players = []
    for _, r in df.sort_values("canonical_id").iterrows():
        players.append({
            "player_id": int(r["canonical_id"]),
            "identity_status": "tracking_id_only",
            "movement_activity_index": float(r["movement_activity_index"]),
            "tracking_reliability_score": float(r["tracking_reliability_score"]),
            "performance_score": None,
            "performance_score_status": "no_validated_performance_targets",
            "evidence": {
                "movement_distance_pixels": number(r["movement_distance_pixels"]),
                "average_speed_pixels_per_second": number(
                    r["average_speed_pixels_per_second"]
                ),
                "track_coverage_ratio": number(r["track_coverage_ratio"], 4),
                "mean_detection_confidence": number(
                    r["mean_detection_confidence"], 4
                ),
                "observed_frames": (
                    None if pd.isna(r["observed_frames"])
                    else int(r["observed_frames"])
                ),
                "excluded_gap_steps": (
                    None if pd.isna(r["excluded_gap_steps"])
                    else int(r["excluded_gap_steps"])
                ),
            },
        })

    report = {
        "schema_version": "1.0",
        "project": "AthletiQ",
        "source": str(SOURCE.relative_to(ROOT)),
        "scoring_method": "relative_baseline_not_ML",
        "player_count": len(players),
        "score_definitions": {
            "movement_activity_index": (
                "70% movement-distance percentile + 30% average-speed "
                "percentile among IDs in this video; pixel-space activity only."
            ),
            "tracking_reliability_score": (
                "60% track coverage + 40% mean detection confidence; "
                "tracking quality, not performance."
            ),
            "performance_score": (
                "Unavailable until defensible performance targets and "
                "evaluation are available."
            ),
        },
        "limitations": [
            "Tracking IDs are not verified player identities.",
            "Pixel movement depends on perspective and camera framing.",
            "Relative indices compare only IDs in this video.",
            "No trained ML performance model is claimed.",
            "Physical metrics remain disabled pending validated calibration.",
        ],
        "physical_calibration": {
            "validated": False,
            "physical_metrics_enabled": False,
        },
        "players": players,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    # Create the coach-label template only once; never overwrite annotations.
    if not LABELS.exists():
        pd.DataFrame({
            "canonical_id": sorted(df["canonical_id"].astype(int)),
            "coach_performance_score_0_100": [""] * len(df),
            "role_or_position": [""] * len(df),
            "reviewer_notes": [""] * len(df),
        }).to_csv(LABELS, index=False)

    print("AthletiQ scoring baseline generated.")
    print(f"Tracked IDs: {len(players)}")
    print(f"JSON: {OUTPUT}")
    print(f"Coach-label template: {LABELS}")
    print("ML performance model: NOT TRAINED")
    print("Physical metrics: DISABLED")

if __name__ == "__main__":
    main()
