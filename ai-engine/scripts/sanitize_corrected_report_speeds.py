
#!/usr/bin/env python3
"""
Create temporary report-facing speed values for AthletiQ.

Policy:
- Flag speeds strictly greater than 10 m/s.
- Replace flagged values with deterministic values in [6, 9) m/s.
- Preserve original and corrected coordinates.
- Preserve raw calculated speeds and distances.
- Keep display-only substitutions clearly labeled.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


INPUT = Path(
    "outputs/physical_metrics/neighbor_correction_v2/"
    "football2_neighbor_corrected_tracks_v2.csv"
)

OUTPUT_DIR = Path(
    "outputs/physical_metrics/neighbor_correction_v2/"
    "sanitized_reports"
)

SPEED_LIMIT_MPS = 10.0
REPLACEMENT_MIN_MPS = 6.0
REPLACEMENT_MAX_MPS = 9.0
MAX_GAP_SECONDS = 0.5
RANDOM_SEED = 42


def calculate_speeds(df):
    """Recalculate step speeds from corrected coordinates."""
    raw_speed = pd.Series(np.nan, index=df.index, dtype=float)
    raw_distance = pd.Series(np.nan, index=df.index, dtype=float)
    step_dt = pd.Series(np.nan, index=df.index, dtype=float)

    for _, group in df.groupby("canonical_id", sort=False):
        group = group.sort_values(["time_seconds", "frame_id"])
        indices = group.index.to_numpy()

        xy = group[
            ["pitch_x_m_corrected", "pitch_y_m_corrected"]
        ].to_numpy(dtype=float)

        times = group["time_seconds"].to_numpy(dtype=float)

        if len(group) < 2:
            continue

        dt = np.diff(times)
        distances = np.linalg.norm(np.diff(xy, axis=0), axis=1)

        valid = (
            np.isfinite(xy[:-1]).all(axis=1)
            & np.isfinite(xy[1:]).all(axis=1)
            & np.isfinite(dt)
            & (dt > 0)
            & (dt <= MAX_GAP_SECONDS)
        )

        speeds = np.full(len(dt), np.nan)
        accepted_distances = np.full(len(dt), np.nan)

        speeds[valid] = distances[valid] / dt[valid]
        accepted_distances[valid] = distances[valid]

        # A step's speed and distance belong to its ending observation.
        raw_speed.loc[indices[1:]] = speeds
        raw_distance.loc[indices[1:]] = accepted_distances
        step_dt.loc[indices[1:]] = dt

    return raw_speed, raw_distance, step_dt


def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT)

    required = {
        "canonical_id",
        "frame_id",
        "time_seconds",
        "pitch_x_m",
        "pitch_y_m",
        "pitch_x_m_corrected",
        "pitch_y_m_corrected",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.sort_values(
        ["canonical_id", "time_seconds", "frame_id"]
    ).reset_index(drop=True)

    # Calculate raw speeds from the candidate's corrected coordinates.
    raw_speed, raw_distance, step_dt = calculate_speeds(df)

    df["raw_step_speed_mps"] = raw_speed
    df["raw_step_distance_m"] = raw_distance
    df["step_dt_seconds"] = step_dt

    # Strict threshold: exactly 10 m/s is NOT replaced.
    flagged = (
        df["raw_step_speed_mps"].notna()
        & (df["raw_step_speed_mps"] > SPEED_LIMIT_MPS)
    )

    rng = np.random.default_rng(RANDOM_SEED)
    replacement_count = int(flagged.sum())

    replacements = rng.uniform(
        REPLACEMENT_MIN_MPS,
        REPLACEMENT_MAX_MPS,
        size=replacement_count,
    )

    # Report-facing value; raw speed remains unchanged.
    df["reported_speed_mps"] = df["raw_step_speed_mps"]
    df.loc[flagged, "reported_speed_mps"] = replacements

    df["speed_sanitized_flag"] = flagged

    df["speed_sanitization_reason"] = np.where(
        flagged,
        "display_only_replacement_raw_speed_over_10_mps",
        np.where(
            df["raw_step_speed_mps"].isna(),
            "speed_unavailable_or_invalid_time_gap",
            "unchanged",
        ),
    )

    # Display-only distance. Raw distance remains available separately.
    df["reported_step_distance_m"] = (
        df["reported_speed_mps"] * df["step_dt_seconds"]
    )

    output_csv = (
        OUTPUT_DIR / "football2_corrected_report_sanitized.csv"
    )
    audit_csv = OUTPUT_DIR / "speed_sanitization_audit.csv"
    summary_json = OUTPUT_DIR / "speed_sanitization_summary.json"

    # Save the report-facing dataset.
    df.to_csv(output_csv, index=False)

    # Save an explicit row-level audit.
    audit_columns = [
        "frame_id",
        "time_seconds",
        "canonical_id",
        "raw_step_speed_mps",
        "reported_speed_mps",
        "raw_step_distance_m",
        "reported_step_distance_m",
        "speed_sanitized_flag",
        "speed_sanitization_reason",
    ]

    for col in ("correction_applied", "correction_method",
                "correction_delta_m", "correction_confidence"):
        if col in df.columns:
            audit_columns.append(col)

    df[audit_columns].to_csv(audit_csv, index=False)

    reported = df["reported_speed_mps"].dropna()
    raw = df["raw_step_speed_mps"].dropna()
    replaced = df.loc[flagged, "reported_speed_mps"]

    summary = {
        "input_file": str(INPUT),
        "output_file": str(output_csv),
        "rows": int(len(df)),
        "players": int(df["canonical_id"].nunique()),
        "threshold_mps": SPEED_LIMIT_MPS,
        "threshold_rule": "strictly greater than 10 m/s",
        "replacement_range_mps": [
            REPLACEMENT_MIN_MPS,
            REPLACEMENT_MAX_MPS,
        ],
        "replacement_interval": "[6, 9)",
        "random_seed": RANDOM_SEED,
        "finite_raw_speeds": int(len(raw)),
        "sanitized_rows": replacement_count,
        "raw_max_speed_mps": (
            float(raw.max()) if len(raw) else None
        ),
        "reported_max_speed_mps": (
            float(reported.max()) if len(reported) else None
        ),
        "replacement_min_actual_mps": (
            float(replaced.min()) if len(replaced) else None
        ),
        "replacement_max_actual_mps": (
            float(replaced.max()) if len(replaced) else None
        ),
        "all_reported_speeds_at_most_10_mps": bool(
            reported.le(SPEED_LIMIT_MPS).all()
        ),
        "all_replacements_within_6_to_9_mps": bool(
            replaced.ge(REPLACEMENT_MIN_MPS).all()
            and replaced.lt(REPLACEMENT_MAX_MPS).all()
        ),
        "warning": (
            "Replaced speeds and corresponding reported distances "
            "are display-only substitutions, not measured player "
            "performance. Use raw speeds and distances for analysis, "
            "with trajectory quality checks."
        ),
    }

    summary_json.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("=== REPORT SPEED SANITIZATION COMPLETE ===")
    print(json.dumps(summary, indent=2))
    print("\nOutput files:")
    print(output_csv)
    print(audit_csv)
    print(summary_json)


if __name__ == "__main__":
    main()
