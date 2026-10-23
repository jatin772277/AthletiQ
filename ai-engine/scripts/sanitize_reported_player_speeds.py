
#!/usr/bin/env python3
"""
Create sanitized, report-facing player speeds without modifying raw physical metrics.

Rules:
- Speeds <= 12 m/s are preserved.
- Speeds > 12 m/s are replaced with reproducible random values in [6, 9] m/s.
- Original speed and distance columns remain unchanged.
- Separate report-facing distance and speed columns are generated.
- Every modified row is explicitly flagged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sanitize extreme player speeds for report display."
    )
    parser.add_argument(
        "--input",
        default="outputs/physical_metrics/football2_stitched_physical_tracks.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/physical_metrics/sanitized_reports",
    )
    parser.add_argument("--threshold", type=float, default=12.0)
    parser.add_argument("--min-speed", type=float, default=6.0)
    parser.add_argument("--max-speed", type=float, default=9.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.threshold <= 0:
        raise ValueError("Threshold must be positive.")
    if args.min_speed < 0 or args.max_speed < args.min_speed:
        raise ValueError("Invalid replacement speed range.")

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)

    required = {"step_speed_mps", "step_distance_m", "step_dt_seconds"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Keep original metrics intact and add separate report-facing columns.
    df["raw_step_speed_mps"] = df["step_speed_mps"]
    df["raw_step_distance_m"] = df["step_distance_m"]

    df["reported_speed_mps"] = df["step_speed_mps"]
    df["speed_sanitized_flag"] = False
    df["speed_sanitization_reason"] = ""

    speed = pd.to_numeric(df["step_speed_mps"], errors="coerce")
    dt = pd.to_numeric(df["step_dt_seconds"], errors="coerce")

    # Only finite speeds strictly above the threshold are replaced.
    mask = speed.notna() & np.isfinite(speed) & (speed > args.threshold)

    rng = np.random.default_rng(args.seed)
    count = int(mask.sum())

    if count:
        replacements = rng.uniform(
            args.min_speed,
            args.max_speed,
            size=count,
        )
        df.loc[mask, "reported_speed_mps"] = replacements
        df.loc[mask, "speed_sanitized_flag"] = True
        df.loc[mask, "speed_sanitization_reason"] = (
            f"Raw speed exceeded {args.threshold:g} m/s; "
            "replaced for report display"
        )

    # Report-facing step distance is consistent with reported speed.
    # Preserve missing/invalid time intervals as missing distances.
    df["reported_step_distance_m"] = df["reported_speed_mps"] * dt
    valid_distance = (
        df["reported_speed_mps"].notna()
        & dt.notna()
        & np.isfinite(dt)
        & (dt > 0)
    )
    df.loc[~valid_distance, "reported_step_distance_m"] = np.nan

    # Keep a separate audit of all altered rows.
    audit_columns = [
        c
        for c in [
            "frame_id",
            "time_seconds",
            "canonical_id",
            "step_speed_mps",
            "reported_speed_mps",
            "step_distance_m",
            "reported_step_distance_m",
            "step_dt_seconds",
            "speed_sanitized_flag",
            "speed_sanitization_reason",
        ]
        if c in df.columns
    ]

    output_csv = output_dir / f"{input_path.stem}_report_sanitized.csv"
    audit_csv = output_dir / "speed_sanitization_audit.csv"
    report_json = output_dir / "speed_sanitization_summary.json"

    df.to_csv(output_csv, index=False)
    df.loc[mask, audit_columns].to_csv(audit_csv, index=False)

    raw_max = float(speed.max()) if speed.notna().any() else None
    reported = pd.to_numeric(df["reported_speed_mps"], errors="coerce")
    report_max = float(reported.max()) if reported.notna().any() else None

    summary = {
        "input_file": str(input_path),
        "output_file": str(output_csv),
        "audit_file": str(audit_csv),
        "threshold_mps": args.threshold,
        "replacement_range_mps": [args.min_speed, args.max_speed],
        "random_seed": args.seed,
        "rows_total": int(len(df)),
        "rows_sanitized": count,
        "raw_max_speed_mps": raw_max,
        "reported_max_speed_mps": report_max,
        "original_speed_and_distance_preserved": True,
        "warning": (
            "Randomized speeds are report-facing substitutions, not "
            "measured or reconstructed player velocities. Do not use "
            "them for scientific evaluation or training ground-truth data."
        ),
    }

    report_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== SPEED SANITIZATION COMPLETE ===")
    print(f"Input:             {input_path}")
    print(f"Sanitized output:  {output_csv}")
    print(f"Changed rows:      {count}")
    print(f"Original max:      {raw_max}")
    print(f"Reported max:      {report_max}")
    print(f"Audit CSV:         {audit_csv}")
    print(f"Summary JSON:      {report_json}")
    print("\nOriginal metrics were preserved.")


if __name__ == "__main__":
    main()
