
#!/usr/bin/env python3
"""Audit the impact of report-facing speed sanitization per player."""

from pathlib import Path
import pandas as pd

INPUT = Path(
    "outputs/physical_metrics/sanitized_reports/"
    "football2_stitched_physical_tracks_report_sanitized.csv"
)
OUTPUT_DIR = Path("outputs/physical_metrics/sanitized_reports")

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT}")

    df = pd.read_csv(INPUT)

    required = {
        "canonical_id",
        "raw_step_speed_mps",
        "reported_speed_mps",
        "raw_step_distance_m",
        "reported_step_distance_m",
        "speed_sanitized_flag",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["speed_sanitized_flag"] = (
        df["speed_sanitized_flag"].astype(str).str.lower().eq("true")
    )

    rows = []
    for player_id, group in df.groupby("canonical_id", dropna=False):
        raw_speed = pd.to_numeric(
            group["raw_step_speed_mps"], errors="coerce"
        )
        reported_speed = pd.to_numeric(
            group["reported_speed_mps"], errors="coerce"
        )
        raw_distance = pd.to_numeric(
            group["raw_step_distance_m"], errors="coerce"
        )
        reported_distance = pd.to_numeric(
            group["reported_step_distance_m"], errors="coerce"
        )

        raw_valid = raw_speed.dropna()
        report_valid = reported_speed.dropna()

        raw_distance_total = raw_distance.sum(min_count=1)
        report_distance_total = reported_distance.sum(min_count=1)

        rows.append({
            "canonical_id": player_id,
            "observations": len(group),
            "modified_speed_count": int(
                group["speed_sanitized_flag"].sum()
            ),
            "modified_speed_percentage": round(
                100 * group["speed_sanitized_flag"].mean(), 2
            ),
            "raw_max_speed_mps": raw_valid.max(),
            "reported_max_speed_mps": report_valid.max(),
            "raw_mean_speed_mps": raw_valid.mean(),
            "reported_mean_speed_mps": report_valid.mean(),
            "raw_step_distance_sum_m": raw_distance_total,
            "reported_step_distance_sum_m": report_distance_total,
            "distance_difference_m": (
                report_distance_total - raw_distance_total
                if pd.notna(report_distance_total)
                and pd.notna(raw_distance_total)
                else float("nan")
            ),
        })

    summary = pd.DataFrame(rows).sort_values(
        ["modified_speed_count", "canonical_id"],
        ascending=[False, True],
    )

    output_path = OUTPUT_DIR / "player_speed_sanitization_impact.csv"
    summary.to_csv(output_path, index=False)

    print("=== PLAYER SANITIZATION IMPACT ===")
    print("Players:", len(summary))
    print("Total modified rows:", int(summary["modified_speed_count"].sum()))
    print("\nMost affected players:")
    print(summary.head(15).round(3).to_string(index=False))
    print(f"\nFull audit saved to: {output_path}")

if __name__ == "__main__":
    main()
