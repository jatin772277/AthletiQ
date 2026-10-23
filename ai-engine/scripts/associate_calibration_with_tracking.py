
from pathlib import Path
import json
import pandas as pd

# ------------------------------------------------------------
# AthletiQ paths
# ------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[2]

TRACKING_FILE = (
    PROJECT_DIR
    / "outputs"
    / "team_identification"
    / "football2_team_predictions.csv"
)

CALIBRATION_FILE = (
    PROJECT_DIR
    / "outputs"
    / "calibration"
    / "pnlcalib"
    / "calibration_summary.csv"
)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "calibration" / "association"

OUTPUT_TRACKING = (
    OUTPUT_DIR / "football2_tracking_with_calibration_candidates.csv"
)
OUTPUT_AUDIT = OUTPUT_DIR / "calibration_association_audit.json"
OUTPUT_SUMMARY = OUTPUT_DIR / "calibration_association_summary.csv"

# Calibration samples were generated every 2 seconds.
# Nearest-sample association is allowed only within 1 second.
MAX_TIME_GAP_SECONDS = 1.0


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not TRACKING_FILE.is_file():
        raise FileNotFoundError(
            f"Tracking predictions not found: {TRACKING_FILE}"
        )

    if not CALIBRATION_FILE.is_file():
        raise FileNotFoundError(
            f"Calibration summary not found: {CALIBRATION_FILE}"
        )

    tracking = pd.read_csv(TRACKING_FILE)
    calibration = pd.read_csv(CALIBRATION_FILE)

    print("Tracking file:", TRACKING_FILE)
    print("Calibration file:", CALIBRATION_FILE)
    print("Tracking rows:", len(tracking))
    print("Calibration rows:", len(calibration))

    # --------------------------------------------------------
    # 1. Validate required columns
    # --------------------------------------------------------
    required_tracking = {"frame_id", "time_seconds"}
    required_calibration = {
        "frame_id",
        "time_seconds",
        "calibration_returned",
        "rep_err",
        "status",
    }

    missing_tracking = required_tracking - set(tracking.columns)
    missing_calibration = required_calibration - set(calibration.columns)

    if missing_tracking:
        raise ValueError(
            f"Tracking file is missing columns: {sorted(missing_tracking)}"
        )

    if missing_calibration:
        raise ValueError(
            "Calibration summary is missing columns: "
            f"{sorted(missing_calibration)}"
        )

    # --------------------------------------------------------
    # 2. Normalize types without changing source files
    # --------------------------------------------------------
    tracking["frame_id"] = pd.to_numeric(
        tracking["frame_id"], errors="coerce"
    )
    tracking["time_seconds"] = pd.to_numeric(
        tracking["time_seconds"], errors="coerce"
    )

    calibration["frame_id"] = pd.to_numeric(
        calibration["frame_id"], errors="coerce"
    )
    calibration["time_seconds"] = pd.to_numeric(
        calibration["time_seconds"], errors="coerce"
    )
    calibration["rep_err"] = pd.to_numeric(
        calibration["rep_err"], errors="coerce"
    )

    tracking = tracking.dropna(
        subset=["frame_id", "time_seconds"]
    ).copy()

    calibration = calibration.dropna(
        subset=["frame_id", "time_seconds"]
    ).copy()

    # Normalize possible string representations of booleans.
    calibration["calibration_returned"] = (
        calibration["calibration_returned"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

    if calibration.empty:
        raise ValueError("No usable calibration records remain.")

    if calibration["time_seconds"].duplicated().any():
        raise ValueError(
            "Calibration timestamps are duplicated. "
            "Resolve duplicates before association."
        )

    # Prefix calibration columns to avoid overwriting tracking fields.
    rename_map = {
        col: f"cal_{col}"
        for col in calibration.columns
        if col != "time_seconds"
    }
    calibration = calibration.rename(columns=rename_map)
    calibration = calibration.rename(
        columns={"time_seconds": "cal_time_seconds"}
    )

    # --------------------------------------------------------
    # 3. Associate each tracking frame with its nearest sample
    # --------------------------------------------------------
    # This is timestamp association only, not geometric
    # interpolation or proof of calibration accuracy.
    tracking_sorted = tracking.sort_values("time_seconds").copy()
    calibration_sorted = calibration.sort_values(
        "cal_time_seconds"
    ).copy()

    associated = pd.merge_asof(
        tracking_sorted,
        calibration_sorted,
        left_on="time_seconds",
        right_on="cal_time_seconds",
        direction="nearest",
        tolerance=MAX_TIME_GAP_SECONDS,
    )

    associated["calibration_time_gap_seconds"] = (
        associated["time_seconds"] - associated["cal_time_seconds"]
    ).abs()

    associated["calibration_associated"] = (
        associated["cal_frame_id"].notna()
    )

    associated["calibration_candidate_available"] = (
        associated["calibration_associated"]
        & associated["cal_calibration_returned"].fillna(False).astype(bool)
    )

    # Crucial safeguard: current overlays have not passed independent
    # geometric validation. Do not use them for physical metrics yet.
    associated["calibration_validated"] = False
    associated["real_world_metrics_enabled"] = False

    associated["calibration_association_status"] = "no_nearby_sample"

    associated.loc[
        associated["calibration_associated"],
        "calibration_association_status",
    ] = "sample_associated_unvalidated"

    associated.loc[
        associated["calibration_associated"]
        & ~associated["calibration_candidate_available"],
        "calibration_association_status",
    ] = "sample_associated_but_no_parameters"

    # --------------------------------------------------------
    # 4. Save the enriched tracking data
    # --------------------------------------------------------
    associated = associated.sort_values(
        ["frame_id", "time_seconds"]
    ).reset_index(drop=True)

    associated.to_csv(OUTPUT_TRACKING, index=False)

    # Per-frame audit: one row per distinct tracking frame.
    audit_columns = [
        "frame_id",
        "time_seconds",
        "cal_frame_id",
        "cal_time_seconds",
        "calibration_time_gap_seconds",
        "cal_calibration_returned",
        "cal_rep_err",
        "cal_mode",
        "cal_status",
        "calibration_association_status",
        "calibration_validated",
        "real_world_metrics_enabled",
    ]

    available_audit_columns = [
        col for col in audit_columns if col in associated.columns
    ]

    frame_audit = (
        associated[available_audit_columns]
        .drop_duplicates(subset=["frame_id"])
        .sort_values("frame_id")
    )
    frame_audit.to_csv(OUTPUT_SUMMARY, index=False)

    # --------------------------------------------------------
    # 5. Write a machine-readable audit report
    # --------------------------------------------------------
    matched = associated["calibration_associated"]
    candidate = associated["calibration_candidate_available"]

    audit = {
        "tracking_file": str(TRACKING_FILE),
        "calibration_file": str(CALIBRATION_FILE),
        "output_file": str(OUTPUT_TRACKING),
        "max_time_gap_seconds": MAX_TIME_GAP_SECONDS,
        "tracking_rows": int(len(associated)),
        "distinct_tracking_frames": int(
            associated["frame_id"].nunique()
        ),
        "calibration_samples": int(len(calibration)),
        "rows_with_associated_sample": int(matched.sum()),
        "rows_with_candidate_parameters": int(candidate.sum()),
        "rows_without_nearby_sample": int((~matched).sum()),
        "validated_calibrations": 0,
        "real_world_metrics_enabled": False,
        "interpretation": (
            "Nearest-timestamp association only. Camera parameters and "
            "overlays remain unvalidated candidates. Do not use these "
            "associations to convert player coordinates into metres or "
            "calculate physical speed until independent geometric "
            "validation has passed."
        ),
    }

    OUTPUT_AUDIT.write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    print("\n=== ASSOCIATION COMPLETE ===")
    print(json.dumps(audit, indent=2))
    print("\nOutput tracking:", OUTPUT_TRACKING)
    print("Per-frame summary:", OUTPUT_SUMMARY)
    print("Audit report:", OUTPUT_AUDIT)


if __name__ == "__main__":
    main()
