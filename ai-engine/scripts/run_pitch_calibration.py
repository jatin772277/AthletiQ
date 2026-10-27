#!/usr/bin/env python3
"""Fit and validate an AthletiQ pitch homography from a landmark CSV."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ai-engine"))

from core.spatial.pitch_calibration import calibrate_homography


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--landmarks",
        default="outputs/calibration/football2_landmarks_template.csv",
    )
    parser.add_argument(
        "--output",
        default="outputs/calibration/football2_pitch_calibration.json",
    )
    parser.add_argument("--max-mean-error-m", type=float, default=1.5)
    parser.add_argument("--max-point-error-m", type=float, default=3.0)
    args = parser.parse_args()

    csv_path = Path(args.landmarks)
    output_path = Path(args.output)

    if not csv_path.exists():
        raise SystemExit(f"Landmark CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {
        "role", "landmark_name", "frame_id",
        "image_x_px", "image_y_px", "pitch_x_m", "pitch_y_m",
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing CSV columns: {sorted(missing)}")

    if df.empty:
        raise SystemExit(
            "No landmarks supplied. Add verified fit and validation "
            "landmarks before running calibration."
        )

    df["role"] = df["role"].astype(str).str.strip().str.lower()
    if not df["role"].isin(["fit", "validation"]).all():
        raise SystemExit("The role column must contain only fit or validation.")

    numeric = [
        "frame_id", "image_x_px", "image_y_px", "pitch_x_m", "pitch_y_m"
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[numeric].isna().any().any():
        raise SystemExit("Numeric landmark fields contain missing/invalid values.")

    fit = df[df["role"] == "fit"]
    validation = df[df["role"] == "validation"]

    if len(fit) < 4:
        raise SystemExit("At least four fit landmarks are required.")
    if len(validation) < 1:
        raise SystemExit("At least one independent validation landmark is required.")

    # A single homography is valid only when all selected landmarks belong
    # to the same stable camera view.
    frame_ids = sorted(df["frame_id"].astype(int).unique().tolist())
    if len(frame_ids) > 1:
        print(
            "WARNING: landmarks span multiple frames. Ensure these frames "
            "share the same camera view and geometry."
        )

    try:
        result = calibrate_homography(
            fit[["image_x_px", "image_y_px"]].to_numpy(),
            fit[["pitch_x_m", "pitch_y_m"]].to_numpy(),
            validation[["image_x_px", "image_y_px"]].to_numpy(),
            validation[["pitch_x_m", "pitch_y_m"]].to_numpy(),
            max_mean_validation_error_m=args.max_mean_error_m,
            max_validation_error_m=args.max_point_error_m,
        )
    except (ValueError, TypeError) as exc:
        raise SystemExit(f"Calibration rejected: {exc}") from exc

    report = {
        "project": "AthletiQ",
        "landmarks_csv": str(csv_path),
        "fit_landmark_count": len(fit),
        "validation_landmark_count": len(validation),
        "frame_ids": frame_ids,
        "homography_image_to_pitch": result.homography.tolist(),
        "fit_errors_m": result.fit_errors_m,
        "validation_errors_m": result.validation_errors_m,
        "mean_validation_error_m": result.mean_validation_error_m,
        "max_validation_error_m": result.max_validation_error_m,
        "validated": result.validated,
        "physical_metrics_enabled": False,
        "reason": result.reason,
        "warning": (
            "A single homography requires a stable camera view and ground-plane "
            "points. Validate landmark geometry and camera movement before "
            "enabling physical metrics."
        ),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("========== ATHLETIQ PITCH CALIBRATION ==========")
    print(f"Fit landmarks:       {len(fit)}")
    print(f"Validation landmarks:{len(validation)}")
    print(f"Mean validation err: {result.mean_validation_error_m:.3f} m")
    print(f"Max validation err:  {result.max_validation_error_m:.3f} m")
    print(f"Validated:           {result.validated}")
    print(f"Report:              {output_path}")
    print("Physical metrics:    DISABLED pending camera/ground-plane review")
    print("================================================")

    if not result.validated:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
