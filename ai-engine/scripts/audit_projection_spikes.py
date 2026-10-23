
#!/usr/bin/env python3
"""Audit player-speed spikes against image motion and calibration disagreement.

Read-only diagnostic. Does not modify physical tracks or calibration files.
"""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


def require_columns(df, columns, path):
    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tracks",
        type=Path,
        default=Path(
            "outputs/physical_metrics/football2_stitched_physical_tracks.csv"
        ),
    )
    parser.add_argument(
        "--homography",
        type=Path,
        default=Path(
            "outputs/physical_metrics/"
            "full_homography_comparison_visible/"
            "full_homography_comparison.csv"
        ),
    )
    parser.add_argument(
        "--calibration-motion",
        type=Path,
        default=Path(
            "outputs/physical_metrics/camera_motion_diagnostic/"
            "calibration_motion.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/physical_metrics/projection_spike_audit"),
    )
    parser.add_argument("--start", type=float, default=13.0)
    parser.add_argument("--end", type=float, default=17.2)
    parser.add_argument("--speed-threshold", type=float, default=12.0)
    args = parser.parse_args()

    for path in (args.tracks, args.homography, args.calibration_motion):
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    tracks = pd.read_csv(args.tracks)
    homography = pd.read_csv(args.homography)
    calibration = pd.read_csv(args.calibration_motion)

    require_columns(
        tracks,
        [
            "time_seconds", "frame_id", "canonical_id",
            "foot_x_px", "foot_y_px", "pitch_x_m", "pitch_y_m",
            "step_distance_m", "step_speed_mps", "step_dt_seconds",
            "speed_outlier_flag", "physical_coordinate_valid",
            "calibration_rep_err", "calibration_gap_seconds",
        ],
        args.tracks,
    )
    require_columns(
        homography,
        [
            "time0", "time1", "inlier_ratio",
            "grid_error_median_px", "grid_error_p90_px",
            "grid_error_max_px", "grid_points_compared",
            "review_flag",
        ],
        args.homography,
    )
    require_columns(
        calibration,
        ["time_seconds"],
        args.calibration_motion,
    )

    tracks = tracks.sort_values(
        ["canonical_id", "time_seconds", "frame_id"]
    ).copy()

    # Compute image-space step distance from the previous observation of
    # the same canonical player. This is diagnostic, not a tracker metric.
    grouped = tracks.groupby("canonical_id", sort=False)

    tracks["prev_time_seconds"] = grouped["time_seconds"].shift(1)
    tracks["prev_foot_x_px"] = grouped["foot_x_px"].shift(1)
    tracks["prev_foot_y_px"] = grouped["foot_y_px"].shift(1)

    tracks["image_step_px"] = np.hypot(
        tracks["foot_x_px"] - tracks["prev_foot_x_px"],
        tracks["foot_y_px"] - tracks["prev_foot_y_px"],
    )

    # Diagnostic amplification: projected metres per pixel of foot motion.
    # This is not a true local Jacobian and is undefined for tiny image steps.
    tracks["projected_m_per_image_px"] = np.where(
        tracks["image_step_px"] > 1.0,
        tracks["step_distance_m"] / tracks["image_step_px"],
        np.nan,
    )

    # Assign each observation to its corresponding video-frame interval.
    # The last interval is inclusive of its endpoint.
    h = homography.sort_values("time0").copy()
    intervals = h[["time0", "time1"]].to_numpy(dtype=float)

    assigned = np.full(len(tracks), -1, dtype=int)
    times = tracks["time_seconds"].to_numpy(dtype=float)

    for i, (start, end) in enumerate(intervals):
        mask = (times >= start - 1e-8) & (times < end - 1e-8)
        if i == len(intervals) - 1:
            mask = (times >= start - 1e-8) & (times <= end + 1e-8)
        assigned[mask] = i

    tracks["homography_interval_index"] = assigned

    h = h.reset_index(drop=True).reset_index(names="homography_interval_index")

    diagnostic_cols = [
        "homography_interval_index",
        "time0", "time1", "inlier_ratio",
        "grid_error_median_px", "grid_error_p90_px",
        "grid_error_max_px", "grid_points_compared", "review_flag",
    ]

    joined = tracks.merge(
        h[diagnostic_cols],
        on="homography_interval_index",
        how="left",
        validate="many_to_one",
    )

    # Attach nearest calibration-motion sample for context only.
    calibration_cols = [
        c for c in [
            "time_seconds", "rep_err", "pan_degrees", "tilt_degrees",
            "roll_degrees", "fx", "delta_fx_percent",
            "delta_camera_position_m", "delta_rotation_frobenius",
        ]
        if c in calibration.columns
    ]

    calibration_small = (
        calibration[calibration_cols]
        .dropna(subset=["time_seconds"])
        .sort_values("time_seconds")
        .rename(columns={"time_seconds": "nearest_calibration_time"})
    )

    joined["nearest_calibration_time"] = np.nan

    if not calibration_small.empty:
        joined = pd.merge_asof(
            joined.sort_values("time_seconds"),
            calibration_small,
            left_on="time_seconds",
            right_on="nearest_calibration_time",
            direction="nearest",
            tolerance=0.15,
            suffixes=("", "_cal"),
        )

    window = joined[
        joined["time_seconds"].between(args.start, args.end)
    ].copy()

    if window.empty:
        raise RuntimeError(
            f"No track rows found between {args.start} and {args.end}s"
        )

    window["computed_speed_outlier"] = (
        window["step_speed_mps"] > args.speed_threshold
    )

    # Per-time summary across players. Counts are based on the rows present
    # at that timestamp; players may have missing observations.
    time_summary = (
        window.groupby(["frame_id", "time_seconds"], as_index=False)
        .agg(
            players_observed=("canonical_id", "nunique"),
            speed_outlier_players=("computed_speed_outlier", "sum"),
            max_speed_mps=("step_speed_mps", "max"),
            p90_speed_mps=(
                "step_speed_mps",
                lambda s: s.quantile(0.90),
            ),
            median_speed_mps=("step_speed_mps", "median"),
            median_image_step_px=("image_step_px", "median"),
            median_projected_m_per_px=(
                "projected_m_per_image_px", "median"
            ),
            median_calibration_rep_err=("calibration_rep_err", "median"),
            median_grid_error_p90_px=("grid_error_p90_px", "median"),
            max_grid_error_p90_px=("grid_error_p90_px", "max"),
            reviewed_homography_intervals=("review_flag", "sum"),
        )
        .sort_values("time_seconds")
    )

    # Per-player summary makes it easier to spot players with persistent
    # spikes rather than a one-frame anomaly.
    player_summary = (
        window.groupby("canonical_id", as_index=False)
        .agg(
            observations=("frame_id", "size"),
            outlier_steps=("computed_speed_outlier", "sum"),
            max_speed_mps=("step_speed_mps", "max"),
            p95_speed_mps=("step_speed_mps", lambda s: s.quantile(0.95)),
            median_image_step_px=("image_step_px", "median"),
            median_projected_m_per_px=(
                "projected_m_per_image_px", "median"
            ),
            median_grid_error_p90_px=("grid_error_p90_px", "median"),
        )
        .sort_values(
            ["outlier_steps", "max_speed_mps"],
            ascending=False,
        )
    )

    # Retain only high-speed rows for a compact detailed report.
    outliers = window[
        window["computed_speed_outlier"]
    ].copy()

    detail_cols = [
        "frame_id", "time_seconds", "canonical_id",
        "foot_x_px", "foot_y_px", "pitch_x_m", "pitch_y_m",
        "image_step_px", "step_distance_m", "step_dt_seconds",
        "step_speed_mps", "speed_outlier_flag",
        "projected_m_per_image_px",
        "calibration_gap_seconds", "calibration_rep_err",
        "grid_error_p90_px", "grid_error_max_px",
        "grid_points_compared", "inlier_ratio", "review_flag",
    ]
    detail_cols = [c for c in detail_cols if c in outliers.columns]
    outliers = outliers[detail_cols].sort_values(
        ["time_seconds", "step_speed_mps"],
        ascending=[True, False],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    detail_path = args.output_dir / "speed_outlier_details.csv"
    time_path = args.output_dir / "time_summary.csv"
    player_path = args.output_dir / "player_summary.csv"
    report_path = args.output_dir / "projection_spike_report.json"

    outliers.to_csv(detail_path, index=False)
    time_summary.to_csv(time_path, index=False)
    player_summary.to_csv(player_path, index=False)

    report = {
        "window_seconds": [args.start, args.end],
        "speed_threshold_mps": args.speed_threshold,
        "track_rows_in_window": int(len(window)),
        "distinct_players_in_window": int(window["canonical_id"].nunique()),
        "computed_speed_outlier_rows": int(
            window["computed_speed_outlier"].sum()
        ),
        "rows_flagged_by_original_pipeline": int(
            window["speed_outlier_flag"].fillna(False).astype(bool).sum()
        ),
        "median_grid_error_p90_px": float(
            window["grid_error_p90_px"].median()
        ),
        "rows_missing_homography_match": int(
            window["grid_error_p90_px"].isna().sum()
        ),
        "outputs": {
            "speed_outlier_details": str(detail_path),
            "time_summary": str(time_path),
            "player_summary": str(player_path),
        },
        "interpretation_warning": (
            "Temporal correlation does not establish that calibration caused "
            "the speed spikes. Image-space tracking errors and real player "
            "motion can also contribute. The metres-per-pixel ratio is a "
            "diagnostic ratio, not a local projection derivative."
        ),
    }

    report_path.write_text(json.dumps(report, indent=2))

    print("\n=== Projection spike audit ===")
    print(json.dumps(report, indent=2))

    print("\n=== Per-time summary ===")
    print(time_summary.round(4).to_string(index=False))

    print("\n=== Players with most speed outliers ===")
    print(player_summary.head(15).round(4).to_string(index=False))

    print(f"\nDetailed outliers: {detail_path}")
    print(f"Time summary:      {time_path}")
    print(f"Player summary:    {player_path}")
    print(f"JSON report:       {report_path}")


if __name__ == "__main__":
    main()
