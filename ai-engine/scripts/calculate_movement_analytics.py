
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# ATHLETIQ MOVEMENT ANALYTICS CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = ROOT / "outputs/joint_tracking/football2_player_tracks.csv"
OUTPUT_DIR = ROOT / "outputs/performance"

OUTPUT_CSV = OUTPUT_DIR / "football2_movement_analytics.csv"
TRAJECTORY_CSV = OUTPUT_DIR / "football2_smoothed_trajectories.csv"

FPS = 25.0

# Maximum allowed missing frames between consecutive detections.
MAX_GAP_FRAMES = 2

# Coordinate and speed smoothing windows.
COORDINATE_SMOOTHING_WINDOW = 5
SPEED_SMOOTHING_WINDOW = 5

# Robust pixel-space outlier filter.
MIN_OUTLIER_SPEED_PX_S = 150.0
ROBUST_MAD_MULTIPLIER = 6.0

# Data-quality thresholds.
MIN_VALID_STEPS = 5
MIN_VALID_MOVEMENT_SECONDS = 1.0
LOW_COVERAGE_THRESHOLD = 0.50
LOW_COVERAGE_MOVEMENT_SECONDS = 5.0

# Experimental sprint/activity threshold.
SPRINT_PERCENTILE = 75
MIN_SPRINT_DURATION_SECONDS = 0.4
MIN_SPRINT_SPEED_SAMPLES = 5


# ============================================================
# LOAD AND VALIDATE INPUT
# ============================================================

def load_tracking_data():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Tracking CSV not found: {INPUT_CSV}"
        )

    df = pd.read_csv(INPUT_CSV)

    required_columns = {
        "frame_id",
        "time_seconds",
        "canonical_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "confidence",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    numeric_columns = [
        "frame_id",
        "time_seconds",
        "canonical_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "confidence",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    df = df.dropna(
        subset=[
            "frame_id",
            "canonical_id",
            "x1",
            "y1",
            "x2",
            "y2",
        ]
    ).copy()

    # Reject invalid bounding boxes.
    df = df[
        (df["x2"] > df["x1"])
        & (df["y2"] > df["y1"])
    ].copy()

    # Approximate ground-contact point: bottom-centre of bounding box.
    df["foot_x"] = (df["x1"] + df["x2"]) / 2.0
    df["foot_y"] = df["y2"]

    # Resolve duplicate player detections in the same frame.
    df = (
        df.sort_values(
            "confidence",
            ascending=False,
            na_position="last",
        )
        .drop_duplicates(
            subset=["canonical_id", "frame_id"],
            keep="first",
        )
        .sort_values(["canonical_id", "frame_id"])
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError("No valid tracking rows remain after cleaning.")

    return df


# ============================================================
# SEGMENTATION
# ============================================================

def split_segments(player_df):
    """
    Split trajectories when gaps exceed the allowed limit.
    Never interpolate missing detections.
    """
    player_df = player_df.sort_values("frame_id").copy()

    if player_df.empty:
        return []

    frame_gap = player_df["frame_id"].diff()

    starts_segment = (
        frame_gap.isna()
        | (frame_gap > MAX_GAP_FRAMES + 1)
        | (frame_gap <= 0)
    )

    player_df["segment_id"] = (
        starts_segment.cumsum().astype(int)
    )

    return [
        segment.copy()
        for _, segment in player_df.groupby(
            "segment_id", sort=False
        )
    ]


# ============================================================
# COORDINATE SMOOTHING
# ============================================================

def smooth_coordinates(segment):
    """
    Median smoothing reduces coordinate jitter.
    Smoothing is confined to one continuous segment.
    """
    segment = segment.sort_values("frame_id").copy()

    segment["smooth_x"] = (
        segment["foot_x"]
        .rolling(
            window=COORDINATE_SMOOTHING_WINDOW,
            center=True,
            min_periods=1,
        )
        .median()
    )

    segment["smooth_y"] = (
        segment["foot_y"]
        .rolling(
            window=COORDINATE_SMOOTHING_WINDOW,
            center=True,
            min_periods=1,
        )
        .median()
    )

    return segment


# ============================================================
# MOVEMENT STEPS AND OUTLIER FILTERING
# ============================================================

def calculate_segment_steps(segment):
    segment = smooth_coordinates(segment)

    frame_gap = segment["frame_id"].diff()
    elapsed = frame_gap / FPS

    dx = segment["smooth_x"].diff()
    dy = segment["smooth_y"].diff()

    distance = np.hypot(dx, dy)
    speed = distance / elapsed

    candidate = (
        frame_gap.gt(0)
        & frame_gap.le(MAX_GAP_FRAMES + 1)
        & elapsed.gt(0)
        & speed.notna()
        & np.isfinite(speed)
    )

    candidate_speeds = speed.loc[candidate]

    if len(candidate_speeds) >= MIN_VALID_STEPS:
        median_speed = candidate_speeds.median()

        mad = (
            candidate_speeds - median_speed
        ).abs().median()

        robust_limit = (
            median_speed
            + ROBUST_MAD_MULTIPLIER * 1.4826 * mad
        )

        outlier_limit = max(
            MIN_OUTLIER_SPEED_PX_S,
            robust_limit,
        )
    else:
        outlier_limit = MIN_OUTLIER_SPEED_PX_S

    outlier = candidate & speed.gt(outlier_limit)
    accepted = candidate & ~outlier

    segment["frame_gap"] = frame_gap
    segment["elapsed_seconds"] = elapsed

    segment["step_distance_pixels"] = distance.where(accepted)
    segment["step_speed_pixels_per_second"] = speed.where(accepted)

    segment["valid_step"] = accepted
    segment["outlier_step"] = outlier

    # Smooth the accepted speed observations before acceleration.
    raw_speed = segment["step_speed_pixels_per_second"]

    segment["smoothed_speed_pixels_per_second"] = (
        raw_speed
        .rolling(
            window=SPEED_SMOOTHING_WINDOW,
            center=True,
            min_periods=1,
        )
        .median()
        .where(accepted)
    )

    # Acceleration only across adjacent, accepted, one-frame steps.
    previous_frame = segment["frame_id"].shift(1)
    previous_valid = accepted.shift(1, fill_value=False)

    consecutive_valid = (
        accepted
        & previous_valid
        & (segment["frame_id"] - previous_frame).eq(1)
    )

    previous_smoothed_speed = (
        segment["smoothed_speed_pixels_per_second"].shift(1)
    )

    acceleration = (
        segment["smoothed_speed_pixels_per_second"]
        - previous_smoothed_speed
    ) * FPS

    segment[
        "step_acceleration_pixels_per_second_squared"
    ] = acceleration.where(consecutive_valid)

    return segment


# ============================================================
# SPRINT INTERVALS (EXPERIMENTAL PIXEL-SPACE ACTIVITY)
# ============================================================

def find_sprint_intervals(segment, threshold):
    if not np.isfinite(threshold):
        return []

    intervals = []
    current = []

    for row in segment.itertuples():
        speed = row.smoothed_speed_pixels_per_second

        qualifies = (
            bool(row.valid_step)
            and pd.notna(speed)
            and np.isfinite(speed)
            and speed >= threshold
        )

        if qualifies:
            current.append(row)
        elif current:
            intervals.append(current)
            current = []

    if current:
        intervals.append(current)

    accepted_intervals = []

    for interval in intervals:
        duration = sum(
            float(row.elapsed_seconds)
            for row in interval
            if pd.notna(row.elapsed_seconds)
        )

        if duration >= MIN_SPRINT_DURATION_SECONDS:
            accepted_intervals.append(interval)

    return accepted_intervals


# ============================================================
# PLAYER DATA-QUALITY CLASSIFICATION
# ============================================================

def classify_data_quality(
    valid_steps,
    valid_time_seconds,
    coverage_ratio,
):
    if (
        valid_steps < MIN_VALID_STEPS
        or valid_time_seconds < MIN_VALID_MOVEMENT_SECONDS
    ):
        return "INSUFFICIENT_DATA"

    if (
        coverage_ratio < LOW_COVERAGE_THRESHOLD
        or valid_time_seconds < LOW_COVERAGE_MOVEMENT_SECONDS
    ):
        return "LOW_COVERAGE"

    return "OK"


# ============================================================
# PLAYER SUMMARY
# ============================================================

def summarize_player(player_df, canonical_id):
    segments = split_segments(player_df)

    if not segments:
        return None, []

    processed_segments = []

    for segment in segments:
        result = calculate_segment_steps(segment)
        result["canonical_id"] = canonical_id
        processed_segments.append(result)

    player = pd.concat(
        processed_segments,
        ignore_index=True,
    )

    valid = player.loc[player["valid_step"]].copy()

    valid_steps = int(len(valid))
    valid_time = float(valid["elapsed_seconds"].sum())

    distances = valid["step_distance_pixels"].dropna()
    raw_speeds = valid["step_speed_pixels_per_second"].dropna()
    smooth_speeds = (
        valid["smoothed_speed_pixels_per_second"].dropna()
    )

    distance_pixels = (
        float(distances.sum())
        if valid_steps > 0
        else np.nan
    )

    average_speed = (
        distance_pixels / valid_time
        if valid_time > 0 and np.isfinite(distance_pixels)
        else np.nan
    )

    # Coverage compares observed detections with the frame span.
    first_frame = int(player["frame_id"].min())
    last_frame = int(player["frame_id"].max())

    frame_span = last_frame - first_frame + 1

    coverage_ratio = (
        float(len(player) / frame_span)
        if frame_span > 0
        else 0.0
    )

    data_quality = classify_data_quality(
        valid_steps=valid_steps,
        valid_time_seconds=valid_time,
        coverage_ratio=coverage_ratio,
    )

    # Require enough accepted movement before summarizing acceleration.
    acceleration_series = player[
        "step_acceleration_pixels_per_second_squared"
    ].dropna()

    if len(acceleration_series) >= MIN_VALID_STEPS:
        positive_acceleration = acceleration_series.clip(lower=0)
        negative_acceleration = -acceleration_series.clip(upper=0)

        max_acceleration = float(positive_acceleration.max())
        max_deceleration = float(negative_acceleration.max())
    else:
        max_acceleration = np.nan
        max_deceleration = np.nan

    max_step_speed = (
        float(raw_speeds.max())
        if len(raw_speeds)
        else np.nan
    )

    median_step_speed = (
        float(raw_speeds.median())
        if len(raw_speeds)
        else np.nan
    )

    # Sprint analysis is unavailable when too few speed samples exist.
    if len(smooth_speeds) >= MIN_SPRINT_SPEED_SAMPLES:
        sprint_threshold = float(
            smooth_speeds.quantile(
                SPRINT_PERCENTILE / 100.0
            )
        )

        sprint_count = 0
        sprint_distance = 0.0
        sprint_duration = 0.0

        for _, segment in player.groupby("segment_id"):
            intervals = find_sprint_intervals(
                segment, sprint_threshold
            )

            for interval in intervals:
                sprint_count += 1

                for row in interval:
                    if pd.notna(row.step_distance_pixels):
                        sprint_distance += max(
                            0.0, float(row.step_distance_pixels)
                        )

                    if pd.notna(row.elapsed_seconds):
                        sprint_duration += float(row.elapsed_seconds)
    else:
        sprint_threshold = np.nan
        sprint_count = np.nan
        sprint_distance = np.nan
        sprint_duration = np.nan

    # Insufficient tracks do not receive invented zero-valued metrics.
    if valid_steps == 0:
        distance_pixels = np.nan
        average_speed = np.nan
        max_step_speed = np.nan
        median_step_speed = np.nan

    summary = {
        "canonical_id": int(canonical_id),
        "first_frame": first_frame,
        "last_frame": last_frame,
        "observed_rows": int(len(player)),
        "track_span_seconds": (last_frame - first_frame) / FPS,
        "continuous_segments": int(
            player["segment_id"].nunique()
        ),
        "coverage_ratio": coverage_ratio,
        "valid_movement_steps": valid_steps,
        "excluded_outlier_steps": int(
            player["outlier_step"].sum()
        ),
        "distance_pixels": distance_pixels,
        "valid_movement_time_seconds": valid_time,
        "average_speed_pixels_per_second": average_speed,
        "maximum_step_speed_pixels_per_second": max_step_speed,
        "median_step_speed_pixels_per_second": median_step_speed,
        "maximum_positive_acceleration_pixels_per_second_squared": (
            max_acceleration
        ),
        "maximum_deceleration_magnitude_pixels_per_second_squared": (
            max_deceleration
        ),
        "experimental_sprint_threshold_pixels_per_second": (
            sprint_threshold
        ),
        "experimental_sprint_count": sprint_count,
        "experimental_sprint_distance_pixels": sprint_distance,
        "experimental_sprint_duration_seconds": sprint_duration,
        "data_quality_status": data_quality,
        "measurement_space": "pixels",
        "physical_metrics_validated": False,
    }

    return summary, processed_segments


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    tracks = load_tracking_data()

    summaries = []
    all_segments = []

    for canonical_id, player_df in tracks.groupby(
        "canonical_id",
        sort=True,
    ):
        summary, segments = summarize_player(
            player_df,
            canonical_id,
        )

        if summary is not None:
            summaries.append(summary)
            all_segments.extend(segments)

    if not summaries:
        raise RuntimeError(
            "No player trajectories could be processed."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_df = (
        pd.DataFrame(summaries)
        .sort_values("canonical_id")
        .reset_index(drop=True)
    )

    trajectory_df = pd.concat(
        all_segments,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # CONSISTENCY CHECKS
    # --------------------------------------------------------

    assert summary_df["canonical_id"].is_unique
    assert summary_df["valid_movement_steps"].ge(0).all()
    assert summary_df["coverage_ratio"].between(0, 1).all()
    assert (
        summary_df["measurement_space"] == "pixels"
    ).all()
    assert not summary_df["physical_metrics_validated"].any()
    assert not trajectory_df.duplicated(
        ["canonical_id", "frame_id"]
    ).any()

    allowed_statuses = {
        "OK",
        "LOW_COVERAGE",
        "INSUFFICIENT_DATA",
    }

    assert set(
        summary_df["data_quality_status"].unique()
    ).issubset(allowed_statuses)

    # Write outputs after checks pass.
    summary_df.to_csv(OUTPUT_CSV, index=False)
    trajectory_df.to_csv(TRAJECTORY_CSV, index=False)

    print("=" * 68)
    print("ATHLETIQ PIXEL-BASED MOVEMENT ANALYTICS")
    print("=" * 68)

    print(f"Input rows: {len(tracks)}")
    print(f"Unique players: {summary_df['canonical_id'].nunique()}")
    print(f"Summary shape: {summary_df.shape}")
    print(f"Trajectory rows: {len(trajectory_df)}")

    print(
        "Accepted movement steps:",
        int(trajectory_df["valid_step"].sum()),
    )
    print(
        "Excluded outlier steps:",
        int(trajectory_df["outlier_step"].sum()),
    )

    print("\nData-quality status:")
    print(
        summary_df["data_quality_status"]
        .value_counts()
        .reindex(
            ["OK", "LOW_COVERAGE", "INSUFFICIENT_DATA"],
            fill_value=0,
        )
        .to_string()
    )

    print("\nTop 10 players by pixel distance:")
    print(
        summary_df.nlargest(10, "distance_pixels")[
            [
                "canonical_id",
                "distance_pixels",
                "valid_movement_time_seconds",
                "average_speed_pixels_per_second",
                "maximum_step_speed_pixels_per_second",
                "data_quality_status",
            ]
        ].to_string(index=False)
    )

    print(f"\nSummary CSV: {OUTPUT_CSV}")
    print(f"Trajectory CSV: {TRAJECTORY_CSV}")
    print("\nValidation: PASSED")
    print("Measurements: PIXEL SPACE ONLY")
    print("Physical metrics: NOT VALIDATED")


if __name__ == "__main__":
    main()
