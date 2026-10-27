
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def calculate_player_metrics(
    tracks: pd.DataFrame,
    fps: float,
    max_interpolation_gap: int = 2,
) -> pd.DataFrame:
    """Calculate pixel-based movement metrics for each canonical player ID."""
    required = {
        "frame_id",
        "time_seconds",
        "canonical_id",
        "center_x",
        "center_y",
        "confidence",
    }
    missing = required - set(tracks.columns)
    if missing:
        raise ValueError(f"Player CSV is missing columns: {sorted(missing)}")
    if fps <= 0:
        raise ValueError("FPS must be greater than zero.")
    if tracks.empty:
        return pd.DataFrame()

    df = tracks.copy()
    numeric_cols = [
        "frame_id", "time_seconds", "center_x", "center_y", "confidence"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=["frame_id", "canonical_id", "center_x", "center_y"]
    )
    df = df.sort_values(["canonical_id", "frame_id"])

    rows = []

    for player_id, group in df.groupby("canonical_id", sort=True):
        # Multiple observations for one identity in a frame are ambiguous.
        # Keep the highest-confidence observation rather than double-counting.
        group = (
            group.sort_values("confidence", ascending=False)
            .drop_duplicates("frame_id", keep="first")
            .sort_values("frame_id")
        )

        frames = group["frame_id"].to_numpy(dtype=float)
        times = group["time_seconds"].to_numpy(dtype=float)
        xs = group["center_x"].to_numpy(dtype=float)
        ys = group["center_y"].to_numpy(dtype=float)
        confidence = group["confidence"].dropna()

        frame_gaps = np.diff(frames)
        dx = np.diff(xs)
        dy = np.diff(ys)
        displacement = np.hypot(dx, dy)

        # Avoid treating long gaps between observations as continuous movement.
        valid_steps = (
            (frame_gaps > 0)
            & (frame_gaps <= max_interpolation_gap + 1)
        )
        valid_displacement = displacement[valid_steps]
        valid_elapsed = frame_gaps[valid_steps] / fps

        speeds = (
            valid_displacement / valid_elapsed
            if len(valid_elapsed)
            else np.array([], dtype=float)
        )

        observed_frames = int(group["frame_id"].nunique())
        first_frame = int(frames[0])
        last_frame = int(frames[-1])
        span_seconds = (last_frame - first_frame) / fps
        observed_seconds = observed_frames / fps

        rows.append({
            "canonical_id": player_id,
            "first_frame": first_frame,
            "last_frame": last_frame,
            "observed_frames": observed_frames,
            "observed_time_seconds": observed_seconds,
            "track_span_seconds": span_seconds,
            "track_coverage_ratio": (
                observed_frames / (last_frame - first_frame + 1)
                if last_frame >= first_frame else 0.0
            ),
            "mean_detection_confidence": (
                float(confidence.mean()) if not confidence.empty else np.nan
            ),
            "median_detection_confidence": (
                float(confidence.median()) if not confidence.empty else np.nan
            ),
            "movement_distance_pixels": float(valid_displacement.sum()),
            "average_speed_pixels_per_second": (
                float(valid_displacement.sum() / valid_elapsed.sum())
                if len(valid_elapsed) and valid_elapsed.sum() > 0
                else 0.0
            ),
            "maximum_step_speed_pixels_per_second": (
                float(speeds.max()) if len(speeds) else 0.0
            ),
            "valid_movement_steps": int(valid_steps.sum()),
            "excluded_gap_steps": int((~valid_steps).sum()),
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            "movement_distance_pixels", ascending=False
        ).reset_index(drop=True)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Calculate per-player movement metrics from AthletiQ tracks."
    )
    parser.add_argument(
        "--input",
        default="outputs/joint_tracking/football2_player_tracks.csv",
        help="Player tracking CSV",
    )
    parser.add_argument(
        "--output",
        default="outputs/performance/football2_player_metrics.csv",
        help="Output metrics CSV",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Source video FPS",
    )
    parser.add_argument(
        "--max-interpolation-gap",
        type=int,
        default=2,
        help="Maximum number of missing frames tolerated between observations",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        raise FileNotFoundError(f"Tracking CSV not found: {input_path}")

    tracks = pd.read_csv(input_path)
    metrics = calculate_player_metrics(
        tracks,
        fps=args.fps,
        max_interpolation_gap=args.max_interpolation_gap,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path, index=False)

    print("Player metrics calculation complete.")
    print(f"Input rows: {len(tracks)}")
    print(f"Players summarized: {len(metrics)}")
    print(f"Output: {output_path}")

    if not metrics.empty:
        print("\nTop 10 players by observed pixel movement:")
        print(metrics.head(10).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
