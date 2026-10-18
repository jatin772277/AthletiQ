
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def analyze_ball_proximity(
    players: pd.DataFrame,
    ball: pd.DataFrame,
    proximity_threshold: float = 45.0,
    fps: float = 25.0,
    min_duration: float = 0.4,
):
    """
    Analyze ball-to-player distance using the bottom-center of each
    player's bounding box as an approximate foot position.

    Candidate possession is a proximity-based estimate, not confirmed
    possession. Only directly detected ball positions count as candidates.
    Predicted positions can be used for proximity analysis but are not
    considered possession evidence. Lost positions are excluded.
    """

    if fps <= 0:
        raise ValueError("FPS must be greater than zero.")

    if proximity_threshold < 0:
        raise ValueError("Distance threshold cannot be negative.")

    if min_duration < 0:
        raise ValueError("Minimum duration cannot be negative.")

    required_players = {
        "frame_id",
        "time_seconds",
        "canonical_id",
        "x1",
        "y2",
        "x2",
        "center_x",
        "center_y",
        "confidence",
    }

    required_ball = {
        "frame_id",
        "x",
        "y",
        "confidence",
        "state",
    }

    missing_players = required_players - set(players.columns)
    missing_ball = required_ball - set(ball.columns)

    if missing_players:
        raise ValueError(
            f"Player CSV is missing columns: {sorted(missing_players)}"
        )

    if missing_ball:
        raise ValueError(
            f"Ball CSV is missing columns: {sorted(missing_ball)}"
        )

    players = players.copy()
    ball = ball.copy()

    players["frame_id"] = pd.to_numeric(
        players["frame_id"], errors="coerce"
    )
    ball["frame_id"] = pd.to_numeric(
        ball["frame_id"], errors="coerce"
    )

    players = players.dropna(
        subset=["frame_id", "canonical_id"]
    )
    ball = ball.dropna(subset=["frame_id"])

    # Calculate the bottom-center point of each player bounding box.
    players["foot_x"] = (players["x1"] + players["x2"]) / 2.0
    players["foot_y"] = players["y2"]

    # Prepare player detections indexed by frame for faster lookup.
    players_by_frame = {
        frame_id: group
        for frame_id, group in players.groupby("frame_id", sort=False)
    }

    results = []

    for b in ball.itertuples(index=False):
        frame_id = int(b.frame_id)
        state = str(b.state).strip().lower()

        ball_x = getattr(b, "x")
        ball_y = getattr(b, "y")
        ball_confidence = getattr(b, "confidence")

        ball_time = getattr(
            b, "time_seconds", frame_id / fps
        )

        result = {
            "frame_id": frame_id,
            "time_seconds": ball_time,
            "ball_x": ball_x,
            "ball_y": ball_y,
            "ball_confidence": ball_confidence,
            "ball_state": state,
            "nearest_player_id": np.nan,
            "nearest_player_foot_x": np.nan,
            "nearest_player_foot_y": np.nan,
            "nearest_distance_px": np.nan,
            "player_confidence": np.nan,
            "within_threshold": False,
            "possession_candidate": False,
        }

        # A lost ball must not be used for distance or possession evidence.
        if state == "lost":
            results.append(result)
            continue

        if pd.isna(ball_x) or pd.isna(ball_y):
            results.append(result)
            continue

        frame_players = players_by_frame.get(frame_id)

        if frame_players is None or frame_players.empty:
            results.append(result)
            continue

        valid_players = frame_players.dropna(
            subset=[
                "canonical_id",
                "foot_x",
                "foot_y",
                "confidence",
            ]
        )

        if valid_players.empty:
            results.append(result)
            continue

        # Ball-to-player distance using bottom-center coordinates.
        dx = valid_players["foot_x"].to_numpy() - float(ball_x)
        dy = valid_players["foot_y"].to_numpy() - float(ball_y)

        distances = np.hypot(dx, dy)

        nearest_index = int(np.argmin(distances))
        nearest = valid_players.iloc[nearest_index]
        nearest_distance = float(distances[nearest_index])

        within_threshold = nearest_distance <= proximity_threshold

        result.update({
            "nearest_player_id": int(nearest["canonical_id"]),
            "nearest_player_foot_x": float(nearest["foot_x"]),
            "nearest_player_foot_y": float(nearest["foot_y"]),
            "nearest_distance_px": nearest_distance,
            "player_confidence": float(nearest["confidence"]),
            "within_threshold": bool(within_threshold),

            # Only detected ball positions can create candidate possession.
            "possession_candidate": bool(
                state == "detected" and within_threshold
            ),
        })

        results.append(result)

    frame_results = (
        pd.DataFrame(results)
        .sort_values("frame_id")
        .reset_index(drop=True)
    )

    intervals = []

    candidates = frame_results[
        frame_results["possession_candidate"]
    ].copy()

    if not candidates.empty:
        candidates["player_id"] = (
            candidates["nearest_player_id"].astype(int)
        )

        # A new interval starts if the player changes or a frame is skipped.
        new_interval = (
            candidates["player_id"].ne(
                candidates["player_id"].shift()
            )
            | candidates["frame_id"].diff().ne(1)
        )

        candidates["interval_id"] = new_interval.cumsum()

        for _, run in candidates.groupby("interval_id", sort=False):
            start_frame = int(run["frame_id"].iloc[0])
            end_frame = int(run["frame_id"].iloc[-1])

            frame_count = len(run)
            duration = frame_count / fps

            if duration < min_duration:
                continue

            intervals.append({
                "player_id": int(run["player_id"].iloc[0]),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time_seconds": round(
                    float(run["time_seconds"].iloc[0]), 3
                ),
                "end_time_seconds": round(
                    float(run["time_seconds"].iloc[-1]), 3
                ),
                "duration_seconds": round(duration, 3),
                "frames": frame_count,
                "mean_ball_distance_px": round(
                    float(run["nearest_distance_px"].mean()), 2
                ),
                "median_ball_distance_px": round(
                    float(run["nearest_distance_px"].median()), 2
                ),
                "mean_ball_confidence": round(
                    float(run["ball_confidence"].mean()), 3
                ),
                "mean_player_confidence": round(
                    float(run["player_confidence"].mean()), 3
                ),
                "evidence_type": "detected_ball_proximity",
            })

    interval_columns = [
        "player_id",
        "start_frame",
        "end_frame",
        "start_time_seconds",
        "end_time_seconds",
        "duration_seconds",
        "frames",
        "mean_ball_distance_px",
        "median_ball_distance_px",
        "mean_ball_confidence",
        "mean_player_confidence",
        "evidence_type",
    ]

    intervals_df = pd.DataFrame(
        intervals,
        columns=interval_columns,
    )

    return frame_results, intervals_df


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze ball proximity and candidate possession "
            "using player bottom-center coordinates."
        )
    )

    parser.add_argument(
        "--players",
        default="outputs/joint_tracking/football2_player_tracks.csv",
        help="Input player tracking CSV.",
    )

    parser.add_argument(
        "--ball",
        default="outputs/joint_tracking/football2_ball_tracks.csv",
        help="Input ball tracking CSV.",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/performance",
        help="Directory for generated CSV files.",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="Video frames per second.",
    )

    parser.add_argument(
        "--distance",
        type=float,
        default=45.0,
        help="Maximum ball-to-player distance in pixels.",
    )

    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.4,
        help="Minimum consecutive candidate duration in seconds.",
    )

    args = parser.parse_args()

    players_path = Path(args.players)
    ball_path = Path(args.ball)

    if not players_path.exists():
        raise FileNotFoundError(
            f"Player tracking file not found: {players_path}"
        )

    if not ball_path.exists():
        raise FileNotFoundError(
            f"Ball tracking file not found: {ball_path}"
        )

    players = pd.read_csv(players_path)
    ball = pd.read_csv(ball_path)

    frame_results, intervals = analyze_ball_proximity(
        players=players,
        ball=ball,
        proximity_threshold=args.distance,
        fps=args.fps,
        min_duration=args.min_duration,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_path = output_dir / "football2_ball_proximity_v2.csv"
    intervals_path = (
        output_dir / "football2_candidate_possessions_v2.csv"
    )

    frame_results.to_csv(frame_path, index=False)
    intervals.to_csv(intervals_path, index=False)

    state_counts = (
        frame_results["ball_state"]
        .value_counts(dropna=False)
    )

    print("\n" + "=" * 55)
    print("ATHLETIQ - FOOT-BASED BALL PROXIMITY ANALYSIS")
    print("=" * 55)

    print(f"Player tracking rows: {len(players)}")
    print(f"Player IDs: {players['canonical_id'].nunique()}")
    print(f"Frames analyzed: {len(frame_results)}")

    print("\nBall states:")
    print(state_counts.to_string())

    print(
        f"\nFrames within {args.distance:.1f}px of a player: "
        f"{int(frame_results['within_threshold'].sum())}"
    )

    print(
        "Candidate-possession frames: "
        f"{int(frame_results['possession_candidate'].sum())}"
    )

    print(
        f"Candidate intervals (minimum {args.min_duration:.2f}s): "
        f"{len(intervals)}"
    )

    print(f"\nFrame-level CSV: {frame_path}")
    print(f"Candidate intervals CSV: {intervals_path}")

    if not intervals.empty:
        print("\nLongest candidate intervals:")

        print(
            intervals.sort_values(
                "duration_seconds",
                ascending=False,
            )
            .head(15)
            .to_string(index=False)
        )
    else:
        print(
            "\nNo intervals met the minimum duration. "
            "Inspect the frame-level output before tuning parameters."
        )

    print(
        "\nInterpretation: these are proximity-based candidates, "
        "not verified ball possession."
    )


if __name__ == "__main__":
    main()
