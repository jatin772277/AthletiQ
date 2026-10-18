import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def analyze_ball_proximity(
    players,
    ball,
    proximity_threshold=45.0,
    fps=25.0,
    min_duration=0.4,
):
    required_players = {
        "frame_id", "canonical_id", "center_x", "center_y", "confidence"
    }
    required_ball = {
        "frame_id", "x", "y", "confidence", "state"
    }

    if not required_players.issubset(players.columns):
        raise ValueError(
            f"Player CSV missing columns: {required_players - set(players.columns)}"
        )

    if not required_ball.issubset(ball.columns):
        raise ValueError(
            f"Ball CSV missing columns: {required_ball - set(ball.columns)}"
        )

    results = []

    # Find the nearest tracked player independently in each frame.
    for ball_row in ball.itertuples(index=False):
        frame_id = ball_row.frame_id
        state = ball_row.state

        frame_players = players[players["frame_id"] == frame_id]

        result = {
            "frame_id": frame_id,
            "time_seconds": getattr(ball_row, "time_seconds", frame_id / fps),
            "ball_x": ball_row.x,
            "ball_y": ball_row.y,
            "ball_confidence": ball_row.confidence,
            "ball_state": state,
            "nearest_player_id": np.nan,
            "nearest_distance_px": np.nan,
            "player_confidence": np.nan,
            "within_threshold": False,
            "possession_candidate": False,
        }

        # Never treat a lost ball position as reliable.
        if (
            state != "lost"
            and pd.notna(ball_row.x)
            and pd.notna(ball_row.y)
            and not frame_players.empty
        ):
            valid_players = frame_players.dropna(
                subset=["center_x", "center_y", "canonical_id"]
            )

            if not valid_players.empty:
                dx = valid_players["center_x"].to_numpy() - ball_row.x
                dy = valid_players["center_y"].to_numpy() - ball_row.y
                distances = np.sqrt(dx * dx + dy * dy)
                nearest_index = int(np.argmin(distances))
                nearest = valid_players.iloc[nearest_index]
                distance = float(distances[nearest_index])

                within_threshold = distance <= proximity_threshold

                result.update({
                    "nearest_player_id": int(nearest["canonical_id"]),
                    "nearest_distance_px": distance,
                    "player_confidence": nearest["confidence"],
                    "within_threshold": within_threshold,
                    # Predicted ball positions are recorded but do not establish
                    # possession candidates in this initial conservative version.
                    "possession_candidate": (
                        state == "detected" and within_threshold
                    ),
                })

        results.append(result)

    frame_results = pd.DataFrame(results).sort_values("frame_id").reset_index(
        drop=True
    )

    # Build consecutive candidate intervals for the same player.
    candidates = frame_results[
        frame_results["possession_candidate"]
    ].copy()

    intervals = []
    if not candidates.empty:
        candidates["player_id"] = candidates["nearest_player_id"].astype(int)

        run_id = (
            candidates["player_id"].ne(candidates["player_id"].shift())
            | candidates["frame_id"].diff().ne(1)
        ).cumsum()

        for _, run in candidates.groupby(run_id):
            start_frame = int(run["frame_id"].iloc[0])
            end_frame = int(run["frame_id"].iloc[-1])
            duration = (end_frame - start_frame + 1) / fps

            if duration < min_duration:
                continue

            intervals.append({
                "player_id": int(run["player_id"].iloc[0]),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time_seconds": round(start_frame / fps, 3),
                "end_time_seconds": round(end_frame / fps, 3),
                "duration_seconds": round(duration, 3),
                "frames": len(run),
                "mean_ball_distance_px": round(
                    run["nearest_distance_px"].mean(), 2
                ),
                "mean_ball_confidence": round(
                    run["ball_confidence"].mean(), 3
                ),
            })

    intervals_df = pd.DataFrame(
        intervals,
        columns=[
            "player_id", "start_frame", "end_frame",
            "start_time_seconds", "end_time_seconds",
            "duration_seconds", "frames",
            "mean_ball_distance_px", "mean_ball_confidence",
        ],
    )

    return frame_results, intervals_df


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ball-to-player proximity and candidate possession."
    )
    parser.add_argument(
        "--players",
        default="outputs/joint_tracking/football2_player_tracks.csv",
    )
    parser.add_argument(
        "--ball",
        default="outputs/joint_tracking/football2_ball_tracks.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/performance",
    )
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--distance", type=float, default=45.0)
    parser.add_argument("--min-duration", type=float, default=0.4)
    args = parser.parse_args()

    players = pd.read_csv(args.players)
    ball = pd.read_csv(args.ball)

    frame_results, intervals = analyze_ball_proximity(
        players=players,
        ball=ball,
        proximity_threshold=args.distance,
        fps=args.fps,
        min_duration=args.min_duration,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_path = output_dir / "football2_ball_proximity.csv"
    intervals_path = output_dir / "football2_candidate_possessions.csv"

    frame_results.to_csv(frame_path, index=False)
    intervals.to_csv(intervals_path, index=False)

    print("\nBall proximity analysis complete.")
    print(f"Frames analyzed: {len(frame_results)}")
    print(f"Detected ball frames: {(frame_results['ball_state'] == 'detected').sum()}")
    print(f"Predicted ball frames: {(frame_results['ball_state'] == 'predicted').sum()}")
    print(f"Lost ball frames: {(frame_results['ball_state'] == 'lost').sum()}")
    print(f"Frames within {args.distance:.1f}px of a player: "
          f"{frame_results['within_threshold'].sum()}")
    print(f"Candidate possession frames: "
          f"{frame_results['possession_candidate'].sum()}")
    print(f"Candidate intervals (minimum {args.min_duration:.1f}s): "
          f"{len(intervals)}")
    print(f"\nFrame-level output: {frame_path}")
    print(f"Candidate intervals: {intervals_path}")

    if not intervals.empty:
        print("\nLongest candidate intervals:")
        print(
            intervals.sort_values("duration_seconds", ascending=False)
            .head(15)
            .to_string(index=False)
        )
    else:
        print(
            "\nNo intervals met the current threshold. "
            "Inspect the frame-level CSV before changing parameters."
        )


if __name__ == "__main__":
    main()
