from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BALL = ROOT / "outputs/joint_tracking/football2_ball_tracks.csv"
DEFAULT_PLAYERS = ROOT / "outputs/team_identification/football2_team_predictions.csv"
DEFAULT_OUT = ROOT / "outputs/ball_possession"

TEAMS = {"team_a", "team_b"}


def find_column(df, candidates, description, required=True):
    """Find a column using case-insensitive candidate names."""
    lookup = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    if required:
        raise ValueError(
            f"Could not identify {description}.\n"
            f"Tried: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )
    return None


def numeric(df, column):
    return pd.to_numeric(df[column], errors="coerce")


def load_ball_tracks(path):
    df = pd.read_csv(path)

    frame_col = find_column(
        df, ["frame_id", "frame", "frame_number"], "ball frame ID"
    )

    # Prefer explicit ball centers; otherwise derive centers from a bbox.
    bx = find_column(
        df,
        ["ball_x", "center_x", "cx", "x", "ball_center_x"],
        "ball X coordinate",
        required=False,
    )
    by = find_column(
        df,
        ["ball_y", "center_y", "cy", "y", "ball_center_y"],
        "ball Y coordinate",
        required=False,
    )

    if bx is None or by is None:
        x1 = find_column(df, ["x1", "xmin", "left"], "ball bbox x1", False)
        y1 = find_column(df, ["y1", "ymin", "top"], "ball bbox y1", False)
        x2 = find_column(df, ["x2", "xmax", "right"], "ball bbox x2", False)
        y2 = find_column(df, ["y2", "ymax", "bottom"], "ball bbox y2", False)

        if all(c is not None for c in [x1, y1, x2, y2]):
            df["_ball_x"] = (numeric(df, x1) + numeric(df, x2)) / 2
            df["_ball_y"] = (numeric(df, y1) + numeric(df, y2)) / 2
            bx, by = "_ball_x", "_ball_y"
        else:
            raise ValueError(
                "Could not identify ball coordinates or a complete ball bbox. "
                f"Columns found: {list(df.columns)}"
            )

    df["frame_id"] = numeric(df, frame_col)
    df["ball_x"] = numeric(df, bx)
    df["ball_y"] = numeric(df, by)

    confidence_col = find_column(
        df,
        ["ball_confidence", "confidence", "conf", "detection_confidence"],
        "ball confidence",
        required=False,
    )
    df["ball_confidence"] = (
        numeric(df, confidence_col).clip(0, 1)
        if confidence_col
        else np.nan
    )

    status_col = find_column(
        df, ["status", "state", "track_status", "ball_status"],
        "ball tracking status",
        required=False,
    )
    df["ball_status"] = (
        df[status_col].astype(str).str.lower().str.strip()
        if status_col
        else "unknown"
    )

    df = df.dropna(subset=["frame_id"]).copy()
    df["frame_id"] = df["frame_id"].astype(int)

    # If a frame has multiple ball rows, prefer rows with valid coordinates.
    df["_valid_xy"] = df["ball_x"].notna() & df["ball_y"].notna()
    df = df.sort_values(
        ["frame_id", "_valid_xy", "ball_confidence"],
        ascending=[True, False, False],
        na_position="last",
    )
    df = df.drop_duplicates("frame_id", keep="first")

    return df[
        ["frame_id", "ball_x", "ball_y", "ball_confidence", "ball_status"]
    ]


def load_player_tracks(path):
    df = pd.read_csv(path)

    frame_col = find_column(
        df, ["frame_id", "frame", "frame_number"], "player frame ID"
    )
    id_col = find_column(
        df, ["canonical_id", "player_id", "track_id", "id"],
        "player tracking ID"
    )
    x1 = find_column(df, ["x1", "xmin", "left"], "player bbox x1")
    x2 = find_column(df, ["x2", "xmax", "right"], "player bbox x2")
    y2 = find_column(df, ["y2", "ymax", "bottom"], "player bbox bottom")

    team_col = find_column(
        df,
        ["aggregated_team", "team_prediction", "team_label", "team"],
        "player team label",
    )
    team_conf_col = find_column(
        df,
        ["team_confidence", "team_probability", "team_score"],
        "team confidence",
        required=False,
    )

    df["frame_id"] = numeric(df, frame_col)
    df["canonical_id"] = numeric(df, id_col)
    df["player_x"] = (numeric(df, x1) + numeric(df, x2)) / 2
    df["player_y"] = numeric(df, y2)
    df["team"] = df[team_col].astype(str).str.lower().str.strip()

    df["team_confidence"] = (
        numeric(df, team_conf_col).clip(0, 1)
        if team_conf_col
        else np.nan
    )

    df = df.dropna(
        subset=["frame_id", "canonical_id", "player_x", "player_y"]
    ).copy()
    df["frame_id"] = df["frame_id"].astype(int)
    df["canonical_id"] = df["canonical_id"].astype(int)

    # Keep one row per player/frame.
    df = df.sort_values(
        ["frame_id", "canonical_id", "team_confidence"],
        ascending=[True, True, False],
        na_position="last",
    )
    df = df.drop_duplicates(["frame_id", "canonical_id"], keep="first")

    return df[
        [
            "frame_id", "canonical_id", "player_x", "player_y",
            "team", "team_confidence"
        ]
    ]


def get_ball_candidate(ball, players, max_distance, contested_margin):
    """Estimate the nearest plausible player and possession state."""
    base = {
        "nearest_player_id": None,
        "nearest_player_team": "unknown",
        "nearest_distance_px": np.nan,
        "opponent_distance_px": np.nan,
        "possession_state": "unknown",
        "possession_team": "unknown",
        "possession_confidence": 0.0,
    }

    if (
        pd.isna(ball["ball_x"])
        or pd.isna(ball["ball_y"])
        or players.empty
    ):
        return base

    # Lost/missing tracks should not be treated as a visible free ball.
    status = str(ball["ball_status"]).lower()
    if any(word in status for word in ["lost", "missing", "not_detected"]):
        return base

    p = players.copy()
    p["distance_px"] = np.hypot(
        p["player_x"] - float(ball["ball_x"]),
        p["player_y"] - float(ball["ball_y"]),
    )
    p = p.sort_values("distance_px").reset_index(drop=True)

    nearest = p.iloc[0]
    distance = float(nearest["distance_px"])
    team = nearest["team"] if nearest["team"] in TEAMS else "unknown"

    base.update(
        nearest_player_id=int(nearest["canonical_id"]),
        nearest_player_team=team,
        nearest_distance_px=round(distance, 3),
    )

    team_players = p[p["team"].isin(TEAMS)]
    if not team_players.empty:
        nearest_team_player = team_players.iloc[0]
        team = nearest_team_player["team"]
        distance = float(nearest_team_player["distance_px"])
        base.update(
            nearest_player_id=int(nearest_team_player["canonical_id"]),
            nearest_player_team=team,
            nearest_distance_px=round(distance, 3),
        )

    # Distance to the nearest player on the opposing team.
    if team in TEAMS:
        opponents = p[p["team"].isin(TEAMS) & (p["team"] != team)]
        if not opponents.empty:
            base["opponent_distance_px"] = round(
                float(opponents.iloc[0]["distance_px"]), 3
            )

    if distance > max_distance or team not in TEAMS:
        base["possession_state"] = "free_ball"
        return base

    # If an opposing player is almost equally close, avoid a confident claim.
    opponent_distance = base["opponent_distance_px"]
    if (
        pd.notna(opponent_distance)
        and opponent_distance <= max_distance
        and abs(float(opponent_distance) - distance) <= contested_margin
    ):
        base["possession_state"] = "contested"
        return base

    player_conf = nearest_team_player.get("team_confidence", np.nan) if not team_players.empty else np.nan
    if pd.isna(player_conf):
        player_conf = 0.5

    ball_conf = ball["ball_confidence"]
    if pd.isna(ball_conf):
        ball_conf = 0.5

    # Heuristic confidence, not a calibrated probability.
    proximity_score = max(0.0, 1.0 - distance / max_distance)
    confidence = (
        0.55 * proximity_score
        + 0.25 * float(player_conf)
        + 0.20 * float(ball_conf)
    )

    base["possession_state"] = "player_near_ball"
    base["possession_team"] = team
    base["possession_confidence"] = round(
        float(np.clip(confidence, 0, 1)), 4
    )
    return base


def stabilize_team_possession(df, min_run):
    """Suppress short team-label changes; do not smooth missing/contested frames."""
    result = df["possession_team"].tolist()
    states = df["possession_state"].tolist()

    i = 0
    while i < len(result):
        if result[i] not in TEAMS or states[i] != "player_near_ball":
            i += 1
            continue

        j = i + 1
        while (
            j < len(result)
            and result[j] == result[i]
            and states[j] == "player_near_ball"
            and int(df.iloc[j]["frame_id"]) == int(df.iloc[j - 1]["frame_id"]) + 1
        ):
            j += 1

        if j - i < min_run:
            # Keep the raw nearest-player estimate but mark it as uncertain.
            for k in range(i, j):
                df.at[df.index[k], "possession_state"] = "uncertain"
                df.at[df.index[k], "possession_team"] = "unknown"
                df.at[df.index[k], "possession_confidence"] = min(
                    float(df.iloc[k]["possession_confidence"]), 0.49
                )
                result[k] = "unknown"

        i = j

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Estimate ball possession from AthletiQ tracking outputs."
    )
    parser.add_argument("--ball", type=Path, default=DEFAULT_BALL)
    parser.add_argument("--players", type=Path, default=DEFAULT_PLAYERS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-distance", type=float, default=65.0)
    parser.add_argument("--contested-margin", type=float, default=15.0)
    parser.add_argument("--min-run", type=int, default=3)
    args = parser.parse_args()

    if not args.ball.exists():
        raise FileNotFoundError(f"Ball tracks not found: {args.ball}")
    if not args.players.exists():
        raise FileNotFoundError(f"Player/team predictions not found: {args.players}")
    if args.max_distance <= 0 or args.min_run < 1:
        raise ValueError("--max-distance must be positive and --min-run >= 1")

    args.output.mkdir(parents=True, exist_ok=True)

    ball = load_ball_tracks(args.ball)
    players = load_player_tracks(args.players)

    player_frames = {
        int(frame): group
        for frame, group in players.groupby("frame_id", sort=False)
    }

    rows = []
    for _, ball_row in ball.iterrows():
        frame = int(ball_row["frame_id"])
        frame_players = player_frames.get(frame, players.iloc[0:0])

        candidate = get_ball_candidate(
            ball_row,
            frame_players,
            max_distance=args.max_distance,
            contested_margin=args.contested_margin,
        )

        rows.append({
            "frame_id": frame,
            "time_seconds": np.nan,
            "ball_x": ball_row["ball_x"],
            "ball_y": ball_row["ball_y"],
            "ball_confidence": ball_row["ball_confidence"],
            "ball_status": ball_row["ball_status"],
            **candidate,
        })

    result = pd.DataFrame(rows).sort_values("frame_id").reset_index(drop=True)
    result["time_seconds"] = result["frame_id"] / 25.0
    result = stabilize_team_possession(result, args.min_run)

    frame_path = args.output / "football2_possession_by_frame.csv"
    result.to_csv(frame_path, index=False)

    # Summaries count only frames with an explicit team possession estimate.
    possessed = result[result["possession_team"].isin(TEAMS)]
    total_frames = len(result)

    team_rows = []
    for team in ["team_a", "team_b"]:
        count = int((possessed["possession_team"] == team).sum())
        team_rows.append({
            "team": team,
            "possession_frames": count,
            "share_of_all_ball_frames_pct": round(
                100 * count / total_frames, 2
            ) if total_frames else 0.0,
            "share_of_assigned_possession_pct": round(
                100 * count / len(possessed), 2
            ) if len(possessed) else 0.0,
        })
    pd.DataFrame(team_rows).to_csv(
        args.output / "football2_team_possession_summary.csv", index=False
    )

    player_rows = []
    for player_id, group in result[
        result["nearest_player_id"].notna()
        & result["possession_state"].eq("player_near_ball")
    ].groupby("nearest_player_id"):
        player_rows.append({
            "canonical_id": int(player_id),
            "team": group["possession_team"].mode().iloc[0],
            "estimated_possession_frames": int(len(group)),
            "estimated_possession_seconds": round(len(group) / 25.0, 3),
            "mean_ball_distance_px": round(
                float(group["nearest_distance_px"].mean()), 3
            ),
            "mean_heuristic_confidence": round(
                float(group["possession_confidence"].mean()), 4
            ),
        })

    pd.DataFrame(
        player_rows,
        columns=[
            "canonical_id", "team", "estimated_possession_frames",
            "estimated_possession_seconds", "mean_ball_distance_px",
            "mean_heuristic_confidence"
        ],
    ).sort_values(
        "estimated_possession_frames", ascending=False
    ).to_csv(args.output / "football2_player_possession_summary.csv", index=False)

    state_counts = result["possession_state"].value_counts().to_dict()
    summary = {
        "schema_version": "1.0",
        "method": "nearest-player distance with team labels and short-run filtering",
        "coordinate_space": "image_pixels",
        "physical_distances_validated": False,
        "possession_confidence_is_calibrated_probability": False,
        "video_fps_assumed": 25.0,
        "ball_input": str(args.ball),
        "player_input": str(args.players),
        "frames_in_ball_track_file": int(total_frames),
        "frames_with_valid_ball_coordinates": int(
            result[["ball_x", "ball_y"]].notna().all(axis=1).sum()
        ),
        "frames_with_estimated_team_possession": int(len(possessed)),
        "state_counts": {str(k): int(v) for k, v in state_counts.items()},
        "team_possession": team_rows,
        "parameters": {
            "max_distance_px": args.max_distance,
            "contested_margin_px": args.contested_margin,
            "minimum_consecutive_frames": args.min_run,
        },
        "limitations": [
            "Nearest-player proximity does not prove ball control.",
            "Distances are in image pixels and are affected by perspective and camera motion.",
            "Temporary tracking IDs may split one real player into multiple IDs.",
            "Team labels are preliminary predictions.",
            "Frames with missing/lost ball coordinates are unknown, not free-ball frames.",
            "Confidence is a heuristic score, not a calibrated probability.",
            "The time conversion assumes 25 FPS.",
        ],
    }

    with open(args.output / "football2_possession_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nBall possession estimation completed.")
    print(f"Ball track rows: {len(ball)}")
    print(f"Player observations: {len(players)}")
    print(f"Frames with ball coordinates: {summary['frames_with_valid_ball_coordinates']}")
    print(f"Frames with estimated team possession: {len(possessed)}")
    print("\nPossession states:")
    print(result["possession_state"].value_counts(dropna=False).to_string())
    print("\nTeam possession summary:")
    print(pd.DataFrame(team_rows).to_string(index=False))
    print(f"\nOutputs: {args.output}")


if __name__ == "__main__":
    main()
