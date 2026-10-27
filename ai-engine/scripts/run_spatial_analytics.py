import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_VIDEO = ROOT / "datasets/football/raw/videos/football2.mp4"
DEFAULT_INPUT = ROOT / "outputs/team_identification/football2_team_predictions.csv"
DEFAULT_OUTPUT = ROOT / "outputs/spatial_analytics"


def save_heatmap(points, width, height, output_path, title, bins=40):
    """Save an image-coordinate density heatmap."""
    fig, ax = plt.subplots(figsize=(12, 6.75))

    if len(points):
        heat, x_edges, y_edges = np.histogram2d(
            points["foot_x"],
            points["foot_y"],
            bins=[bins, max(1, round(bins * height / width))],
            range=[[0, width], [0, height]],
        )
        heat = heat.T
        mesh = ax.imshow(
            heat,
            extent=[0, width, height, 0],
            aspect="auto",
            interpolation="bilinear",
        )
        fig.colorbar(mesh, ax=ax, label="Observed position count")
    else:
        ax.text(
            0.5, 0.5, "No valid positions",
            transform=ax.transAxes, ha="center", va="center"
        )

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_xlabel("Image X (pixels)")
    ax.set_ylabel("Image Y (pixels)")
    ax.set_title(title)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def nearest_distances(frame):
    """
    Compute nearest same-team and opposing-team distances for each player.
    Distances are in image pixels, not physical units.
    """
    results = []

    for _, group in frame.groupby("frame_id", sort=False):
        group = group.drop_duplicates("canonical_id").copy()
        if len(group) < 2:
            continue

        xy = group[["foot_x", "foot_y"]].to_numpy(dtype=float)
        labels = group["team_label"].fillna("unknown").to_numpy()
        ids = group["canonical_id"].to_numpy()

        delta = xy[:, None, :] - xy[None, :, :]
        distances = np.sqrt(np.sum(delta * delta, axis=2))
        np.fill_diagonal(distances, np.inf)

        for i in range(len(group)):
            same = (labels == labels[i]) & (labels[i] != "unknown")
            opponent = (
                (labels != labels[i])
                & (labels != "unknown")
                & (labels[i] != "unknown")
            )
            same[i] = False

            nearest_same_idx = (
                np.argmin(np.where(same, distances[i], np.inf))
                if same.any() else None
            )
            nearest_opp_idx = (
                np.argmin(np.where(opponent, distances[i], np.inf))
                if opponent.any() else None
            )

            results.append({
                "frame_id": group.iloc[i]["frame_id"],
                "time_seconds": group.iloc[i].get("time_seconds", np.nan),
                "canonical_id": ids[i],
                "team_label": labels[i],
                "nearest_teammate_id": (
                    ids[nearest_same_idx]
                    if nearest_same_idx is not None else np.nan
                ),
                "nearest_teammate_distance_px": (
                    float(distances[i, nearest_same_idx])
                    if nearest_same_idx is not None else np.nan
                ),
                "nearest_opponent_id": (
                    ids[nearest_opp_idx]
                    if nearest_opp_idx is not None else np.nan
                ),
                "nearest_opponent_distance_px": (
                    float(distances[i, nearest_opp_idx])
                    if nearest_opp_idx is not None else np.nan
                ),
            })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Generate image-space spatial analytics for AthletiQ."
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--grid-cols", type=int, default=5)
    parser.add_argument("--grid-rows", type=int, default=3)
    parser.add_argument("--min-player-observations", type=int, default=5)
    args = parser.parse_args()

    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.input.exists():
        raise FileNotFoundError(f"Team prediction CSV not found: {args.input}")
    if args.grid_cols < 1 or args.grid_rows < 1:
        raise ValueError("Grid rows and columns must be positive.")

    output = args.output_dir
    plots = output / "plots"
    player_heatmaps_dir = plots / "player_heatmaps"
    player_heatmaps_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    required = {
        "frame_id", "canonical_id", "x1", "y1", "x2", "y2"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {sorted(missing)}")

    if "aggregated_team" in df.columns:
        df["team_label"] = df["aggregated_team"]
    elif "team_prediction" in df.columns:
        df["team_label"] = df["team_prediction"]
    else:
        raise ValueError(
            "Expected aggregated_team or team_prediction in input CSV."
        )

    if "time_seconds" not in df.columns:
        df["time_seconds"] = np.nan

    for col in ["frame_id", "canonical_id", "x1", "y1", "x2", "y2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if width <= 0 or height <= 0:
        raise RuntimeError("Could not read video dimensions.")

    # Use the estimated foot point (bottom-center of each player box).
    df["foot_x"] = (df["x1"] + df["x2"]) / 2.0
    df["foot_y"] = df["y2"]

    if df["time_seconds"].isna().all() and fps > 0:
        df["time_seconds"] = df["frame_id"] / fps

    valid = (
        df["frame_id"].notna()
        & df["canonical_id"].notna()
        & df["foot_x"].between(0, width)
        & df["foot_y"].between(0, height)
    )
    points = df.loc[valid].copy()

    # One position per temporary ID per frame.
    points = (
        points.sort_values("team_confidence", ascending=False, na_position="last")
        .drop_duplicates(["frame_id", "canonical_id"])
        .reset_index(drop=True)
    )

    # Relative image grid: not calibrated football pitch zones.
    points["zone_col"] = np.minimum(
        (points["foot_x"] / width * args.grid_cols).astype(int),
        args.grid_cols - 1,
    )
    points["zone_row"] = np.minimum(
        (points["foot_y"] / height * args.grid_rows).astype(int),
        args.grid_rows - 1,
    )
    points["zone_id"] = (
        "R" + (points["zone_row"] + 1).astype(str)
        + "_C" + (points["zone_col"] + 1).astype(str)
    )

    # Occupancy measured in observed frames and seconds.
    zone = (
        points.groupby(["canonical_id", "team_label", "zone_id",
                        "zone_row", "zone_col"], dropna=False)
        .agg(
            observed_frames=("frame_id", "nunique"),
            first_frame=("frame_id", "min"),
            last_frame=("frame_id", "max"),
            mean_x_px=("foot_x", "mean"),
            mean_y_px=("foot_y", "mean"),
        )
        .reset_index()
    )
    zone["observed_seconds"] = (
        zone["observed_frames"] / fps if fps > 0 else np.nan
    )
    zone.to_csv(output / "football2_player_zone_occupancy.csv", index=False)

    team_zone = (
        points[points["team_label"].isin(["team_a", "team_b"])]
        .groupby(["team_label", "zone_id", "zone_row", "zone_col"])
        .agg(
            player_observations=("canonical_id", "size"),
            distinct_temporary_ids=("canonical_id", "nunique"),
            distinct_frames=("frame_id", "nunique"),
        )
        .reset_index()
    )
    team_zone.to_csv(output / "football2_team_zone_occupancy.csv", index=False)

    # Team centroids and spatial spread per frame.
    team_rows = []
    for (frame_id, team), group in points[
        points["team_label"].isin(["team_a", "team_b"])
    ].groupby(["frame_id", "team_label"]):
        xy = group[["foot_x", "foot_y"]].to_numpy(dtype=float)
        centroid_x = float(xy[:, 0].mean())
        centroid_y = float(xy[:, 1].mean())
        radial = np.sqrt(
            (xy[:, 0] - centroid_x) ** 2
            + (xy[:, 1] - centroid_y) ** 2
        )

        team_rows.append({
            "frame_id": frame_id,
            "time_seconds": group["time_seconds"].mean(),
            "team_label": team,
            "visible_temporary_ids": int(group["canonical_id"].nunique()),
            "centroid_x_px": centroid_x,
            "centroid_y_px": centroid_y,
            "spread_rms_px": float(np.sqrt(np.mean(radial ** 2))),
            "spread_mean_px": float(np.mean(radial)),
            "spread_max_px": float(np.max(radial)),
        })

    team_metrics = pd.DataFrame(team_rows)
    team_metrics.to_csv(
        output / "football2_team_spatial_metrics.csv", index=False
    )

    # Per-player nearest teammate and opponent.
    proximity = nearest_distances(points)
    proximity.to_csv(output / "football2_player_proximity.csv", index=False)

    # Team heatmaps.
    for team in ["team_a", "team_b"]:
        save_heatmap(
            points[points["team_label"] == team],
            width, height,
            plots / f"{team}_heatmap.png",
            f"{team.replace('_', ' ').title()} positional density (image space)",
        )

    save_heatmap(
        points, width, height,
        plots / "all_players_heatmap.png",
        "All tracked players: positional density (image space)",
    )

    # Individual player heatmaps for IDs with enough observations.
    player_counts = points.groupby("canonical_id").size()
    for player_id, count in player_counts.items():
        if count < args.min_player_observations:
            continue
        player_points = points[points["canonical_id"] == player_id]
        team = player_points["team_label"].mode()
        team_name = team.iloc[0] if len(team) else "unknown"
        save_heatmap(
            player_points, width, height,
            player_heatmaps_dir / f"player_{int(player_id):03d}.png",
            f"Temporary ID {int(player_id)} | {team_name} | {count} observations",
        )

    # Zone occupancy visualization.
    fig, ax = plt.subplots(figsize=(10, 6))
    counts = np.zeros((args.grid_rows, args.grid_cols), dtype=int)
    for row in points.itertuples():
        counts[int(row.zone_row), int(row.zone_col)] += 1

    mesh = ax.imshow(counts, aspect="auto", interpolation="nearest")
    fig.colorbar(mesh, ax=ax, label="Observed player positions")
    ax.set_xticks(range(args.grid_cols), [
        f"C{i+1}" for i in range(args.grid_cols)
    ])
    ax.set_yticks(range(args.grid_rows), [
        f"R{i+1}" for i in range(args.grid_rows)
    ])
    ax.set_xlabel("Image grid column")
    ax.set_ylabel("Image grid row")
    ax.set_title("Zone occupancy (image-relative grid, not pitch zones)")
    fig.tight_layout()
    fig.savefig(plots / "zone_occupancy.png", dpi=160)
    plt.close(fig)

    # Team spread over time.
    if not team_metrics.empty:
        fig, ax = plt.subplots(figsize=(12, 5))
        for team in ["team_a", "team_b"]:
            subset = team_metrics[team_metrics["team_label"] == team]
            ax.plot(
                subset["time_seconds"],
                subset["spread_rms_px"],
                label=team.replace("_", " ").title(),
            )
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("RMS spread (image pixels)")
        ax.set_title("Team spatial spread over time (image space)")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots / "team_spread.png", dpi=160)
        plt.close(fig)

    # Aggregate summary for future API/dashboard use.
    summary = {
        "schema_version": "1.0",
        "video": str(args.video),
        "input_predictions": str(args.input),
        "coordinate_space": "image_pixels",
        "pitch_calibration_validated": False,
        "physical_distances_validated": False,
        "video_width": width,
        "video_height": height,
        "fps": fps,
        "video_frames": video_frames,
        "input_observations": int(len(df)),
        "valid_unique_frame_id_positions": int(len(points)),
        "temporary_ids": int(points["canonical_id"].nunique()),
        "team_labels": {
            str(k): int(v)
            for k, v in points["team_label"].value_counts(dropna=False).items()
        },
        "zone_grid": {
            "rows": args.grid_rows,
            "columns": args.grid_cols,
            "meaning": "Image-relative cells; not calibrated pitch zones.",
        },
        "outputs": [
            "football2_player_zone_occupancy.csv",
            "football2_team_zone_occupancy.csv",
            "football2_team_spatial_metrics.csv",
            "football2_player_proximity.csv",
            "plots/all_players_heatmap.png",
            "plots/team_a_heatmap.png",
            "plots/team_b_heatmap.png",
            "plots/zone_occupancy.png",
            "plots/team_spread.png",
            "plots/player_heatmaps/",
        ],
        "limitations": [
            "Camera movement can distort image-space spatial summaries.",
            "Distances are pixels, not meters.",
            "Image grid cells are not football pitch zones.",
            "Team labels and temporary tracking IDs are not fully verified identities.",
            "Observed-frame counts are not continuous time-in-zone measurements.",
        ],
    }

    with open(output / "football2_spatial_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSpatial analytics completed.")
    print(f"Video size: {width}x{height} at {fps:.2f} FPS")
    print(f"Unique frame/ID positions: {len(points)}")
    print(f"Temporary IDs: {points['canonical_id'].nunique()}")
    print("\nTeam labels:")
    print(points["team_label"].value_counts(dropna=False).to_string())
    print(f"\nOutput directory: {output}")
    print(f"Plots directory:  {plots}")


if __name__ == "__main__":
    main()
