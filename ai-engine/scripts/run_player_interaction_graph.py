
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


# ============================================================
# ATHLETIQ - PLAYER INTERACTION GRAPH
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_CSV = (
    ROOT
    / "outputs"
    / "team_identification"
    / "football2_team_predictions.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "player_interaction_graph"

# Initial image-space parameters; not calibrated pitch distances.
DISTANCE_THRESHOLD_PX = 75.0
MIN_SHARED_FRAMES = 3
FPS = 25.0

TEAM_LABELS = {"team_a", "team_b"}


def find_column(df, candidates, required=True):
    """Find a column without depending on capitalization."""
    lookup = {str(column).lower(): column for column in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    if required:
        raise ValueError(
            f"Required column not found: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def load_player_tracks():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_CSV}\n"
            "Run the existing team-identification pipeline first."
        )

    df = pd.read_csv(INPUT_CSV)

    frame_col = find_column(
        df, ["frame_id", "frame", "frame_number"]
    )
    id_col = find_column(
        df, ["canonical_id", "player_id", "track_id", "id"]
    )
    x1_col = find_column(df, ["x1", "xmin", "left"])
    x2_col = find_column(df, ["x2", "xmax", "right"])
    y2_col = find_column(df, ["y2", "ymax", "bottom"])

    team_col = find_column(
        df,
        ["aggregated_team", "team_prediction", "team_label", "team"],
    )

    confidence_col = find_column(
        df,
        ["team_confidence", "team_probability", "team_score"],
        required=False,
    )

    df["frame_id"] = pd.to_numeric(
        df[frame_col], errors="coerce"
    )
    df["canonical_id"] = pd.to_numeric(
        df[id_col], errors="coerce"
    )

    # Approximate ground-contact point at the bottom-center of each bbox.
    df["player_x"] = (
        pd.to_numeric(df[x1_col], errors="coerce")
        + pd.to_numeric(df[x2_col], errors="coerce")
    ) / 2

    df["player_y"] = pd.to_numeric(
        df[y2_col], errors="coerce"
    )

    df["team"] = (
        df[team_col].astype(str).str.lower().str.strip()
    )

    if confidence_col:
        df["team_confidence"] = pd.to_numeric(
            df[confidence_col], errors="coerce"
        )
    else:
        df["team_confidence"] = np.nan

    df = df.dropna(
        subset=[
            "frame_id",
            "canonical_id",
            "player_x",
            "player_y",
        ]
    ).copy()

    df["frame_id"] = df["frame_id"].astype(int)
    df["canonical_id"] = df["canonical_id"].astype(int)

    # Retain one observation per temporary player ID per frame.
    df = df.sort_values(
        ["frame_id", "canonical_id", "team_confidence"],
        ascending=[True, True, False],
        na_position="last",
    )

    df = df.drop_duplicates(
        subset=["frame_id", "canonical_id"],
        keep="first",
    )

    return df[
        [
            "frame_id",
            "canonical_id",
            "player_x",
            "player_y",
            "team",
            "team_confidence",
        ]
    ]


def build_interaction_graph(tracks):
    graph = nx.Graph()

    # --------------------------------------------------------
    # 1. Add a node for every tracked player ID.
    # --------------------------------------------------------
    for player_id, group in tracks.groupby("canonical_id"):
        labels = group.loc[
            group["team"].isin(TEAM_LABELS), "team"
        ]

        team = (
            labels.mode().iloc[0]
            if not labels.empty
            else "unknown"
        )

        graph.add_node(
            int(player_id),
            team=str(team),
            observations=int(len(group)),
            first_frame=int(group["frame_id"].min()),
            last_frame=int(group["frame_id"].max()),
        )

    # --------------------------------------------------------
    # 2. Collect nearby-player observations by shared frame.
    # --------------------------------------------------------
    pair_data = {}

    for frame_id, group in tracks.groupby("frame_id", sort=True):
        group = group.reset_index(drop=True)

        if len(group) < 2:
            continue

        coordinates = group[
            ["player_x", "player_y"]
        ].to_numpy(dtype=float)

        ids = group["canonical_id"].to_numpy(dtype=int)

        # Compare each pair only once in this frame.
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                distance = float(
                    np.linalg.norm(
                        coordinates[i] - coordinates[j]
                    )
                )

                if distance > DISTANCE_THRESHOLD_PX:
                    continue

                player_a, player_b = sorted(
                    (int(ids[i]), int(ids[j]))
                )

                key = (player_a, player_b)

                if key not in pair_data:
                    pair_data[key] = {
                        "frames": [],
                        "distances": [],
                    }

                pair_data[key]["frames"].append(int(frame_id))
                pair_data[key]["distances"].append(distance)

    # --------------------------------------------------------
    # 3. Add edges for repeated proximity.
    # --------------------------------------------------------
    for (player_a, player_b), data in pair_data.items():
        frames = data["frames"]
        distances = data["distances"]

        if len(frames) < MIN_SHARED_FRAMES:
            continue

        team_a = graph.nodes[player_a]["team"]
        team_b = graph.nodes[player_b]["team"]

        if team_a not in TEAM_LABELS or team_b not in TEAM_LABELS:
            relation = "unknown"
        elif team_a == team_b:
            relation = "same_team"
        else:
            relation = "opponents"

        graph.add_edge(
            player_a,
            player_b,
            relation=relation,
            shared_nearby_frames=len(frames),
            nearby_seconds=round(len(frames) / FPS, 3),
            mean_distance_px=round(float(np.mean(distances)), 2),
            median_distance_px=round(float(np.median(distances)), 2),
            minimum_distance_px=round(float(np.min(distances)), 2),
            first_nearby_frame=min(frames),
            last_nearby_frame=max(frames),
        )

    return graph


def export_csv(graph):
    # Node-level metrics.
    node_rows = []

    for player_id, attributes in graph.nodes(data=True):
        edges = list(graph.edges(player_id, data=True))

        node_rows.append({
            "canonical_id": player_id,
            **attributes,
            "interaction_degree": graph.degree(player_id),
            "same_team_edges": sum(
                edge["relation"] == "same_team"
                for _, _, edge in edges
            ),
            "opponent_edges": sum(
                edge["relation"] == "opponents"
                for _, _, edge in edges
            ),
        })

    node_columns = [
        "canonical_id",
        "team",
        "observations",
        "first_frame",
        "last_frame",
        "interaction_degree",
        "same_team_edges",
        "opponent_edges",
    ]

    pd.DataFrame(
        node_rows, columns=node_columns
    ).sort_values("canonical_id").to_csv(
        OUTPUT_DIR / "football2_interaction_nodes.csv",
        index=False,
    )

    # Edge-level metrics.
    edge_rows = []

    for player_a, player_b, attributes in graph.edges(data=True):
        edge_rows.append({
            "player_id_a": player_a,
            "player_id_b": player_b,
            **attributes,
        })

    edge_columns = [
        "player_id_a",
        "player_id_b",
        "relation",
        "shared_nearby_frames",
        "nearby_seconds",
        "mean_distance_px",
        "median_distance_px",
        "minimum_distance_px",
        "first_nearby_frame",
        "last_nearby_frame",
    ]

    edges_df = pd.DataFrame(
        edge_rows, columns=edge_columns
    )

    if not edges_df.empty:
        edges_df = edges_df.sort_values(
            "shared_nearby_frames", ascending=False
        )

    edges_df.to_csv(
        OUTPUT_DIR / "football2_interaction_edges.csv",
        index=False,
    )


def draw_graph(graph):
    output_path = (
        OUTPUT_DIR / "football2_interaction_graph.png"
    )

    plt.figure(figsize=(14, 10))

    if graph.number_of_nodes() == 0:
        plt.text(
            0.5, 0.5, "No player observations found",
            ha="center", va="center",
        )
        plt.axis("off")
        plt.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close()
        return

    positions = nx.spring_layout(graph, seed=42, k=0.8)

    node_colors = []

    for _, attributes in graph.nodes(data=True):
        if attributes["team"] == "team_a":
            node_colors.append("#3977C3")
        elif attributes["team"] == "team_b":
            node_colors.append("#E4A044")
        else:
            node_colors.append("#999999")

    edge_colors = []
    edge_widths = []

    for _, _, attributes in graph.edges(data=True):
        if attributes["relation"] == "same_team":
            edge_colors.append("#5B9E68")
        elif attributes["relation"] == "opponents":
            edge_colors.append("#C85B5B")
        else:
            edge_colors.append("#999999")

        edge_widths.append(
            0.5 + min(
                attributes["shared_nearby_frames"] / 10,
                4,
            )
        )

    nx.draw_networkx_edges(
        graph,
        positions,
        edge_color=edge_colors,
        width=edge_widths,
        alpha=0.45,
    )

    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=node_colors,
        node_size=280,
        edgecolors="white",
        linewidths=0.7,
    )

    nx.draw_networkx_labels(
        graph,
        positions,
        labels={
            node: str(node)
            for node in graph.nodes()
        },
        font_size=7,
    )

    plt.title(
        "AthletiQ Player Interaction Graph\n"
        "Blue: Team A | Orange: Team B | "
        "Green: same-team edges | Red: opponent edges"
    )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading AthletiQ player tracks...")
    tracks = load_player_tracks()

    print(f"Loaded observations: {len(tracks)}")
    print(
        "Building graph with distance threshold "
        f"{DISTANCE_THRESHOLD_PX}px..."
    )

    graph = build_interaction_graph(tracks)

    export_csv(graph)
    draw_graph(graph)

    same_team_edges = sum(
        data["relation"] == "same_team"
        for _, _, data in graph.edges(data=True)
    )

    opponent_edges = sum(
        data["relation"] == "opponents"
        for _, _, data in graph.edges(data=True)
    )

    unknown_edges = (
        graph.number_of_edges()
        - same_team_edges
        - opponent_edges
    )

    summary = {
        "schema_version": "1.0",
        "method": "same-frame player proximity graph",
        "coordinate_space": "image_pixels",
        "physical_distances_validated": False,
        "parameters": {
            "distance_threshold_px": DISTANCE_THRESHOLD_PX,
            "minimum_shared_nearby_frames": MIN_SHARED_FRAMES,
            "fps_assumed": FPS,
        },
        "summary": {
            "player_nodes": graph.number_of_nodes(),
            "interaction_edges": graph.number_of_edges(),
            "same_team_edges": int(same_team_edges),
            "opponent_edges": int(opponent_edges),
            "unknown_relation_edges": int(unknown_edges),
            "average_degree": round(
                float(np.mean([
                    degree for _, degree in graph.degree()
                ])),
                3,
            ) if graph.number_of_nodes() else 0.0,
        },
        "limitations": [
            "Nodes are temporary IDs, not verified real-player identities.",
            "Edges represent proximity, not confirmed passes or interactions.",
            "Distances are pixels and are affected by perspective and camera motion.",
            "Team labels are preliminary predictions.",
        ],
    }

    with open(
        OUTPUT_DIR / "football2_interaction_graph.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print("\nPlayer interaction graph completed.")
    print(json.dumps(summary["summary"], indent=2))
    print(f"\nOutputs saved in:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
