
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


# ============================================================
# ATHLETIQ - PLAYER INTERACTION GRAPH V2
# Uses existing graph CSV files; does not rerun detection.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "outputs" / "player_interaction_graph"
OUTPUT_DIR = INPUT_DIR / "v2"

NODES_FILE = INPUT_DIR / "football2_interaction_nodes.csv"
EDGES_FILE = INPUT_DIR / "football2_interaction_edges.csv"

# Keep interactions observed in at least this many shared frames.
MIN_INTERACTION_FRAMES = 8

# Fixed layout seed makes the visualizations reproducible.
LAYOUT_SEED = 42

TEAM_A_COLOR = "#3977C3"
TEAM_B_COLOR = "#E4A044"
UNKNOWN_COLOR = "#999999"
SAME_TEAM_COLOR = "#5B9E68"
OPPONENT_COLOR = "#C85B5B"


def load_existing_graph():
    if not NODES_FILE.exists():
        raise FileNotFoundError(
            f"Missing nodes file:\n{NODES_FILE}\n"
            "Run the original interaction graph script first."
        )

    if not EDGES_FILE.exists():
        raise FileNotFoundError(
            f"Missing edges file:\n{EDGES_FILE}\n"
            "Run the original interaction graph script first."
        )

    nodes_df = pd.read_csv(NODES_FILE)
    edges_df = pd.read_csv(EDGES_FILE)

    graph = nx.Graph()

    for _, row in nodes_df.iterrows():
        player_id = int(row["canonical_id"])

        graph.add_node(
            player_id,
            team=str(row["team"]),
            observations=int(row["observations"]),
            original_degree=int(row["interaction_degree"]),
        )

    for _, row in edges_df.iterrows():
        graph.add_edge(
            int(row["player_id_a"]),
            int(row["player_id_b"]),
            relation=str(row["relation"]),
            shared_nearby_frames=int(row["shared_nearby_frames"]),
            nearby_seconds=float(row["nearby_seconds"]),
            mean_distance_px=float(row["mean_distance_px"]),
            median_distance_px=float(row["median_distance_px"]),
            minimum_distance_px=float(row["minimum_distance_px"]),
        )

    return graph


def make_filtered_graph(graph, relation=None):
    """Apply duration and optional relationship filters."""
    filtered = nx.Graph()

    # Add only nodes that participate in retained edges.
    for a, b, attrs in graph.edges(data=True):
        if attrs["shared_nearby_frames"] < MIN_INTERACTION_FRAMES:
            continue

        if relation is not None and attrs["relation"] != relation:
            continue

        filtered.add_node(a, **graph.nodes[a])
        filtered.add_node(b, **graph.nodes[b])
        filtered.add_edge(a, b, **attrs)

    return filtered


def draw_graph(graph, output_path, title, edge_color):
    fig, ax = plt.subplots(figsize=(15, 10))

    if graph.number_of_nodes() == 0:
        ax.text(
            0.5, 0.5,
            "No interactions passed the current filter",
            ha="center", va="center",
            transform=ax.transAxes,
        )
        ax.set_title(title)
        ax.axis("off")
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    positions = nx.spring_layout(
        graph,
        seed=LAYOUT_SEED,
        k=0.9,
        weight="shared_nearby_frames",
    )

    node_colors = []
    for _, attrs in graph.nodes(data=True):
        team = attrs.get("team", "unknown")

        if team == "team_a":
            node_colors.append(TEAM_A_COLOR)
        elif team == "team_b":
            node_colors.append(TEAM_B_COLOR)
        else:
            node_colors.append(UNKNOWN_COLOR)

    edge_widths = [
        0.8 + min(
            graph[a][b]["shared_nearby_frames"] / 8.0,
            5.0,
        )
        for a, b in graph.edges()
    ]

    nx.draw_networkx_edges(
        graph,
        positions,
        ax=ax,
        edge_color=edge_color,
        width=edge_widths,
        alpha=0.65,
    )

    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=ax,
        node_color=node_colors,
        node_size=420,
        edgecolors="white",
        linewidths=0.9,
    )

    nx.draw_networkx_labels(
        graph,
        positions,
        ax=ax,
        labels={node: str(node) for node in graph.nodes()},
        font_size=8,
        font_weight="bold",
    )

    ax.set_title(
        f"{title}\n"
        f"{graph.number_of_nodes()} participating IDs | "
        f"{graph.number_of_edges()} retained interactions",
        fontsize=13,
    )
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def export_player_ranking(graph, output_path):
    rows = []

    for player_id, attrs in graph.nodes(data=True):
        edges = list(graph.edges(player_id, data=True))

        same_team_edges = [
            data for _, _, data in edges
            if data["relation"] == "same_team"
        ]

        opponent_edges = [
            data for _, _, data in edges
            if data["relation"] == "opponents"
        ]

        all_frames = sum(
            data["shared_nearby_frames"] for _, _, data in edges
        )

        rows.append({
            "canonical_id": player_id,
            "team": attrs.get("team", "unknown"),
            "observations": attrs.get("observations", 0),
            "interaction_degree": len(edges),
            "weighted_interaction_frames": all_frames,
            "same_team_connections": len(same_team_edges),
            "opponent_connections": len(opponent_edges),
            "same_team_nearby_frames": sum(
                data["shared_nearby_frames"]
                for data in same_team_edges
            ),
            "opponent_nearby_frames": sum(
                data["shared_nearby_frames"]
                for data in opponent_edges
            ),
        })

    columns = [
        "canonical_id",
        "team",
        "observations",
        "interaction_degree",
        "weighted_interaction_frames",
        "same_team_connections",
        "opponent_connections",
        "same_team_nearby_frames",
        "opponent_nearby_frames",
    ]

    ranking = pd.DataFrame(rows, columns=columns)

    if not ranking.empty:
        ranking = ranking.sort_values(
            ["weighted_interaction_frames", "interaction_degree"],
            ascending=False,
        )
        ranking.insert(0, "rank", range(1, len(ranking) + 1))

    ranking.to_csv(output_path, index=False)
    return ranking


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading existing interaction graph...")
    original_graph = load_existing_graph()

    print(f"Original nodes: {original_graph.number_of_nodes()}")
    print(f"Original edges: {original_graph.number_of_edges()}")
    print(
        "Minimum retained interaction: "
        f"{MIN_INTERACTION_FRAMES} shared frames"
    )

    # All retained relationships.
    strong_graph = make_filtered_graph(original_graph)

    # Separate relationship types.
    same_team_graph = make_filtered_graph(
        original_graph, relation="same_team"
    )
    opponent_graph = make_filtered_graph(
        original_graph, relation="opponents"
    )

    draw_graph(
        strong_graph,
        OUTPUT_DIR / "football2_strong_interactions.png",
        "AthletiQ - Stronger Player Proximity Network",
        "#777777",
    )

    draw_graph(
        same_team_graph,
        OUTPUT_DIR / "football2_same_team_interactions.png",
        "AthletiQ - Same-Team Proximity",
        SAME_TEAM_COLOR,
    )

    draw_graph(
        opponent_graph,
        OUTPUT_DIR / "football2_opponent_interactions.png",
        "AthletiQ - Opponent Proximity",
        OPPONENT_COLOR,
    )

    ranking = export_player_ranking(
        strong_graph,
        OUTPUT_DIR / "football2_player_interaction_ranking.csv",
    )

    summary = {
        "schema_version": "1.0",
        "method": "filtered existing player proximity graph",
        "minimum_shared_nearby_frames": MIN_INTERACTION_FRAMES,
        "fps_assumed": 25.0,
        "coordinate_space": "image_pixels",
        "physical_distances_validated": False,
        "original_nodes": original_graph.number_of_nodes(),
        "original_edges": original_graph.number_of_edges(),
        "filtered_nodes": strong_graph.number_of_nodes(),
        "filtered_edges": strong_graph.number_of_edges(),
        "same_team_edges": same_team_graph.number_of_edges(),
        "opponent_edges": opponent_graph.number_of_edges(),
        "top_10_by_weighted_interaction_frames": (
            ranking.head(10).to_dict(orient="records")
            if not ranking.empty else []
        ),
        "limitations": [
            "IDs are temporary tracking identities.",
            "Proximity does not prove a pass or intentional interaction.",
            "Distances are image pixels, not calibrated pitch distances.",
            "Ranking measures proximity-network connectivity, not player quality.",
        ],
    }

    with open(
        OUTPUT_DIR / "football2_interaction_v2_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print("\nVersion 2 completed.")
    print(json.dumps({
        "original_nodes": summary["original_nodes"],
        "original_edges": summary["original_edges"],
        "filtered_nodes": summary["filtered_nodes"],
        "filtered_edges": summary["filtered_edges"],
        "same_team_edges": summary["same_team_edges"],
        "opponent_edges": summary["opponent_edges"],
    }, indent=2))

    print(f"\nOutputs saved in:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
