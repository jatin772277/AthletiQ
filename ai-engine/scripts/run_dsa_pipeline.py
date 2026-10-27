
from pathlib import Path
from collections import defaultdict
import heapq
import json

import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import cKDTree


# ============================================================
# ATHLETIQ: INTEGRATED DSA PIPELINE
# HashMap + KD-Tree + Interaction Graph + Priority Queue
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TRACKS = ROOT / "outputs/joint_tracking/football2_player_tracks.csv"
TEAMS = ROOT / "outputs/team_identification/football2_team_predictions.csv"
MOVEMENT = ROOT / "outputs/performance/football2_movement_analytics.csv"
SPATIAL = ROOT / "outputs/spatial_analytics/football2_player_zone_occupancy.csv"
POSSESSION = ROOT / "outputs/ball_possession/football2_player_possession_summary.csv"
GRAPH_NODES = ROOT / "outputs/player_interaction_graph/football2_interaction_nodes.csv"
GRAPH_EDGES = ROOT / "outputs/player_interaction_graph/football2_interaction_edges.csv"

OUTPUT = ROOT / "outputs/dsa"
OUTPUT.mkdir(parents=True, exist_ok=True)

FPS = 25.0
KD_TREE_NEIGHBOR_RADIUS_PX = 75.0


# ============================================================
# GENERAL UTILITIES
# ============================================================

def find_column(df, candidates, required=True):
    """Find a column using case-insensitive exact-name matching."""
    lookup = {str(col).lower(): col for col in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    if required:
        raise ValueError(
            f"Could not find any of {candidates}.\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def find_player_id_column(df):
    return find_column(
        df,
        ["canonical_id", "player_id", "player_id_a", "track_id", "id"],
    )


def normalize_player_id(df):
    """Standardize player IDs to integer canonical_id values."""
    result = df.copy()
    id_col = find_player_id_column(result)

    result["canonical_id"] = pd.to_numeric(
        result[id_col], errors="coerce"
    )
    result = result.dropna(subset=["canonical_id"]).copy()
    result["canonical_id"] = result["canonical_id"].astype(int)

    return result


def load_csv(path, required=False):
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required input missing: {path}")

        print(f"[SKIP] Optional input not found: {path}")
        return None

    print(f"[LOAD] {path.relative_to(ROOT)}")
    return pd.read_csv(path)


def aggregate_numeric_features(df, prefix, exclude=None):
    """
    Aggregate numeric data per player.

    Counts, durations, distances and frame totals are summed.
    Maximum/peak metrics use max.
    Other numeric features use mean.

    For inputs that already have one row per player, aggregation
    preserves that player's value.
    """
    if df is None or df.empty:
        return None

    df = normalize_player_id(df)
    excluded = set(exclude or [])
    excluded.update({
        "canonical_id",
        "frame_id",
        "frame",
        "frame_number",
        "time_seconds",
        "timestamp",
        "id",
    })

    numeric_cols = [
        col
        for col in df.select_dtypes(include=np.number).columns
        if col not in excluded
    ]

    if not numeric_cols:
        return None

    aggregations = {}

    for col in numeric_cols:
        name = str(col).lower()

        if any(word in name for word in (
            "distance", "total", "count", "frames"
        )):
            aggregations[col] = "sum"
        elif any(word in name for word in ("max", "peak")):
            aggregations[col] = "max"
        else:
            aggregations[col] = "mean"

    result = df.groupby("canonical_id").agg(aggregations)
    result.columns = [
        f"{prefix}_{str(col).lower()}"
        for col in result.columns
    ]

    return result.reset_index()


def merge_features(base, features):
    """Left-merge features while preserving every tracked player ID."""
    if features is None or features.empty:
        return base

    if "canonical_id" not in features.columns:
        raise ValueError("Feature table is missing canonical_id.")

    if features["canonical_id"].duplicated().any():
        raise ValueError(
            "Feature table contains duplicate canonical_id values."
        )

    return base.merge(
        features,
        on="canonical_id",
        how="left",
        validate="one_to_one",
    )


# ============================================================
# HASHMAP-STYLE PLAYER LOOKUP
# ============================================================

def build_player_lookup(features):
    """Build a JSON-safe dictionary keyed by temporary player ID."""
    lookup = {}

    for _, row in features.iterrows():
        player_id = int(row["canonical_id"])
        record = {}

        for column, value in row.items():
            if pd.isna(value):
                record[column] = None
            elif isinstance(value, (np.integer,)):
                record[column] = int(value)
            elif isinstance(value, (np.floating,)):
                record[column] = float(value)
            else:
                record[column] = value

        lookup[player_id] = record

    return lookup


# ============================================================
# KD-TREE SPATIAL FEATURES
# ============================================================

def calculate_kdtree_features():
    tracks = load_csv(TEAMS, required=True)

    frame_col = find_column(
        tracks, ["frame_id", "frame", "frame_number"]
    )
    id_col = find_column(
        tracks, ["canonical_id", "player_id", "track_id", "id"]
    )
    x1_col = find_column(tracks, ["x1", "xmin", "left"])
    x2_col = find_column(tracks, ["x2", "xmax", "right"])
    y2_col = find_column(tracks, ["y2", "ymax", "bottom"])

    tracks["frame_id"] = pd.to_numeric(
        tracks[frame_col], errors="coerce"
    )
    tracks["canonical_id"] = pd.to_numeric(
        tracks[id_col], errors="coerce"
    )
    tracks["foot_x"] = (
        pd.to_numeric(tracks[x1_col], errors="coerce")
        + pd.to_numeric(tracks[x2_col], errors="coerce")
    ) / 2.0
    tracks["foot_y"] = pd.to_numeric(
        tracks[y2_col], errors="coerce"
    )

    tracks = tracks.dropna(
        subset=["frame_id", "canonical_id", "foot_x", "foot_y"]
    ).copy()

    tracks["frame_id"] = tracks["frame_id"].astype(int)
    tracks["canonical_id"] = tracks["canonical_id"].astype(int)

    tracks = tracks.drop_duplicates(["frame_id", "canonical_id"])

    per_player = defaultdict(lambda: {
        "nearest_distances": [],
        "neighbor_counts": [],
        "frames": 0,
    })

    for _, frame in tracks.groupby("frame_id", sort=True):
        xy = frame[["foot_x", "foot_y"]].to_numpy(dtype=float)
        ids = frame["canonical_id"].to_numpy(dtype=int)

        if len(frame) < 2:
            for player_id in ids:
                per_player[int(player_id)]["frames"] += 1
            continue

        tree = cKDTree(xy)

        # The first returned point is the player itself.
        distances, _ = tree.query(xy, k=2)
        nearest_other = distances[:, 1]

        neighbor_lists = tree.query_ball_point(
            xy, r=KD_TREE_NEIGHBOR_RADIUS_PX
        )

        for i, player_id in enumerate(ids):
            record = per_player[int(player_id)]
            record["frames"] += 1
            record["nearest_distances"].append(
                float(nearest_other[i])
            )
            record["neighbor_counts"].append(
                max(0, len(neighbor_lists[i]) - 1)
            )

    rows = []

    for player_id, record in per_player.items():
        distances = record["nearest_distances"]
        counts = record["neighbor_counts"]

        rows.append({
            "canonical_id": player_id,
            "kdtree_observed_frames": record["frames"],
            "kdtree_mean_nearest_distance_px": (
                float(np.mean(distances)) if distances else np.nan
            ),
            "kdtree_median_nearest_distance_px": (
                float(np.median(distances)) if distances else np.nan
            ),
            "kdtree_mean_neighbors_within_radius": (
                float(np.mean(counts)) if counts else np.nan
            ),
        })

    result = pd.DataFrame(rows)

    result.to_csv(
        OUTPUT / "football2_kdtree_features.csv",
        index=False,
    )

    print(f"[KD-TREE] Features for {len(result)} player IDs")
    return result


# ============================================================
# PLAYER INTERACTION GRAPH FEATURES
# ============================================================

def calculate_graph_features():
    nodes = load_csv(GRAPH_NODES)
    edges = load_csv(GRAPH_EDGES)

    graph = nx.Graph()

    if nodes is not None and not nodes.empty:
        nodes = normalize_player_id(nodes)

        for _, row in nodes.iterrows():
            player_id = int(row["canonical_id"])
            team = str(row.get("team", "unknown"))
            graph.add_node(player_id, team=team)

    if edges is not None and not edges.empty:
        a_col = find_column(edges, ["player_id_a"])
        b_col = find_column(edges, ["player_id_b"])
        relation_col = find_column(
            edges, ["relation"], required=False
        )
        frames_col = find_column(
            edges, ["shared_nearby_frames"], required=False
        )

        for _, row in edges.iterrows():
            a = pd.to_numeric(row[a_col], errors="coerce")
            b = pd.to_numeric(row[b_col], errors="coerce")

            if pd.isna(a) or pd.isna(b):
                continue

            a, b = int(a), int(b)

            relation = (
                str(row[relation_col])
                if relation_col else "unknown"
            )

            weight = (
                float(row[frames_col])
                if frames_col and pd.notna(row[frames_col])
                else 1.0
            )

            graph.add_edge(
                a, b, relation=relation, weight=weight
            )

    component_by_player = {}
    component_size_by_player = {}

    for component_id, component in enumerate(
        nx.connected_components(graph), start=1
    ):
        for player_id in component:
            component_by_player[player_id] = component_id
            component_size_by_player[player_id] = len(component)

    rows = []

    for player_id in graph.nodes:
        edges_for_player = list(
            graph.edges(player_id, data=True)
        )

        rows.append({
            "canonical_id": int(player_id),
            "graph_degree": int(graph.degree(player_id)),
            "graph_weighted_degree": float(sum(
                data.get("weight", 1.0)
                for _, _, data in edges_for_player
            )),
            "graph_same_team_edges": sum(
                data.get("relation") == "same_team"
                for _, _, data in edges_for_player
            ),
            "graph_opponent_edges": sum(
                data.get("relation") == "opponents"
                for _, _, data in edges_for_player
            ),
            "graph_component_id": component_by_player.get(player_id),
            "graph_component_size": component_size_by_player.get(
                player_id, 1
            ),
        })

    result = pd.DataFrame(rows)

    result.to_csv(
        OUTPUT / "football2_graph_features.csv",
        index=False,
    )

    print(
        f"[GRAPH] {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges, "
        f"{nx.number_connected_components(graph) if graph else 0} components"
    )

    return result


# ============================================================
# BASELINE SCORE AND PRIORITY QUEUE
# ============================================================

def percentile_score(series):
    """
    Convert a numeric feature into a 0-100 within-video percentile.

    Missing values receive a neutral 50 rather than being interpreted
    as the worst player. If fewer than two usable observations exist,
    the feature cannot distinguish players and receives 50.
    """
    values = pd.to_numeric(series, errors="coerce")
    valid = values.notna()

    result = pd.Series(50.0, index=series.index, dtype=float)

    if valid.sum() < 2:
        return result

    result.loc[valid] = (
        values.loc[valid].rank(pct=True, method="average") * 100.0
    )

    return result


def find_first_existing_column(df, candidates):
    """Return the first exact matching column that exists."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def create_baseline_ranking(features):
    """
    Create a transparent movement-activity baseline.

    This is not a trained ML performance score. It combines:
      - 70% distance percentile
      - 30% average-speed percentile

    Both components are ranked within the current video. They are
    not calibrated measures of real-world athletic performance.
    """
    result = features.copy()

    # Support both calibrated real-world metres/mps and fallback pixel columns
    distance_col = find_first_existing_column(
        result,
        [
            "movement_distance_meters",
            "movement_total_distance_m",
            "movement_distance_pixels",
            "movement_distance_pixels_sum",
            "movement_total_distance_pixels",
        ],
    )

    speed_col = find_first_existing_column(
        result,
        [
            "movement_average_speed_mps",
            "movement_avg_speed_mps",
            "movement_average_speed_pixels_per_second",
            "movement_avg_speed_pixels_per_second",
            "movement_mean_speed_pixels_per_second",
        ],
    )

    if distance_col is None:
        print(
            "[WARNING] No supported movement-distance column found. "
            "Distance contribution will be neutral."
        )
        distance_score = pd.Series(
            50.0, index=result.index, dtype=float
        )
    else:
        print(f"[SCORING] Distance feature: {distance_col}")
        distance_score = percentile_score(result[distance_col])

    if speed_col is None:
        print(
            "[WARNING] No supported average-speed column found. "
            "Speed contribution will be neutral."
        )
        speed_score = pd.Series(
            50.0, index=result.index, dtype=float
        )
    else:
        print(f"[SCORING] Speed feature: {speed_col}")
        speed_score = percentile_score(result[speed_col])

    # Export score components so the score is auditable.
    result["baseline_distance_percentile"] = distance_score.round(2)
    result["baseline_speed_percentile"] = speed_score.round(2)

    result["baseline_activity_index"] = (
        0.70 * distance_score + 0.30 * speed_score
    ).round(2)

    result["score_type"] = "heuristic_baseline_not_ML"
    result["activity_score_status"] = (
        "experimental_within_video_activity_percentile"
    )

    # Priority queue: highest score first; ID resolves ties consistently.
    priority_queue = []

    for _, row in result.iterrows():
        player_id = int(row["canonical_id"])
        score = float(row["baseline_activity_index"])

        heapq.heappush(
            priority_queue, (-score, player_id)
        )

    ranking = []
    rank = 1

    while priority_queue:
        negative_score, player_id = heapq.heappop(priority_queue)

        ranking.append({
            "rank": rank,
            "canonical_id": player_id,
            "baseline_activity_index": -negative_score,
        })
        rank += 1

    ranking_df = pd.DataFrame(ranking)

    ranking_df.to_csv(
        OUTPUT / "football2_priority_queue_ranking.csv",
        index=False,
    )

    return result, ranking_df


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("\n========== ATHLETIQ DSA PIPELINE ==========\n")

    # 1. Base set of IDs comes from the tracking output.
    base_tracks = load_csv(TRACKS, required=True)
    base_tracks = normalize_player_id(base_tracks)

    player_ids = sorted(
        base_tracks["canonical_id"].unique().astype(int)
    )

    base = pd.DataFrame({"canonical_id": player_ids})

    print(f"[BASE] {len(base)} temporary player IDs")

    # 2. Movement features.
    movement_df = load_csv(MOVEMENT)
    movement_features = aggregate_numeric_features(
        movement_df, "movement"
    )
    base = merge_features(base, movement_features)

    # 3. Spatial features.
    spatial_df = load_csv(SPATIAL)
    spatial_features = aggregate_numeric_features(
        spatial_df, "spatial"
    )
    base = merge_features(base, spatial_features)

    # 4. Ball-possession features.
    possession_df = load_csv(POSSESSION)
    possession_features = aggregate_numeric_features(
        possession_df, "possession"
    )
    base = merge_features(base, possession_features)

    # 5. KD-tree spatial-neighbour features.
    kdtree_features = calculate_kdtree_features()
    base = merge_features(base, kdtree_features)

    # 6. Interaction graph features.
    graph_features = calculate_graph_features()
    base = merge_features(base, graph_features)

    # 7. Heuristic baseline and priority-queue ranking.
    base, ranking = create_baseline_ranking(base)

    # 8. Export player lookup for downstream API/report generation.
    lookup = build_player_lookup(base)

    with open(
        OUTPUT / "football2_player_lookup.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            lookup,
            file,
            indent=2,
            allow_nan=False,
        )

    # 9. Unified feature table.
    base.to_csv(
        OUTPUT / "football2_unified_player_features.csv",
        index=False,
    )

    # 10. Summary and validation diagnostics.
    score = base["baseline_activity_index"]

    summary = {
        "schema_version": "1.1",
        "pipeline": "AthletiQ DSA feature integration",
        "player_count": int(len(base)),
        "feature_column_count": int(len(base.columns)),
        "kdtree_radius_px": KD_TREE_NEIGHBOR_RADIUS_PX,
        "priority_queue": "ranks heuristic baseline activity index",
        "ml_model_trained": False,
        "validated_physical_metrics": False,
        "scoring": {
            "distance_weight": 0.70,
            "average_speed_weight": 0.30,
            "distance_column": find_first_existing_column(
                base,
                [
                    "movement_distance_pixels",
                    "movement_distance_pixels_sum",
                    "movement_total_distance_pixels",
                ],
            ),
            "speed_column": find_first_existing_column(
                base,
                [
                    "movement_average_speed_pixels_per_second",
                    "movement_avg_speed_pixels_per_second",
                    "movement_mean_speed_pixels_per_second",
                ],
            ),
            "unique_score_count": int(score.nunique(dropna=True)),
            "minimum_score": (
                float(score.min()) if score.notna().any() else None
            ),
            "maximum_score": (
                float(score.max()) if score.notna().any() else None
            ),
        },
        "outputs": [
            "football2_unified_player_features.csv",
            "football2_kdtree_features.csv",
            "football2_graph_features.csv",
            "football2_priority_queue_ranking.csv",
            "football2_player_lookup.json",
        ],
        "limitations": [
            "Temporary IDs are not verified real-player identities.",
            "Image-space distances are pixels, not metres.",
            "The activity index is a heuristic, not a trained ML performance score.",
            "Percentiles are relative to this video and should not be compared directly across videos.",
            "Missing features receive neutral percentile scores.",
            "Proximity graph connections do not prove passes or tactical cooperation.",
        ],
    }

    with open(
        OUTPUT / "football2_dsa_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            allow_nan=False,
        )

    # 11. Human-readable diagnostics.
    print("\n========== DSA PIPELINE COMPLETE ==========")
    print(f"Players in unified table: {len(base)}")
    print(f"Feature columns: {len(base.columns)}")
    print(f"Unique baseline scores: {score.nunique(dropna=True)}")
    print(f"Score range: {score.min():.2f} - {score.max():.2f}")

    print("\nTop 10 baseline activity rankings:")
    print(ranking.head(10).to_string(index=False))

    print(f"\nOutputs saved to:\n{OUTPUT}")

    if score.nunique(dropna=True) <= 1:
        print(
            "\n[WARNING] All players still have the same score. "
            "Inspect the selected feature columns and their variation."
        )
    else:
        print(
            "\n[OK] Baseline scores vary across players. "
            "This remains an experimental within-video activity index."
        )


if __name__ == "__main__":
    main()
