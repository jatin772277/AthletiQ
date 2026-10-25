"""Unit and Integration Tests for AthletiQ FastAPI Backend."""

from __future__ import annotations
import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "3.0.0"
    assert "data_retention_policy" in data


def test_list_matches():
    response = client.get("/api/matches")
    assert response.status_code == 200
    matches = response.json()
    assert isinstance(matches, list)
    assert len(matches) >= 1
    match_ids = [m["match_id"] for m in matches]
    assert "football4" in match_ids


def test_match_summary():
    response = client.get("/api/matches/football4/summary")
    assert response.status_code == 200
    summary = response.json()
    assert summary["match_id"] == "football4"
    assert summary["total_active_players"] == 32
    assert summary["observation_retention_pct"] == 100.0
    assert summary["total_raw_observations_preserved"] == 7215
    assert summary["match_total_touches"] == 171
    assert summary["team_a"]["squad_size"] == 16
    assert summary["team_b"]["squad_size"] == 16
    assert summary["team_a"]["total_collective_touches"] == 137
    assert summary["team_b"]["total_collective_touches"] == 34


def test_get_players():
    response = client.get("/api/matches/football4/players")
    assert response.status_code == 200
    players = response.json()
    assert len(players) == 32
    # Verify player 1 is Team A, player 17 is Team B
    p1 = next(p for p in players if p["player_id"] == 1)
    p17 = next(p for p in players if p["player_id"] == 17)
    assert p1["team"] == "team_a"
    assert p17["team"] == "team_b"


def test_get_player_detail_and_bucket():
    response = client.get("/api/matches/football4/players/14")
    assert response.status_code == 200
    p14 = response.json()
    assert p14["player_id"] == 14
    assert p14["team"] == "team_a"

    # Verify primary kinematics
    prim = p14["primary_trajectory_kinematics"]
    assert prim["observations"] == 370
    assert prim["distance_meters"] == 86.34

    # Verify removed players bucket
    bucket = p14["removed_players_bucket"]
    assert bucket["removed_players_count"] == 18
    assert bucket["absorbed_observations"] == 500
    assert bucket["absorbed_distance_meters"] == 120.02
    assert bucket["absorbed_touches"] == 8
    assert len(bucket["absorbed_tracklets"]) == 18

    # Verify collective metrics (zero wastage)
    coll = p14["collective_metrics"]
    assert coll["total_observations"] == 591
    assert coll["total_distance_meters"] == 206.36
    assert coll["total_touches_detected"] == 8
    assert coll["observation_retention_pct"] == 100.0


def test_get_buckets():
    response = client.get("/api/matches/football4/buckets")
    assert response.status_code == 200
    buckets = response.json()
    assert len(buckets) == 32
    assert "14" in buckets
    assert "1" in buckets


def test_get_dsa_rankings():
    response = client.get("/api/matches/football4/rankings")
    assert response.status_code == 200
    rankings = response.json()
    assert len(rankings) == 32
    # Ensure rank 1 to 32
    ranks = [r["dsa_rank"] for r in rankings]
    assert ranks == list(range(1, 33))
    # Verify no tied scores
    scores = [r["dsa_activity_score_0_100"] for r in rankings]
    assert len(set(scores)) == 32


def test_dsa_lookup():
    response = client.get("/api/matches/football4/lookup/14")
    assert response.status_code == 200
    item = response.json()
    assert item["player_id"] == 14
    assert item["collective_distance_meters"] == 206.36
    assert item["collective_touches"] == 8


def test_get_possession():
    response = client.get("/api/matches/football4/possession")
    assert response.status_code == 200
    poss = response.json()
    assert poss["possession_split"]["team_a_pct"] == 80.1
    assert poss["possession_split"]["team_b_pct"] == 19.9
    assert poss["touches_summary"]["total_collective_touches"] == 171


def test_get_interaction_graph():
    response = client.get("/api/matches/football4/interaction-graph")
    assert response.status_code == 200
    graph = response.json()
    assert graph["total_nodes"] == 32
    assert graph["total_edges"] >= 1
    assert len(graph["edges"]) == graph["total_edges"]


def test_get_rag_document():
    response = client.get("/api/matches/football4/rag-document")
    assert response.status_code == 200
    doc = response.json()
    assert doc["metadata"]["schema_version"] == "3.0"
    assert len(doc["player_cards"]) == 32


def test_rag_tactical_query():
    # Test player evaluation query
    res_player = client.post(
        "/api/matches/football4/tactical-query",
        json={"query": "How did Player 14 perform?"},
    )
    assert res_player.status_code == 200
    ans_p = res_player.json()
    assert "Player 14" in ans_p["answer"]
    assert "206.36" in ans_p["answer"]

    # Test possession query
    res_poss = client.post(
        "/api/matches/football4/tactical-query",
        json={"query": "Analyze team possession and touches dominance"},
    )
    assert res_poss.status_code == 200
    ans_poss = res_poss.json()
    assert "80.1%" in ans_poss["answer"]
    assert "Team A" in ans_poss["answer"]


def test_get_manifest():
    response = client.get("/api/matches/football4/manifest")
    assert response.status_code == 200
    manifest = response.text
    assert "ATHLETIQ CENTRALIZED PIPELINE MANIFEST" in manifest
    assert "zero_wastage" in manifest.lower() or "zero wastage" in manifest.lower()
