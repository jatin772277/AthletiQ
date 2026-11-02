"""Removed Players Bucket API routes."""

from __future__ import annotations
from typing import Any
from fastapi import APIRouter, HTTPException
from backend.app.services.data_loader import data_loader

router = APIRouter(prefix="/api/matches", tags=["Removed Players Bucket"])


@router.get("/{match_id}/buckets", summary="Get all removed player buckets across all 32 players")
def get_all_buckets(match_id: str) -> dict[str, Any]:
    buckets = data_loader.get_removed_players_bucket(match_id)
    if not buckets:
        raise HTTPException(status_code=404, detail=f"No buckets found for match '{match_id}'")
    return buckets


@router.get("/{match_id}/buckets/{player_id}", summary="Get the absorbed tracklets bucket for a responsible player")
def get_player_bucket(match_id: str, player_id: int) -> dict[str, Any]:
    bucket = data_loader.get_removed_players_bucket(match_id, player_id=player_id)
    if not bucket:
        raise HTTPException(status_code=404, detail=f"No bucket found for player {player_id} in match '{match_id}'")
    return bucket
