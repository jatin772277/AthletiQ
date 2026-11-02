"""DSA Engine (PriorityQueue & HashMap) API routes."""

from __future__ import annotations
from typing import Any
from fastapi import APIRouter, HTTPException
from backend.app.schemas.models import DSARankingItem
from backend.app.services.data_loader import data_loader

router = APIRouter(prefix="/api/matches", tags=["DSA Engine"])


@router.get("/{match_id}/rankings", response_model=list[DSARankingItem], summary="Get PriorityQueue activity rankings (1 to 32)")
def get_dsa_rankings(match_id: str):
    rankings = data_loader.get_dsa_rankings(match_id)
    if not rankings:
        raise HTTPException(status_code=404, detail=f"No rankings found for match '{match_id}'")
    return rankings


@router.get("/{match_id}/lookup/{player_id}", summary="Get O(1) DSA HashMap lookup item for player")
def get_dsa_lookup(match_id: str, player_id: int) -> dict[str, Any]:
    item = data_loader.get_dsa_lookup(match_id, player_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"Player ID {player_id} not found in DSA lookup for match '{match_id}'")
    return item
