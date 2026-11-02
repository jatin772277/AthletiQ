"""Players API routes."""

from __future__ import annotations
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from backend.app.schemas.models import PlayerCard
from backend.app.services.data_loader import data_loader

router = APIRouter(prefix="/api/matches", tags=["Players"])


@router.get("/{match_id}/players", response_model=list[PlayerCard], summary="List all 32 players for a match")
def get_players(
    match_id: str,
    team: Optional[str] = Query(None, description="Filter by team ('team_a' or 'team_b')"),
):
    cards = data_loader.get_player_cards(match_id, team=team)
    if not cards:
        raise HTTPException(status_code=404, detail=f"No players found for match '{match_id}'")
    return cards


@router.get("/{match_id}/players/{player_id}", response_model=PlayerCard, summary="Get full player card by ID")
def get_player(match_id: str, player_id: int):
    card = data_loader.get_player_by_id(match_id, player_id)
    if not card:
        raise HTTPException(status_code=404, detail=f"Player ID {player_id} not found in match '{match_id}'")
    return card
