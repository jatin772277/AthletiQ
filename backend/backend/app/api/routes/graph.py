"""Player Interaction Graph API routes."""

from __future__ import annotations
from fastapi import APIRouter, HTTPException
from backend.app.schemas.models import InteractionGraphResponse
from backend.app.services.data_loader import data_loader

router = APIRouter(prefix="/api/matches", tags=["Player Interaction Graph"])


@router.get("/{match_id}/interaction-graph", response_model=InteractionGraphResponse, summary="Get player interaction network nodes & proximity edges")
def get_interaction_graph(match_id: str):
    try:
        return data_loader.get_interaction_graph(match_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Interaction graph not found for match '{match_id}': {str(e)}")
