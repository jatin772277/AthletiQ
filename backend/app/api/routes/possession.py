"""Ball Possession and Touch Analytics API routes."""

from __future__ import annotations
from fastapi import APIRouter, HTTPException
from backend.app.schemas.models import PossessionResponse
from backend.app.services.data_loader import data_loader

router = APIRouter(prefix="/api/matches", tags=["Ball Possession"])


@router.get("/{match_id}/possession", response_model=PossessionResponse, summary="Get ball possession split & touch breakdown")
def get_possession(match_id: str):
    try:
        return data_loader.get_possession_summary(match_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Possession data not found for match '{match_id}': {str(e)}")
