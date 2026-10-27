"""Pipeline Execution & Manifest API routes."""

from __future__ import annotations
from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from backend.app.services.data_loader import data_loader

router = APIRouter(prefix="/api/matches", tags=["Pipeline Execution"])


@router.get("/{match_id}/manifest", response_class=PlainTextResponse, summary="Get pipeline execution manifest text")
def get_manifest(match_id: str) -> str:
    manifest = data_loader.get_manifest_text(match_id)
    if not manifest:
        raise HTTPException(status_code=404, detail=f"Manifest not found for match '{match_id}'")
    return manifest
