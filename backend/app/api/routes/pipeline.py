"""Pipeline Execution & Manifest API routes."""

from __future__ import annotations
from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from backend.app.services.data_loader import data_loader

router = APIRouter(prefix="/api/matches", tags=["Pipeline Execution"])


@router.get("/{match_id}/manifest", response_class=PlainTextResponse, summary="Get pipeline execution manifest text")
def get_manifest(match_id: str) -> str:
    manifest = data_loader.get_manifest_text(match_id)
    return manifest or f"Pipeline execution manifest for match '{match_id}'."
