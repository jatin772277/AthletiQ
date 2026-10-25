"""Matches API routes."""

from __future__ import annotations
from typing import Any
from fastapi import APIRouter, HTTPException
from backend.app.schemas.models import MatchSummary
from backend.app.services.data_loader import data_loader

router = APIRouter(prefix="/api/matches", tags=["Matches"])


@router.get("", summary="List all ingested matches")
def list_matches() -> list[dict[str, Any]]:
    return data_loader.get_available_matches()


@router.get("/{match_id}/summary", response_model=MatchSummary, summary="Get comprehensive match summary")
def get_match_summary(match_id: str):
    try:
        return data_loader.get_match_summary(match_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Match '{match_id}' not found: {str(e)}")


@router.get("/{match_id}/report", summary="Get executive match scouting report in Markdown")
def get_match_scouting_report(match_id: str):
    from backend.app.services.report_generator import report_generator
    try:
        report_text = report_generator.generate_markdown_report(match_id)
        return {
            "match_id": match_id,
            "report_format": "markdown",
            "report_content": report_text,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate scouting report: {str(e)}")

