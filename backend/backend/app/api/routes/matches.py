"""Matches API routes."""

from __future__ import annotations
from typing import Any
from fastapi import APIRouter, Depends, HTTPException
from backend.app import db
from backend.app.config import settings
from backend.app.schemas.models import MatchSummary
from backend.app.security import get_current_user
from backend.app.services.data_loader import data_loader

router = APIRouter(prefix="/api/matches", tags=["Matches"])


@router.get("", summary="List demo matches plus the signed-in user's analyses")
def list_matches(user: dict[str, Any] = Depends(get_current_user)) -> list[dict[str, Any]]:
    demos = [
        {**m, "title": f"Demo · {m['match_id']}", "is_demo": True}
        for m in data_loader.get_available_matches()
        if m["match_id"] in settings.demo_match_ids
    ]
    own = [
        {
            "match_id": j["_id"],
            "title": j["title"],
            "status": j["status"].upper(),
            "has_rag_package": j["status"] == "completed",
            "has_manifest": j["status"] == "completed",
            "is_demo": False,
            "created_at": j["created_at"],
        }
        for j in db.jobs().find({"owner_id": user["_id"]}).sort("created_at", -1)
    ]
    return own + demos


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

