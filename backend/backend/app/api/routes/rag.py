"""RAG Knowledge Document and LLM Query Assistant API routes."""

from __future__ import annotations
from typing import Any
from fastapi import APIRouter, HTTPException
from backend.app.schemas.models import RAGQueryRequest, RAGQueryResponse
from backend.app.services.data_loader import data_loader
from backend.app.services.rag_service import rag_service

router = APIRouter(prefix="/api/matches", tags=["RAG Document & Tactical Assistant"])


@router.get("/{match_id}/rag-document", summary="Get full RAG-ready match JSON document")
def get_rag_document(match_id: str) -> dict[str, Any]:
    doc = data_loader.get_rag_document(match_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"RAG document not found for match '{match_id}'")
    return doc


@router.post("/{match_id}/tactical-query", response_model=RAGQueryResponse, summary="Query tactical coach assistant on match analytics")
def query_tactical_rag(match_id: str, payload: RAGQueryRequest):
    try:
        return rag_service.answer_query(
            match_id=match_id,
            query=payload.query,
            player_id=payload.player_id,
            team=payload.team,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query execution error: {str(e)}")
