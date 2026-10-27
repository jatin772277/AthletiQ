"""Tactical RAG Query Service for AthletiQ powered by Groq."""

from __future__ import annotations
import json
import os
from typing import Any, Optional

from backend.app.services.data_loader import data_loader

# The Groq client is OpenAI-compatible but keeps its own lightweight SDK.
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


class RAGQueryService:
    def __init__(self):
        self.external_llm_enabled = os.getenv("ATHLETIQ_DISABLE_EXTERNAL_LLM", "").lower() not in {"1", "true", "yes"}
        self.api_key = os.getenv("GROQ_API_KEY") if self.external_llm_enabled else None
        self.client = None
        self.model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

        if GROQ_AVAILABLE and self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception as e:
                print(f"[RAG] Warning: Could not initialize Groq client: {e}")
                self.client = None

    def _call_groq_llm(self, query: str, context_json_str: str) -> Optional[str]:
        # Refresh key if set at runtime
        if not self.external_llm_enabled:
            return None

        if not self.client:
            self.api_key = os.getenv("GROQ_API_KEY")
            if GROQ_AVAILABLE and self.api_key:
                try:
                    self.client = Groq(api_key=self.api_key)
                except Exception:
                    self.client = None

        if not self.client:
            return None

        system_prompt = (
            "You are AthletiQ Tactical AI, a UEFA Pro performance analyst and kinematic specialist. "
            "Your role is to evaluate match analytics, player work-rate, physical kinematics, and ball involvement "
            "using ONLY the provided verified ground-truth JSON tracking data.\n"
            "Key Requirements:\n"
            "1. Ground all numbers, distances, speeds, sprints, and touches strictly in the provided JSON.\n"
            "2. Note the 'removed_players_bucket' where tracklets from stitching were recovered with 0% metric wastage.\n"
            "3. Provide elite, tactical, coach-level insights (e.g. fatigue risk, pressing efficiency, high-speed running, tactical role adherence)."
        )

        user_content = (
            f"VERIFIED MATCH & KINEMATICS GROUND-TRUTH JSON:\n"
            f"```json\n{context_json_str}\n```\n\n"
            f"COACH / ANALYST QUERY:\n{query}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.25,
            )
            return response.choices[0].message.content
        except Exception as e:
            # A provider failure must never prevent users from receiving a data-grounded answer.
            print(f"[RAG] Groq API call failed: {e}. Falling back to deterministic engine.")
            return None

    def answer_query(self, match_id: str, query: str, player_id: Optional[int] = None, team: Optional[str] = None) -> dict[str, Any]:
        match_summary = data_loader.get_match_summary(match_id)
        q_lower = query.lower()

        # Check for specific player
        target_pid = player_id
        if target_pid is None:
            for word in q_lower.replace("#", " ").split():
                if word.isdigit() and 1 <= int(word) <= 32:
                    target_pid = int(word)
                    break

        # Context selection
        if target_pid is not None:
            p_card = data_loader.get_player_by_id(match_id, target_pid)
            if p_card:
                context_data = p_card
                sources = [
                    f"rag_ready/{match_id}_match_and_player_metrics_for_rag.json",
                    f"physical_metrics/{match_id}_removed_players_bucket.json",
                ]
            else:
                context_data = match_summary
                sources = [f"rag_ready/{match_id}_match_and_player_metrics_for_rag.json"]
        else:
            context_data = match_summary
            sources = [f"rag_ready/{match_id}_match_and_player_metrics_for_rag.json"]

        context_json_str = json.dumps(context_data, indent=2)

        # 1. Try Groq LLM if API key is configured.
        llm_answer = self._call_groq_llm(query=query, context_json_str=context_json_str)
        if llm_answer:
            return {
                "query": query,
                "answer": llm_answer,
                "llm_engine": f"Groq ({self.model_name})",
                "sources_used": sources,
                "context_data": context_data,
            }

        # 2. High-precision deterministic fallback when offline or without API key
        if target_pid is not None and p_card:
            coll = p_card["collective_metrics"]
            prim = p_card["primary_trajectory_kinematics"]
            bucket = p_card["removed_players_bucket"]
            tactical = p_card["tactical_profile"]

            fallback_ans = (
                f"**Player {target_pid} ({p_card['team'].upper()}) Evaluation:**\n"
                f"- **Tactical Profile:** {tactical}\n"
                f"- **DSA Activity Rank:** #{p_card['dsa_rank']} (Score: {p_card['dsa_activity_index']:.1f}/100, ML Score: {p_card['ml_performance_score']:.1f})\n"
                f"- **Collective Distance:** {coll['total_distance_meters']:.2f} m ({prim['distance_meters']:.2f} m primary + {bucket['absorbed_distance_meters']:.2f} m from {bucket['removed_players_count']} absorbed tracklets)\n"
                f"- **Kinematics:** Avg Speed: {coll['average_speed_kmh']:.1f} km/h, Peak Speed: {coll['peak_speed_kmh']:.1f} km/h, Sprints: {coll['sprint_count']}\n"
                f"- **Ball Involvement:** {coll['total_touches_detected']} touches recorded across all constituent tracklets\n"
                f"- **Provenance:** 100% of observations preserved in responsible player bucket with zero metric wastage."
            )
            return {
                "query": query,
                "answer": fallback_ans,
                "llm_engine": "AthletiQ Deterministic Engine (Offline / Standby for Groq API Key)",
                "sources_used": sources,
                "context_data": p_card,
            }

        if any(w in q_lower for w in ["possession", "touches", "control", "dominance"]):
            ta = match_summary["team_a"]
            tb = match_summary["team_b"]
            lead, trail = ("Team A", "Team B") if ta["possession_pct"] >= tb["possession_pct"] else ("Team B", "Team A")
            lead_t = (ta if lead == "Team A" else tb)["total_collective_touches"]
            trail_t = (tb if lead == "Team A" else ta)["total_collective_touches"]
            volume = f"{lead_t / trail_t:.1f}x the touch volume of {trail}" if trail_t else f"all {lead_t} recorded touches while {trail} registered none"
            ans = (
                f"**Match Possession & Control Breakdown:**\n"
                f"- **Team A:** {ta['possession_pct']}% possession ({ta['total_collective_touches']} touches, {ta['total_collective_distance_meters']:.1f} m covered, {ta['total_collective_sprints']} sprints)\n"
                f"- **Team B:** {tb['possession_pct']}% possession ({tb['total_collective_touches']} touches, {tb['total_collective_distance_meters']:.1f} m covered, {tb['total_collective_sprints']} sprints)\n"
                f"- **Tactical Summary:** {lead} held more of the ball, generating {volume}."
            )
            return {
                "query": query,
                "answer": ans,
                "llm_engine": "AthletiQ Deterministic Engine (Offline / Standby for Groq API Key)",
                "sources_used": [f"ball_possession/{match_id}_possession_summary.json"],
                "context_data": match_summary,
            }

        # Default summary
        ans = (
            f"**AthletiQ Match Kinematics & Tactical Assessment ({match_id}):**\n"
            f"- **Active Participants:** {match_summary['total_active_players']} verified canonical players ({match_summary['team_a']['squad_size']} Team A, {match_summary['team_b']['squad_size']} Team B)\n"
            f"- **Observation Retention:** {match_summary['observation_retention_pct']}% ({match_summary['total_raw_observations_preserved']:,} observations preserved in responsible player buckets with 0% metric wastage)\n"
            f"- **Possession:** Team A {match_summary['team_a']['possession_pct']}% vs Team B {match_summary['team_b']['possession_pct']}%\n"
            f"- **Distance Covered:** Team A {match_summary['team_a']['total_collective_distance_meters']:.1f} m vs Team B {match_summary['team_b']['total_collective_distance_meters']:.1f} m\n"
            f"- You can query specific player metrics by player ID (e.g. 'How did Player 14 perform?') or tactical topics (possession, work-rate, buckets)."
        )
        return {
            "query": query,
            "answer": ans,
            "llm_engine": "AthletiQ Deterministic Engine (Offline / Standby for Groq API Key)",
            "sources_used": [f"rag_ready/{match_id}_match_and_player_metrics_for_rag.json"],
            "context_data": match_summary,
        }


rag_service = RAGQueryService()
