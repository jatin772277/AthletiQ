"""Pydantic schemas for AthletiQ API data models."""

from __future__ import annotations
from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str
    environment: str
    default_match: str
    database: str = "unknown"
    database_ready: bool = False
    pipeline_mode: str = "local"
    data_retention_policy: str = "zero_wastage_collective_metrics_verified"


class MatchMetadata(BaseModel):
    schema_version: str
    document_type: str
    video_id: str
    video_filename: str
    duration_analyzed_seconds: float
    total_frames_analyzed: int
    pitch_dimensions: str
    stitching_architecture: str
    metric_preservation_policy: str
    outlier_clipping: str


class TeamAnalytics(BaseModel):
    squad_size: int
    total_collective_distance_meters: float
    total_collective_sprints: int
    total_collective_touches: int
    possession_pct: float
    average_player_collective_distance_m: float


class MatchSummary(BaseModel):
    match_id: str
    video_filename: str
    duration_seconds: float
    total_frames: int
    pitch_dimensions: str
    total_active_players: int
    team_a: TeamAnalytics
    team_b: TeamAnalytics
    match_total_distance_meters: float
    match_total_sprints: int
    match_total_touches: int
    total_raw_observations_preserved: int
    observation_retention_pct: float
    metric_wastage_pct: float = 0.0


class IntensityZones(BaseModel):
    walking: float
    jogging: float
    running: float
    sprinting: float


class AbsorbedTracklet(BaseModel):
    raw_track_id: int
    role: str = "absorbed_tracklet"
    observations: int
    frame_span: list[int]
    distance_meters: float
    average_speed_mps: float
    average_speed_kmh: float
    peak_speed_kmh: float
    sprint_count: int
    sprint_distance_meters: float
    touches_detected: int
    intensity_zones_meters: IntensityZones


class RemovedPlayersBucket(BaseModel):
    removed_players_count: int
    absorbed_observations: int
    absorbed_distance_meters: float
    absorbed_sprint_count: int
    absorbed_touches: int
    absorbed_tracklets: list[AbsorbedTracklet] = Field(default_factory=list)


class PrimaryTrajectoryKinematics(BaseModel):
    observations: int
    distance_meters: float
    sprint_count: int
    touches_detected: int


class CollectiveMetrics(BaseModel):
    total_observations: int
    total_distance_meters: float
    average_speed_mps: float
    average_speed_kmh: float
    peak_speed_kmh: float
    sprint_count: int
    sprint_distance_meters: float
    total_touches_detected: int
    intensity_zones_meters: IntensityZones
    observation_retention_pct: float = 100.0


class SpatialDistribution(BaseModel):
    pitch_centroid_meters: list[float]
    defensive_third_pct: float
    middle_third_pct: float
    attacking_third_pct: float


class PlayerCard(BaseModel):
    player_id: int
    team: str
    team_confidence: float
    dsa_rank: int
    dsa_activity_index: float
    ml_performance_score: float
    tactical_profile: str
    primary_trajectory_kinematics: PrimaryTrajectoryKinematics
    removed_players_bucket: RemovedPlayersBucket
    collective_metrics: CollectiveMetrics
    spatial_distribution: SpatialDistribution


class DSARankingItem(BaseModel):
    dsa_rank: int
    canonical_id: int
    team_label: str
    collective_distance_meters: float
    collective_sprint_count: int
    collective_touches: int
    collective_average_speed_mps: Optional[float] = None
    dsa_activity_score_0_100: float


class PossessionResponse(BaseModel):
    match_id: str
    possession_split: dict[str, float]
    touches_summary: dict[str, int]
    player_touches: dict[str, int]


class InteractionEdge(BaseModel):
    player_u: int
    player_v: int
    proximity_weight: int


class InteractionGraphResponse(BaseModel):
    match_id: str
    total_nodes: int
    total_edges: int
    edges: list[InteractionEdge]


class RAGQueryRequest(BaseModel):
    query: str
    player_id: Optional[int] = None
    team: Optional[str] = None


class RAGQueryResponse(BaseModel):
    query: str
    answer: str
    sources_used: list[str]
    context_data: dict[str, Any]


class RegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    email: str = Field(max_length=254)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(max_length=128)


class UserOut(BaseModel):
    id: str
    name: str
    email: str
    created_at: datetime


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class VideoInfo(BaseModel):
    fps: float
    frame_count: int
    duration_seconds: float
    width: int
    height: int


class JobOut(BaseModel):
    id: str
    title: str
    original_filename: str
    status: str
    progress: float
    phase: str
    error: Optional[str] = None
    log_tail: list[str] = Field(default_factory=list)
    video: VideoInfo
    start_frame: int
    max_frames: int
    size_bytes: int
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
