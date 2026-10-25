export type TeamId = "team_a" | "team_b";

export interface MatchListItem {
  match_id: string;
  folder_name: string;
  status: string;
  has_rag_package: boolean;
  has_manifest: boolean;
}

export interface TeamAnalytics {
  squad_size: number;
  total_collective_distance_meters: number;
  total_collective_sprints: number;
  total_collective_touches: number;
  possession_pct: number;
  average_player_collective_distance_m: number;
}

export interface MatchSummary {
  match_id: string;
  video_filename: string;
  duration_seconds: number;
  total_frames: number;
  pitch_dimensions: string;
  total_active_players: number;
  team_a: TeamAnalytics;
  team_b: TeamAnalytics;
  match_total_distance_meters: number;
  match_total_sprints: number;
  match_total_touches: number;
  total_raw_observations_preserved: number;
  observation_retention_pct: number;
  metric_wastage_pct: number;
}

export interface IntensityZones {
  walking: number;
  jogging: number;
  running: number;
  sprinting: number;
}

export interface AbsorbedTracklet {
  raw_track_id: number;
  observations: number;
  frame_span: [number, number];
  distance_meters: number;
  average_speed_kmh: number;
  peak_speed_kmh: number;
  sprint_count: number;
  touches_detected: number;
  intensity_zones_meters: IntensityZones;
}

export interface PlayerCard {
  player_id: number;
  team: TeamId;
  team_confidence: number;
  dsa_rank: number;
  dsa_activity_index: number;
  ml_performance_score: number;
  tactical_profile: string;
  primary_trajectory_kinematics: {
    observations: number;
    distance_meters: number;
    sprint_count: number;
    touches_detected: number;
  };
  removed_players_bucket: {
    removed_players_count: number;
    absorbed_observations: number;
    absorbed_distance_meters: number;
    absorbed_sprint_count: number;
    absorbed_touches: number;
    absorbed_tracklets: AbsorbedTracklet[];
  };
  collective_metrics: {
    total_observations: number;
    total_distance_meters: number;
    average_speed_mps: number;
    average_speed_kmh: number;
    peak_speed_kmh: number;
    sprint_count: number;
    sprint_distance_meters: number;
    total_touches_detected: number;
    intensity_zones_meters: IntensityZones;
    observation_retention_pct: number;
  };
  spatial_distribution: {
    pitch_centroid_meters: [number, number];
    defensive_third_pct: number;
    middle_third_pct: number;
    attacking_third_pct: number;
  };
}

export interface RankingItem {
  dsa_rank: number;
  canonical_id: number;
  team_label: TeamId;
  collective_distance_meters: number;
  collective_sprint_count: number;
  collective_touches: number;
  collective_average_speed_mps?: number | null;
  dsa_activity_score_0_100: number;
}

export interface Possession {
  match_id: string;
  possession_split: Record<string, number>;
  touches_summary: Record<string, number>;
  player_touches: Record<string, number>;
}

export interface InteractionEdge {
  player_u: number;
  player_v: number;
  proximity_weight: number;
}

export interface InteractionGraph {
  match_id: string;
  total_nodes: number;
  total_edges: number;
  edges: InteractionEdge[];
}

export interface TacticalAnswer {
  query: string;
  answer: string;
  sources_used: string[];
}

export interface MatchBundle {
  summary: MatchSummary;
  players: PlayerCard[];
  rankings: RankingItem[];
  possession: Possession;
  graph: InteractionGraph;
  manifest: string;
}
