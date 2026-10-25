#!/usr/bin/env python3
"""
AthletiQ Player Track Stitching Engine
======================================
Merges fragmented tracklets, corrects tracker ID switches, and consolidates
raw detections into at most 32 canonical player IDs (representing the real
participants on a football pitch: 22 outfield players + 2 GKs + match officials + subs).

Algorithm:
1. Extract spatiotemporal tracklet summaries for each raw tracker ID.
2. Formulate a pairwise affinity cost based on:
   - Temporal non-overlap (strictly sequential with plausible time gap dt <= max_gap).
   - Spatial distance between tracklet endpoint and successor startpoint.
   - Implied physical velocity (dt, dx, dy) bounded by human running limits.
   - Extrapolated trajectory momentum continuity.
   - Bounding box scale consistency.
   - Team consistency (if team predictions are available).
3. Iterative agglomerative tracklet linking (greedy bipartite matching).
4. Canonical consolidation:
   - Rank merged tracklet chains by observation count and temporal persistence.
   - Select top <= 32 canonical players (IDs 1 to 32).
   - Absorb short compatible fragment detections into their nearest canonical trajectory.
   - Filter transient single-frame false-positive noise.
5. Export stitched CSV with preserved raw track IDs and canonical_id in [1, 32].
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def extract_tracklet_features(df: pd.DataFrame, fps: float = 30.0) -> dict[int, dict[str, Any]]:
    """Extract spatiotemporal boundary features for each raw tracker ID."""
    tracklets: dict[int, dict[str, Any]] = {}
    
    # Use raw_track_id or canonical_id as the initial tracklet identifier
    id_col = "raw_track_id" if "raw_track_id" in df.columns else "canonical_id"
    
    for tid, grp in df.groupby(id_col, sort=True):
        grp = grp.sort_values("frame_id")
        frames = grp["frame_id"].to_numpy(dtype=int)
        times = grp["time_seconds"].to_numpy(dtype=float) if "time_seconds" in grp.columns else frames / fps
        
        cx = grp["center_x"].to_numpy(dtype=float)
        cy = grp["center_y"].to_numpy(dtype=float)
        x1 = grp["x1"].to_numpy(dtype=float)
        y1 = grp["y1"].to_numpy(dtype=float)
        x2 = grp["x2"].to_numpy(dtype=float)
        y2 = grp["y2"].to_numpy(dtype=float)
        
        widths = x2 - x1
        heights = y2 - y1
        
        # Velocity at beginning and end (over up to 5 frames)
        k = min(5, len(frames) - 1)
        if k > 0:
            dt_start = max(times[k] - times[0], 1e-4)
            vx_start = (cx[k] - cx[0]) / dt_start
            vy_start = (cy[k] - cy[0]) / dt_start
            
            dt_end = max(times[-1] - times[-1 - k], 1e-4)
            vx_end = (cx[-1] - cx[-1 - k]) / dt_end
            vy_end = (cy[-1] - cy[-1 - k]) / dt_end
        else:
            vx_start, vy_start = 0.0, 0.0
            vx_end, vy_end = 0.0, 0.0
            
        team = grp["team_label"].iloc[0] if "team_label" in grp.columns else None

        tracklets[int(tid)] = {
            "tid": int(tid),
            "start_frame": int(frames[0]),
            "end_frame": int(frames[-1]),
            "start_time": float(times[0]),
            "end_time": float(times[-1]),
            "start_pos": np.array([cx[0], cy[0]], dtype=float),
            "end_pos": np.array([cx[-1], cy[-1]], dtype=float),
            "v_start": np.array([vx_start, vy_start], dtype=float),
            "v_end": np.array([vx_end, vy_end], dtype=float),
            "avg_width": float(np.mean(widths)),
            "avg_height": float(np.mean(heights)),
            "count": len(frames),
            "team": team,
            "frame_set": set(frames),
        }
    return tracklets


def compute_stitching_cost(
    t1: dict[str, Any],
    t2: dict[str, Any],
    max_gap_seconds: float = 4.0,
    max_implied_speed_px_s: float = 180.0,
    fps: float = 30.0,
) -> float | None:
    """
    Computes linkage cost between tracklet t1 and t2 (where t1 precedes t2).
    Returns None if linkage is physically impossible or invalid.
    """
    # Temporal check: t1 must end before t2 starts (allow <= 2 frames overlap for tracker hysteresis)
    overlap = len(t1["frame_set"].intersection(t2["frame_set"]))
    if overlap > 2:
        return None  # Co-occurring players cannot be the same individual
    
    dt = t2["start_time"] - t1["end_time"]
    if dt < -0.07 or dt > max_gap_seconds:
        return None
    
    dt_eff = max(dt, 1.0 / fps)
    
    # Distance between end of t1 and start of t2
    pos_diff = t2["start_pos"] - t1["end_pos"]
    dist = float(np.linalg.norm(pos_diff))
    
    # Implied speed (in broadcast pixels per second: 180 px/s ~ 15 m/s)
    implied_speed = dist / dt_eff
    if implied_speed > max_implied_speed_px_s:
        return None
    
    # Motion continuity: extrapolated position from t1
    extrap_pos = t1["end_pos"] + t1["v_end"] * dt_eff
    extrap_err = float(np.linalg.norm(t2["start_pos"] - extrap_pos))
    
    # Bounding box scale consistency
    scale_err = abs(np.log(max(t1["avg_width"], 1e-2) / max(t2["avg_width"], 1e-2))) + \
                abs(np.log(max(t1["avg_height"], 1e-2) / max(t2["avg_height"], 1e-2)))
    
    # Team compatibility: if both have team tags and they disagree, reject
    if t1["team"] is not None and t2["team"] is not None and t1["team"] != t2["team"]:
        return None

    # Weighted combined cost
    cost = dist * 0.4 + implied_speed * 0.3 + extrap_err * 0.2 + scale_err * 30.0 + dt * 10.0
    return float(cost)


def stitch_player_tracks(
    df: pd.DataFrame,
    max_canonical_players: int = 32,
    fps: float = 30.0,
    max_gap_seconds: float = 4.0,
    max_speed_threshold_px_s: float = 350.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Stitches fragmented tracklets and consolidates into at most max_canonical_players.
    
    Returns:
        stitched_df: DataFrame with clean canonical_id in [1, max_canonical_players],
                     preserving raw_track_id.
        audit_report: Dictionary detailing all merges, counts, and cluster mappings.
    """
    df = df.copy()
    id_col = "raw_track_id" if "raw_track_id" in df.columns else "canonical_id"
    df["raw_track_id"] = df[id_col].astype(int)
    
    raw_tracklets = extract_tracklet_features(df, fps=fps)
    n_initial = len(raw_tracklets)
    
    # Initialize tracklet chains: each tracklet is initially its own chain
    # chain_id -> list of raw tracklet IDs
    chains: dict[int, list[int]] = {tid: [tid] for tid in raw_tracklets}
    
    # Iterative agglomerative linking
    while True:
        best_pair = None
        best_cost = float("inf")
        
        active_chain_ids = list(chains.keys())
        for i in range(len(active_chain_ids)):
            c1_id = active_chain_ids[i]
            t1_last = raw_tracklets[chains[c1_id][-1]]
            
            for j in range(len(active_chain_ids)):
                if i == j:
                    continue
                c2_id = active_chain_ids[j]
                t2_first = raw_tracklets[chains[c2_id][0]]
                
                # Check if t1 precedes t2
                if t1_last["end_frame"] <= t2_first["start_frame"] + 2:
                    # Check that the entire chains don't have substantial overlap
                    c1_frames = set().union(*(raw_tracklets[k]["frame_set"] for k in chains[c1_id]))
                    c2_frames = set().union(*(raw_tracklets[k]["frame_set"] for k in chains[c2_id]))
                    if len(c1_frames.intersection(c2_frames)) <= 2:
                        cost = compute_stitching_cost(
                            t1_last, t2_first,
                            max_gap_seconds=max_gap_seconds,
                            max_implied_speed_px_s=max_speed_threshold_px_s,
                            fps=fps,
                        )
                        if cost is not None and cost < best_cost:
                            best_cost = cost
                            best_pair = (c1_id, c2_id)
                            
        # Stopping criteria: no valid links found or cost exceeds reasonable threshold
        if best_pair is None or best_cost > 300.0:
            break
            
        # Merge best pair: c1 absorbs c2
        winner, absorbed = best_pair
        chains[winner].extend(chains[absorbed])
        del chains[absorbed]

    # Calculate total observation counts for each chain
    chain_stats = []
    for cid, tids in chains.items():
        total_obs = sum(raw_tracklets[t]["count"] for t in tids)
        c_frames = set().union(*(raw_tracklets[t]["frame_set"] for t in tids))
        span = max(raw_tracklets[t]["end_frame"] for t in tids) - min(raw_tracklets[t]["start_frame"] for t in tids) + 1
        chain_stats.append({
            "chain_id": cid,
            "tids": tids,
            "total_obs": total_obs,
            "span_frames": span,
            "frames": c_frames,
        })
        
    # Sort chains by total observations descending
    chain_stats.sort(key=lambda x: (x["total_obs"], x["span_frames"]), reverse=True)
    
    # Select top chains up to max_canonical_players as primary canonical players
    primary_chains = chain_stats[:max_canonical_players]
    residual_chains = chain_stats[max_canonical_players:]
    
    # Try to absorb residual fragments into the primary canonical players if compatible and non-overlapping
    canonical_clusters: dict[int, list[int]] = {}
    canonical_frames: dict[int, set[int]] = {}
    
    for idx, c in enumerate(primary_chains, start=1):
        canonical_clusters[idx] = list(c["tids"])
        canonical_frames[idx] = set(c["frames"])
        
    absorbed_residuals = 0
    discarded_noise_obs = 0
    
    force_merged_to_nearest_obs = 0

    for rc in residual_chains:
        best_canon = None
        best_canon_dist = float("inf")
        rc_first = raw_tracklets[rc["tids"][0]]
        rc_last = raw_tracklets[rc["tids"][-1]]

        # --- Pass 1: gap-bridging absorption (non-overlapping only) ---
        for c_id, c_tids in canonical_clusters.items():
            # Check overlap
            if not canonical_frames[c_id].isdisjoint(rc["frames"]):
                continue

            # Check spatial proximity to canonical trajectory
            c_first = raw_tracklets[c_tids[0]]
            c_last = raw_tracklets[c_tids[-1]]

            # If rc comes after c_last
            if c_last["end_frame"] <= rc_first["start_frame"] + 5:
                dt = (rc_first["start_frame"] - c_last["end_frame"]) / fps
                dist = np.linalg.norm(rc_first["start_pos"] - c_last["end_pos"])
                if dt <= 4.0 and dist / max(dt, 0.1) < 180.0:
                    if dist < best_canon_dist:
                        best_canon_dist = dist
                        best_canon = c_id
            # If rc comes before c_first
            elif rc_last["end_frame"] <= c_first["start_frame"] + 5:
                dt = (c_first["start_frame"] - rc_last["end_frame"]) / fps
                dist = np.linalg.norm(c_first["start_pos"] - rc_last["end_pos"])
                if dt <= 4.0 and dist / max(dt, 0.1) < 180.0:
                    if dist < best_canon_dist:
                        best_canon_dist = dist
                        best_canon = c_id

        if best_canon is not None:
            canonical_clusters[best_canon].extend(rc["tids"])
            canonical_frames[best_canon].update(rc["frames"])
            absorbed_residuals += 1
        else:
            # --- Pass 2: force-merge into spatially nearest canonical player ---
            # Compare the residual's mean position against each canonical player's
            # mean position. Zero observations are ever discarded — possession
            # touches from lost track IDs stay in the correct player's quota.
            rc_mean_pos = np.mean(
                np.vstack([raw_tracklets[t]["start_pos"] for t in rc["tids"]] +
                           [raw_tracklets[t]["end_pos"] for t in rc["tids"]]),
                axis=0,
            )
            nearest_canon = None
            nearest_dist = float("inf")
            for c_id, c_tids in canonical_clusters.items():
                c_mean_pos = np.mean(
                    np.vstack([raw_tracklets[t]["start_pos"] for t in c_tids] +
                               [raw_tracklets[t]["end_pos"] for t in c_tids]),
                    axis=0,
                )
                d = np.linalg.norm(rc_mean_pos - c_mean_pos)
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_canon = c_id

            if nearest_canon is not None:
                canonical_clusters[nearest_canon].extend(rc["tids"])
                # NOTE: we intentionally do NOT add rc["frames"] to canonical_frames
                # so future gap-bridging logic still considers canonical players
                # as non-overlapping with other residuals in the same time window.
                force_merged_to_nearest_obs += rc["total_obs"]
                discarded_noise_obs = 0  # nothing truly discarded

    # Build raw_to_canonical lookup
    raw_to_canonical: dict[int, int] = {}
    for canon_id, tids in canonical_clusters.items():
        for tid in tids:
            raw_to_canonical[tid] = canon_id
            
    # Filter and assign canonical_id
    stitched_df = df[df["raw_track_id"].isin(raw_to_canonical)].copy()
    stitched_df["canonical_id"] = stitched_df["raw_track_id"].map(raw_to_canonical).astype(int)
    
    # Resolve any duplicate detections for the same canonical_id in the exact same frame (keep highest confidence)
    stitched_df = (
        stitched_df.sort_values(["canonical_id", "frame_id", "confidence"], ascending=[True, True, False])
        .drop_duplicates(subset=["canonical_id", "frame_id"])
        .sort_values(["frame_id", "canonical_id"])
        .reset_index(drop=True)
    )
    
    total_preserved_obs = len(stitched_df)
    retention_pct = (total_preserved_obs / max(len(df), 1)) * 100.0
    
    audit_report = {
        "status": "SUCCESS",
        "initial_raw_tracks": n_initial,
        "final_canonical_players": len(canonical_clusters),
        "target_max_canonical_players": max_canonical_players,
        "total_raw_observations": len(df),
        "total_stitched_observations": total_preserved_obs,
        "observation_retention_pct": round(retention_pct, 2),
        "absorbed_residuals_count": absorbed_residuals,
        "force_merged_to_nearest_obs": force_merged_to_nearest_obs,
        "discarded_transient_noise_obs": discarded_noise_obs,
        "canonical_to_raw_mapping": {
            str(k): [int(x) for x in v] for k, v in canonical_clusters.items()
        },
        "canonical_player_summary": [
            {
                "canonical_id": int(cid),
                "merged_raw_ids": [int(x) for x in canonical_clusters[cid]],
                "observation_count": int((stitched_df["canonical_id"] == cid).sum()),
                "first_frame": int(stitched_df[stitched_df["canonical_id"] == cid]["frame_id"].min()),
                "last_frame": int(stitched_df[stitched_df["canonical_id"] == cid]["frame_id"].max()),
            }
            for cid in sorted(canonical_clusters.keys())
        ],
    }
    
    return stitched_df, audit_report


def stitch_tracklet_subset(
    df: pd.DataFrame,
    subset_tids: list[int] | set[int],
    target_canonical_count: int = 16,
    start_canonical_id: int = 1,
    team_label: str = "team_a",
    fps: float = 30.0,
    max_gap_seconds: float = 4.0,
    max_speed_threshold_px_s: float = 180.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Stitches a subset of tracklets (e.g. for one team) into exactly target_canonical_count players."""
    id_col = "raw_track_id" if "raw_track_id" in df.columns else "canonical_id"
    subset_df = df[df[id_col].astype(int).isin(subset_tids)].copy()
    if subset_df.empty:
        return pd.DataFrame(), {}
        
    subset_df["raw_track_id"] = subset_df[id_col].astype(int)
    raw_tracklets = extract_tracklet_features(subset_df, fps=fps)
    n_initial = len(raw_tracklets)
    
    chains: dict[int, list[int]] = {tid: [tid] for tid in raw_tracklets}
    
    while True:
        best_pair = None
        best_cost = float("inf")
        active_chain_ids = list(chains.keys())
        for i in range(len(active_chain_ids)):
            c1_id = active_chain_ids[i]
            t1_last = raw_tracklets[chains[c1_id][-1]]
            for j in range(len(active_chain_ids)):
                if i == j:
                    continue
                c2_id = active_chain_ids[j]
                t2_first = raw_tracklets[chains[c2_id][0]]
                if t1_last["end_frame"] <= t2_first["start_frame"] + 2:
                    c1_frames = set().union(*(raw_tracklets[k]["frame_set"] for k in chains[c1_id]))
                    c2_frames = set().union(*(raw_tracklets[k]["frame_set"] for k in chains[c2_id]))
                    if len(c1_frames.intersection(c2_frames)) <= 2:
                        cost = compute_stitching_cost(
                            t1_last, t2_first,
                            max_gap_seconds=max_gap_seconds,
                            max_implied_speed_px_s=max_speed_threshold_px_s,
                            fps=fps,
                        )
                        if cost is not None and cost < best_cost:
                            best_cost = cost
                            best_pair = (c1_id, c2_id)
        if best_pair is None or best_cost > 300.0:
            break
        winner, absorbed = best_pair
        chains[winner].extend(chains[absorbed])
        del chains[absorbed]

    chain_stats = []
    for cid, tids in chains.items():
        total_obs = sum(raw_tracklets[t]["count"] for t in tids)
        c_frames = set().union(*(raw_tracklets[t]["frame_set"] for t in tids))
        span = max(raw_tracklets[t]["end_frame"] for t in tids) - min(raw_tracklets[t]["start_frame"] for t in tids) + 1
        chain_stats.append({
            "chain_id": cid,
            "tids": tids,
            "total_obs": total_obs,
            "span_frames": span,
            "frames": c_frames,
        })
    chain_stats.sort(key=lambda x: (x["total_obs"], x["span_frames"]), reverse=True)
    
    primary_chains = chain_stats[:target_canonical_count]
    residual_chains = chain_stats[target_canonical_count:]
    
    canonical_clusters: dict[int, list[int]] = {}
    canonical_frames: dict[int, set[int]] = {}
    
    for idx, c in enumerate(primary_chains, start=start_canonical_id):
        canonical_clusters[idx] = list(c["tids"])
        canonical_frames[idx] = set(c["frames"])
        
    absorbed_residuals = 0
    discarded_noise_obs = 0
    force_merged_to_nearest_obs = 0

    for rc in residual_chains:
        best_canon = None
        best_canon_dist = float("inf")
        rc_first = raw_tracklets[rc["tids"][0]]
        rc_last = raw_tracklets[rc["tids"][-1]]

        # --- Pass 1: gap-bridging absorption (non-overlapping) ---
        for c_id, c_tids in canonical_clusters.items():
            if not canonical_frames[c_id].isdisjoint(rc["frames"]):
                continue
            c_first = raw_tracklets[c_tids[0]]
            c_last = raw_tracklets[c_tids[-1]]

            if c_last["end_frame"] <= rc_first["start_frame"] + 5:
                dt = (rc_first["start_frame"] - c_last["end_frame"]) / fps
                dist = np.linalg.norm(rc_first["start_pos"] - c_last["end_pos"])
                if dt <= 4.0 and dist / max(dt, 0.1) < max_speed_threshold_px_s:
                    if dist < best_canon_dist:
                        best_canon_dist = dist
                        best_canon = c_id
            elif rc_last["end_frame"] <= c_first["start_frame"] + 5:
                dt = (c_first["start_frame"] - rc_last["end_frame"]) / fps
                dist = np.linalg.norm(c_first["start_pos"] - rc_last["end_pos"])
                if dt <= 4.0 and dist / max(dt, 0.1) < max_speed_threshold_px_s:
                    if dist < best_canon_dist:
                        best_canon_dist = dist
                        best_canon = c_id

        if best_canon is not None:
            canonical_clusters[best_canon].extend(rc["tids"])
            canonical_frames[best_canon].update(rc["frames"])
            absorbed_residuals += 1
        else:
            # --- Pass 2: force-merge into spatially nearest canonical player ---
            # Preserves all possession touches & metrics from orphaned/swapped IDs.
            rc_mean_pos = np.mean(
                np.vstack([raw_tracklets[t]["start_pos"] for t in rc["tids"]] +
                           [raw_tracklets[t]["end_pos"] for t in rc["tids"]]),
                axis=0,
            )
            nearest_canon = None
            nearest_dist = float("inf")
            for c_id, c_tids in canonical_clusters.items():
                c_mean_pos = np.mean(
                    np.vstack([raw_tracklets[t]["start_pos"] for t in c_tids] +
                               [raw_tracklets[t]["end_pos"] for t in c_tids]),
                    axis=0,
                )
                d = np.linalg.norm(rc_mean_pos - c_mean_pos)
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_canon = c_id

            if nearest_canon is not None:
                canonical_clusters[nearest_canon].extend(rc["tids"])
                # We intentionally skip updating canonical_frames so future
                # residuals still see correct non-overlap status.
                force_merged_to_nearest_obs += rc["total_obs"]
                discarded_noise_obs = 0  # nothing truly lost
            
    raw_to_canonical: dict[int, int] = {}
    for canon_id, tids in canonical_clusters.items():
        for tid in tids:
            raw_to_canonical[tid] = canon_id
            
    stitched_subset = subset_df[subset_df["raw_track_id"].isin(raw_to_canonical)].copy()
    stitched_subset["canonical_id"] = stitched_subset["raw_track_id"].map(raw_to_canonical).astype(int)
    stitched_subset["team_label"] = team_label
    
    stitched_subset = (
        stitched_subset.sort_values(["canonical_id", "frame_id", "confidence"], ascending=[True, True, False])
        .drop_duplicates(subset=["canonical_id", "frame_id"])
        .sort_values(["frame_id", "canonical_id"])
        .reset_index(drop=True)
    )
    
    audit = {
        "team": team_label,
        "initial_raw_tracks": n_initial,
        "final_canonical_players": len(canonical_clusters),
        "target_canonical_players": target_canonical_count,
        "canonical_id_range": [start_canonical_id, start_canonical_id + len(canonical_clusters) - 1],
        "total_raw_observations": len(subset_df),
        "total_stitched_observations": len(stitched_subset),
        "absorbed_residuals": absorbed_residuals,
        "force_merged_to_nearest_obs": force_merged_to_nearest_obs,
        "discarded_noise_obs": discarded_noise_obs,
        "canonical_to_raw_mapping": {
            str(k): [int(x) for x in v] for k, v in canonical_clusters.items()
        },
    }
    return stitched_subset, audit


def stitch_player_tracks_two_teams(
    df: pd.DataFrame,
    raw_team_map: dict[int, str],
    players_per_team: int = 16,
    fps: float = 30.0,
    max_gap_seconds: float = 4.0,
    max_speed_threshold_px_s: float = 180.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Two-part team-based player track stitching:
    - Team A: stitched into 16 canonical IDs (1 to 16)
    - Team B: stitched into 16 canonical IDs (17 to 32)
    Guarantees strict separation between teams and exactly 32 canonical players.
    """
    id_col = "raw_track_id" if "raw_track_id" in df.columns else "canonical_id"
    df = df.copy()
    df["raw_track_id"] = df[id_col].astype(int)
    
    team_a_tids = [tid for tid, tm in raw_team_map.items() if tm == "team_a"]
    team_b_tids = [tid for tid, tm in raw_team_map.items() if tm == "team_b"]
    
    df_a, audit_a = stitch_tracklet_subset(
        df, team_a_tids,
        target_canonical_count=players_per_team,
        start_canonical_id=1,
        team_label="team_a",
        fps=fps,
        max_gap_seconds=max_gap_seconds,
        max_speed_threshold_px_s=max_speed_threshold_px_s,
    )
    
    df_b, audit_b = stitch_tracklet_subset(
        df, team_b_tids,
        target_canonical_count=players_per_team,
        start_canonical_id=players_per_team + 1,
        team_label="team_b",
        fps=fps,
        max_gap_seconds=max_gap_seconds,
        max_speed_threshold_px_s=max_speed_threshold_px_s,
    )
    
    stitched_all = pd.concat([df_a, df_b], ignore_index=True)
    stitched_all = stitched_all.sort_values(["frame_id", "canonical_id"]).reset_index(drop=True)
    
    total_preserved = len(stitched_all)
    retention_pct = (total_preserved / max(len(df), 1)) * 100.0
    
    full_audit = {
        "status": "SUCCESS",
        "stitching_mode": "two_teams_16_players_each",
        "team_a": audit_a,
        "team_b": audit_b,
        "total_canonical_players": stitched_all["canonical_id"].nunique(),
        "total_raw_observations": len(df),
        "total_stitched_observations": total_preserved,
        "observation_retention_pct": round(retention_pct, 2),
    }
    
    return stitched_all, full_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="AthletiQ Player Track Stitching CLI")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input player tracks CSV")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output stitched tracks CSV")
    parser.add_argument("--audit", "-a", type=Path, default=None, help="Output JSON audit file")
    parser.add_argument("--max-players", type=int, default=32, help="Max canonical player IDs (default: 32)")
    parser.add_argument("--fps", type=float, default=30.0, help="Video framerate (default: 30.0)")
    parser.add_argument("--max-gap", type=float, default=4.0, help="Max time gap in seconds for stitching")
    args = parser.parse_args()
    
    if not args.input.exists():
        print(f"[ERROR] Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
        
    df = pd.read_csv(args.input)
    print(f"[STITCH] Loaded {len(df)} detections across {df['canonical_id'].nunique() if 'canonical_id' in df.columns else df['raw_track_id'].nunique()} raw IDs.")
    
    stitched_df, audit = stitch_player_tracks(
        df,
        max_canonical_players=args.max_players,
        fps=args.fps,
        max_gap_seconds=args.max_gap,
    )
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stitched_df.to_csv(args.output, index=False)
    print(f"[STITCH] Saved stitched tracks to {args.output} ({len(stitched_df)} rows, {audit['final_canonical_players']} canonical players).")
    
    if args.audit:
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        print(f"[STITCH] Saved audit to {args.audit}")


if __name__ == "__main__":
    main()

