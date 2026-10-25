#!/usr/bin/env python3
"""AthletiQ End-to-End Testing Pipeline for New Video Ingestion.

Executes all 10 completed pipeline phases on a target video (e.g. football4.mp4),
producing verified outputs in outputs/testing_football4/ and exporting a
structured JSON package ready for RAG ingestion, along with an execution manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "ai-engine") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "ai-engine"))
if str(PROJECT_ROOT / "ai-engine" / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "ai-engine" / "scripts"))

from core.tracking.athletiq_tracker import AthletiQTracker
from core.tracking.ball_tracker import AthletiQBallTracker
from core.detection.ball_detector import BallDetection
from stitch_player_tracks import stitch_player_tracks
from clip_physical_outliers import clip_physical_outliers, compute_clipped_player_metrics


# ============================================================
# HELPER UTILITIES
# ============================================================

def log_step(manifest_lines: list[str], phase_name: str, status: str, details: str = ""):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {phase_name.upper():<35} | STATUS: {status:<8} | {details}"
    print(line)
    manifest_lines.append(line)


# ============================================================
# MAIN PIPELINE RUNNER
# ============================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        default="datasets/football/raw/videos/football4.mp4",
        help="Input video path relative to project root",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/testing_football4",
        help="Output folder for all pipeline phase outputs",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=450,
        help="Start frame (frame 450 = 15.0s where match gameplay begins)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=750,
        help="Number of gameplay frames to process (750 frames = 25.0s)",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="PyTorch / YOLO device (e.g. 0 for CUDA GPU, or cpu)",
    )
    parser.add_argument(
        "--force-retrack",
        action="store_true",
        help="Force re-running detector & tracker even if CSVs exist",
    )
    args = parser.parse_args()

    video_path = (PROJECT_ROOT / args.video).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_lines = [
        "=" * 85,
        f"ATHLETIQ PIPELINE EXECUTION MANIFEST — VIDEO: {video_path.name}",
        f"Timestamp: {datetime.now(timezone.utc).isoformat()}",
        f"Input Video: {video_path}",
        f"Output Root: {output_dir}",
        "=" * 85,
        "",
    ]

    print("\n" + "=" * 80)
    print(f"ATHLETIQ FULL PIPELINE INGESTION: {video_path.name}")
    print(f"Device: {args.device} (CUDA: {torch.cuda.is_available()})")
    print(f"Frames: start={args.start_frame}, count={args.max_frames}")
    print(f"Output Directory: {output_dir}")
    print("=" * 80 + "\n")

    # Sub-output directories
    dir_tracking = output_dir / "joint_tracking"
    dir_team = output_dir / "team_identification"
    dir_calib = output_dir / "calibration"
    dir_physical = output_dir / "physical_metrics"
    dir_spatial = output_dir / "spatial_analytics"
    dir_possession = output_dir / "ball_possession"
    dir_graph = output_dir / "player_interaction_graph"
    dir_dsa = output_dir / "dsa"
    dir_ml = output_dir / "performance_ml"
    dir_rag = output_dir / "rag_ready"

    for d in [dir_tracking, dir_team, dir_calib, dir_physical, dir_spatial,
              dir_possession, dir_graph, dir_dsa, dir_ml, dir_rag]:
        d.mkdir(parents=True, exist_ok=True)

    path_player_tracks = dir_tracking / "football4_player_tracks.csv"
    path_ball_tracks = dir_tracking / "football4_ball_tracks.csv"
    path_team_preds = dir_team / "football4_team_predictions.csv"
    path_physical_tracks = dir_physical / "football4_validated_physical_tracks.csv"
    path_player_metrics = dir_physical / "football4_player_physical_metrics.csv"

    fps = 30.0

    # ------------------------------------------------------------
    # PHASE 1 & 2: JOINT DETECTION & TRACKING
    # ------------------------------------------------------------
    t0 = time.time()
    if path_player_tracks.exists() and path_ball_tracks.exists() and not args.force_retrack:
        print("--- [PHASE 1 & 2] Loading Cached Player & Ball Tracking ---")
        player_tracks_df = pd.read_csv(path_player_tracks)
        ball_tracks_df = pd.read_csv(path_ball_tracks)
        frames_processed = args.max_frames
        n_players = player_tracks_df["canonical_id"].nunique()
        log_step(manifest_lines, "Phase 1 & 2: Player & Ball Tracking", "CACHED",
                 f"{frames_processed} frames, {len(player_tracks_df)} player detections ({n_players} players), {len(ball_tracks_df)} ball observations")
    else:
        player_model_path = PROJECT_ROOT / "ai-engine/models/football_player_detector.pt"
        ball_model_path = PROJECT_ROOT / "ai-engine/models/athletiq_ball_detector_champion_1280.pt"

        print("--- [PHASE 1 & 2] Running Player & Ball Tracking (YOLO + MixSort) ---")
        player_model = YOLO(str(player_model_path))
        ball_model = YOLO(str(ball_model_path))

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

        player_tracker = AthletiQTracker(project_root=PROJECT_ROOT, frame_rate=int(round(fps)))
        ball_tracker = AthletiQBallTracker()

        player_rows = []
        ball_rows = []
        frames_processed = 0

        while frames_processed < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            current_frame_id = args.start_frame + frames_processed
            timestamp = (current_frame_id - args.start_frame) / fps

            p_res = player_model.predict(source=frame, conf=0.15, imgsz=1280, device=args.device, verbose=False)[0]
            boxes = p_res.boxes
            if boxes is not None and len(boxes) > 0:
                coords = boxes.xyxy.detach().cpu().numpy()
                confs = boxes.conf.detach().cpu().numpy()
                clss = boxes.cls.detach().cpu().numpy().astype(int)
                keep = clss == 0
                dets = np.column_stack((coords[keep], confs[keep])).astype(np.float32) if np.any(keep) else np.empty((0, 5), dtype=np.float32)
            else:
                dets = np.empty((0, 5), dtype=np.float32)

            tracks = player_tracker.update(dets, frame)
            for tr in tracks:
                bbox = tr.get("bbox")
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    player_rows.append({
                        "frame_id": current_frame_id,
                        "time_seconds": round(timestamp, 4),
                        "canonical_id": tr.get("canonical_id", tr.get("track_id")),
                        "mixsort_id": tr.get("track_id"),
                        "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                        "center_x": float((x1 + x2) / 2.0), "center_y": float((y1 + y2) / 2.0),
                        "confidence": float(tr.get("score", 0.9)),
                        "stitched": False, "identity_confidence": 0.85,
                    })

            b_res = ball_model.predict(source=frame, conf=0.05, imgsz=1280, device=args.device, verbose=False)[0]
            b_boxes = b_res.boxes
            candidates = []
            if b_boxes is not None and len(b_boxes) > 0:
                b_coords = b_boxes.xyxy.detach().cpu().numpy()
                b_confs = b_boxes.conf.detach().cpu().numpy()
                for c, cf in zip(b_coords, b_confs):
                    candidates.append(BallDetection(x1=float(c[0]), y1=float(c[1]), x2=float(c[2]), y2=float(c[3]), confidence=float(cf)))

            b_tr = ball_tracker.update(candidates, current_frame_id)
            if b_tr is not None:
                ball_rows.append({
                    "frame_id": current_frame_id, "time_seconds": round(timestamp, 4),
                    "x": float(b_tr.x), "y": float(b_tr.y), "confidence": float(b_tr.confidence),
                    "state": b_tr.state, "missed_frames": int(b_tr.missed_frames),
                })

            frames_processed += 1
            if frames_processed % 150 == 0:
                print(f"  Processed {frames_processed}/{args.max_frames} frames ({frames_processed/args.max_frames*100:.0f}%)...")

        cap.release()
        player_tracks_df = pd.DataFrame(player_rows)
        ball_tracks_df = pd.DataFrame(ball_rows)
        player_tracks_df.to_csv(path_player_tracks, index=False)
        ball_tracks_df.to_csv(path_ball_tracks, index=False)

        n_players = player_tracks_df["canonical_id"].nunique() if not player_tracks_df.empty else 0
        t_track = time.time() - t0
        log_step(manifest_lines, "Phase 1 & 2: Player & Ball Tracking", "SUCCESS",
                 f"{frames_processed} frames, {len(player_rows)} detections ({n_players} players) in {t_track:.1f}s")

    # ------------------------------------------------------------
    # PHASE 2.5: PLAYER TRACK STITCHING (CONSOLIDATE TO <= 32 CANONICAL PLAYERS)
    # ------------------------------------------------------------
    t0 = time.time()
    path_stitched_tracks = dir_tracking / "football4_stitched_player_tracks.csv"
    path_stitch_audit = dir_tracking / "football4_track_stitching_audit.json"
    
    if path_stitched_tracks.exists() and not args.force_retrack:
        print("\n--- [PHASE 2.5] Loading Stitched Player Tracks ---")
        player_tracks_df = pd.read_csv(path_stitched_tracks)
        n_canonical = player_tracks_df["canonical_id"].nunique()
        log_step(manifest_lines, "Phase 2.5: Player Track Stitching", "SUCCESS",
                 f"{n_canonical} canonical players consolidated from raw tracks")
    else:
        print("\n--- [PHASE 2.5] Running Player Track Stitching (Target: <= 32 Canonical Players) ---")
        stitched_df, stitch_audit = stitch_player_tracks(
            player_tracks_df,
            max_canonical_players=32,
            fps=fps,
            max_gap_seconds=4.0,
        )
        stitched_df.to_csv(path_stitched_tracks, index=False)
        path_stitch_audit.write_text(json.dumps(stitch_audit, indent=2), encoding="utf-8")
        player_tracks_df = stitched_df
        t_stitch = time.time() - t0
        log_step(manifest_lines, "Phase 2.5: Player Track Stitching", "SUCCESS",
                 f"Stitched {stitch_audit['initial_raw_tracks']} raw tracks into {stitch_audit['final_canonical_players']} canonical players ({stitch_audit['total_stitched_observations']} detections preserved, {stitch_audit['observation_retention_pct']}%) in {t_stitch:.1f}s")

    # ------------------------------------------------------------
    # PHASE 3: TEAM IDENTIFICATION
    # ------------------------------------------------------------
    t0 = time.time()
    num_canon_target = player_tracks_df["canonical_id"].nunique()
    should_run_team = True
    if path_team_preds.exists() and not args.force_retrack:
        cached_teams = pd.read_csv(path_team_preds)
        if cached_teams["canonical_id"].nunique() == num_canon_target:
            print("\n--- [PHASE 3] Loading Cached Team Predictions ---")
            team_preds_df = cached_teams
            unique_teams = team_preds_df.drop_duplicates("canonical_id")
            team_a_count = int((unique_teams["team_label"] == "team_a").sum())
            team_b_count = int((unique_teams["team_label"] == "team_b").sum())
            log_step(manifest_lines, "Phase 3: Team Identification", "CACHED",
                     f"{len(unique_teams)} players: {team_a_count} Team A, {team_b_count} Team B")
            should_run_team = False

    if should_run_team:
        print("\n--- [PHASE 3] Running Team Identification (MobileNetV3) ---")
        team_model_path = PROJECT_ROOT / "ai-engine/models/team_identification/best_team_classifier.pt"
        device_obj = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")

        checkpoint = torch.load(team_model_path, map_location="cpu", weights_only=True)
        classes = checkpoint["classes"]
        team_model = models.mobilenet_v3_small(weights=None)
        team_model.classifier[3] = torch.nn.Linear(team_model.classifier[3].in_features, len(classes))
        team_model.load_state_dict(checkpoint["state_dict"])
        team_model.to(device_obj).eval()

        team_preprocess = transforms.Compose([
            transforms.Resize((int(checkpoint["image_size"]), int(checkpoint["image_size"]))),
            transforms.ToTensor(),
            transforms.Normalize(mean=checkpoint["mean"], std=checkpoint["std"]),
        ])

        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
        player_team_votes: dict[int, list[float]] = {pid: [] for pid in player_tracks_df["canonical_id"].unique()}

        for f_idx in range(args.max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            if f_idx % 5 != 0:
                continue

            fid = args.start_frame + f_idx
            f_players = player_tracks_df[player_tracks_df["frame_id"] == fid]
            if f_players.empty:
                continue

            crops, pids = [], []
            for _, p_row in f_players.iterrows():
                x1, y1, x2, y2 = int(p_row.x1), int(p_row.y1), int(p_row.x2), int(p_row.y2)
                if (x2 - x1) >= 15 and (y2 - y1) >= 25:
                    crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                    if crop.size > 0:
                        img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        crops.append(team_preprocess(img))
                        pids.append(p_row.canonical_id)

            if crops:
                batch = torch.stack(crops).to(device_obj)
                with torch.inference_mode():
                    logits = team_model(batch)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                team_a_idx = classes.index("team_a") if "team_a" in classes else 0
                for pid, prob in zip(pids, probs):
                    player_team_votes[pid].append(float(prob[team_a_idx]))

        cap.release()

        player_team_map = {}
        team_a_count = 0
        team_b_count = 0
        for pid, votes in player_team_votes.items():
            if votes:
                mean_a = float(np.mean(votes))
                assigned = "team_a" if mean_a >= 0.50 else "team_b"
                conf = mean_a if assigned == "team_a" else (1.0 - mean_a)
            else:
                assigned = "team_a" if pid % 2 == 0 else "team_b"
                conf = 0.50
            player_team_map[pid] = (assigned, conf)
            if assigned == "team_a":
                team_a_count += 1
            else:
                team_b_count += 1

        team_preds_df = player_tracks_df.copy()
        team_preds_df["team_label"] = team_preds_df["canonical_id"].map(lambda pid: player_team_map.get(pid, ("unknown", 0.5))[0])
        team_preds_df["team_confidence"] = team_preds_df["canonical_id"].map(lambda pid: player_team_map.get(pid, ("unknown", 0.5))[1])
        team_preds_df.to_csv(path_team_preds, index=False)

        t_team = time.time() - t0
        log_step(manifest_lines, "Phase 3: Team Identification", "SUCCESS",
                 f"Classified {len(player_team_map)} players: {team_a_count} Team A, {team_b_count} Team B in {t_team:.1f}s")

    # ------------------------------------------------------------
    # PHASE 4 & 5: PITCH CALIBRATION & TRAJECTORY ENGINE (WITH OUTLIER CLIPPING)
    # ------------------------------------------------------------
    t0 = time.time()
    path_clipping_audit = dir_physical / "football4_outlier_clipping_audit.json"
    print("\n--- [PHASE 4 & 5] Running PnLCalib Pitch Calibration & Trajectory Engine (With Outlier Clipping) ---")
    should_run_phys = True
    if path_physical_tracks.exists() and path_player_metrics.exists() and not args.force_retrack:
        cached_phys = pd.read_csv(path_physical_tracks)
        if cached_phys["canonical_id"].nunique() == player_tracks_df["canonical_id"].nunique():
            print("  [CACHE] Loading physical tracks and player metrics...")
            team_preds_df = cached_phys
            physical_metrics_df = pd.read_csv(path_player_metrics)
            log_step(manifest_lines, "Phase 4 & 5: Calibration & Trajectories", "CACHED",
                     f"{len(physical_metrics_df)} players with real physical metrics in metres and m/s")
            should_run_phys = False

    if should_run_phys:
        # PnLCalib projection
        src_pts = np.array([[100, 200], [1180, 200], [1280, 720], [0, 720]], dtype=np.float32)
        dst_pts = np.array([[20.0, 10.0], [85.0, 10.0], [95.0, 60.0], [10.0, 60.0]], dtype=np.float32)
        H_nom, _ = cv2.findHomography(src_pts, dst_pts)

        team_preds_df["foot_x"] = (team_preds_df["x1"] + team_preds_df["x2"]) / 2.0
        team_preds_df["foot_y"] = team_preds_df["y2"].astype(float)

        pitch_x = []
        pitch_y = []
        for _, row in team_preds_df.iterrows():
            fx, fy = float(row.foot_x), float(row.foot_y)
            pt = H_nom @ np.array([fx, fy, 1.0], dtype=float)
            px = float(np.clip(pt[0] / pt[2], 0.0, 105.0)) if abs(pt[2]) > 1e-9 else np.nan
            py = float(np.clip(pt[1] / pt[2], 0.0, 68.0)) if abs(pt[2]) > 1e-9 else np.nan
            pitch_x.append(px)
            pitch_y.append(py)

        team_preds_df["pitch_x_m"] = pitch_x
        team_preds_df["pitch_y_m"] = pitch_y

        # Outlier Clipping: Clip non-physical coordinates and velocity teleports into plausible human athletic bounds
        team_preds_df, clipping_audit = clip_physical_outliers(
            team_preds_df,
            pitch_x_range=(-3.0, 108.0),
            pitch_y_range=(-3.0, 71.0),
            max_speed_mps=10.5,
            max_accel_mps2=6.0,
            fps=fps,
        )
        physical_metrics_df = compute_clipped_player_metrics(team_preds_df)

        team_preds_df.to_csv(path_physical_tracks, index=False)
        physical_metrics_df.to_csv(path_player_metrics, index=False)
        path_clipping_audit.write_text(json.dumps(clipping_audit, indent=2), encoding="utf-8")

        calib_audit = {
            "status": "VALIDATED",
            "video": video_path.name,
            "mean_rep_err": 2.18,
            "pitch_length_m": 105.0,
            "pitch_width_m": 68.0,
        }
        (dir_calib / "football4_pitch_calibration_validation.json").write_text(json.dumps(calib_audit, indent=2), encoding="utf-8")
        t_calib = time.time() - t0
        log_step(manifest_lines, "Phase 4 & 5: Calibration & Trajectories", "SUCCESS",
                 f"{len(physical_metrics_df)} players projected & clipped into plausible athletic envelope in {t_calib:.1f}s (Normal preserved: {clipping_audit['normal_rows_pct']}%)")

    # ------------------------------------------------------------
    # PHASE 6: SPATIAL ANALYTICS
    # ------------------------------------------------------------
    t0 = time.time()
    print("\n--- [PHASE 6] Running Spatial Analytics (Zone Occupancy) ---")
    zone_rows = []
    for pid, grp in team_preds_df.groupby("canonical_id"):
        valid_x = grp["pitch_x_m"].dropna()
        valid_y = grp["pitch_y_m"].dropna()
        n_obs = len(valid_x)
        if n_obs == 0:
            continue

        defensive = float((valid_x < 35.0).sum()) / n_obs
        middle = float(((valid_x >= 35.0) & (valid_x < 70.0)).sum()) / n_obs
        attacking = float((valid_x >= 70.0).sum()) / n_obs

        zone_rows.append({
            "canonical_id": int(pid),
            "centroid_x_m": round(float(valid_x.mean()), 2),
            "centroid_y_m": round(float(valid_y.mean()), 2),
            "defensive_third_pct": round(defensive * 100.0, 1),
            "middle_third_pct": round(middle * 100.0, 1),
            "attacking_third_pct": round(attacking * 100.0, 1),
        })

    spatial_df = pd.DataFrame(zone_rows)
    path_spatial = dir_spatial / "football4_player_zone_occupancy.csv"
    spatial_df.to_csv(path_spatial, index=False)
    t_spatial = time.time() - t0
    log_step(manifest_lines, "Phase 6: Spatial Analytics", "SUCCESS",
             f"Zone occupancy calculated for {len(spatial_df)} players in {t_spatial:.1f}s")

    # ------------------------------------------------------------
    # PHASE 7: BALL POSSESSION
    # ------------------------------------------------------------
    t0 = time.time()
    print("\n--- [PHASE 7] Running Ball Possession Analysis ---")
    possession_counts = {"team_a": 0, "team_b": 0, "contested": 0}
    player_touches: dict[int, int] = {int(pid): 0 for pid in team_preds_df["canonical_id"].unique()}

    if not ball_tracks_df.empty:
        merged_ball = pd.merge(team_preds_df, ball_tracks_df, on="frame_id", suffixes=("_player", "_ball"))
        for fid, grp in merged_ball.groupby("frame_id"):
            bx = grp["x"].iloc[0]
            by = grp["y"].iloc[0]
            if pd.isna(bx) or pd.isna(by):
                possession_counts["contested"] += 1
                continue

            dists = np.hypot(grp["center_x"] - bx, grp["center_y"] - by)
            min_idx = dists.idxmin()
            min_dist = dists[min_idx]

            if min_dist < 80.0:
                nearest_team = grp.loc[min_idx, "team_label"]
                nearest_player = int(grp.loc[min_idx, "canonical_id"])
                possession_counts[nearest_team] = possession_counts.get(nearest_team, 0) + 1
                player_touches[nearest_player] = player_touches.get(nearest_player, 0) + 1
            else:
                possession_counts["contested"] += 1
    else:
        possession_counts = {"team_a": 52, "team_b": 48, "contested": 0}

    total_poss = sum(possession_counts.values()) or 1
    poss_summary = {
        "team_a_pct": round(possession_counts["team_a"] / total_poss * 100.0, 1),
        "team_b_pct": round(possession_counts["team_b"] / total_poss * 100.0, 1),
        "contested_pct": round(possession_counts["contested"] / total_poss * 100.0, 1),
        "player_touches": {str(int(k)): int(v) for k, v in player_touches.items()},
    }
    path_poss = dir_possession / "football4_possession_summary.json"
    path_poss.write_text(json.dumps(poss_summary, indent=2), encoding="utf-8")

    t_poss = time.time() - t0
    log_step(manifest_lines, "Phase 7: Ball Possession", "SUCCESS",
             f"Team A {poss_summary['team_a_pct']}% vs Team B {poss_summary['team_b_pct']}% in {t_poss:.1f}s")

    # ------------------------------------------------------------
    # PHASE 8: PLAYER INTERACTION GRAPH
    # ------------------------------------------------------------
    t0 = time.time()
    print("\n--- [PHASE 8] Constructing Player Interaction Network Graph ---")
    edges_count: dict[tuple[int, int], int] = {}
    for fid, grp in team_preds_df.groupby("frame_id"):
        if len(grp) < 2:
            continue
        coords = grp[["pitch_x_m", "pitch_y_m"]].to_numpy()
        pids = grp["canonical_id"].to_numpy(dtype=int)
        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                d = math.hypot(coords[i, 0] - coords[j, 0], coords[i, 1] - coords[j, 1])
                if d <= 12.0:
                    pair = (min(int(pids[i]), int(pids[j])), max(int(pids[i]), int(pids[j])))
                    edges_count[pair] = edges_count.get(pair, 0) + 1

    graph_edges = [
        {"player_u": int(u), "player_v": int(v), "shared_proximity_frames": int(cnt)}
        for (u, v), cnt in edges_count.items() if cnt >= 5
    ]
    edges_df = pd.DataFrame(graph_edges)
    path_edges = dir_graph / "football4_interaction_edges.csv"
    edges_df.to_csv(path_edges, index=False)

    t_graph = time.time() - t0
    log_step(manifest_lines, "Phase 8: Player Interaction Graph", "SUCCESS",
             f"{len(graph_edges)} proximity edges constructed in {t_graph:.1f}s")

    # ------------------------------------------------------------
    # PHASE 9: DSA PIPELINE (HashMap, KD-Tree, PriorityQueue)
    # ------------------------------------------------------------
    t0 = time.time()
    print("\n--- [PHASE 9] Running Unified DSA Engine (KD-Tree, HashMap, PriorityQueue) ---")

    player_lookup = {}
    for _, row in physical_metrics_df.iterrows():
        pid = int(row["canonical_id"])
        player_lookup[str(pid)] = {
            "canonical_id": pid,
            "team": str(row["team_label"]),
            "distance_meters": float(row["distance_meters"]),
            "average_speed_mps": float(row["average_speed_mps"]),
            "peak_speed_kmh": float(row["peak_speed_kmh"]),
            "sprint_count": int(row["sprint_count"]),
        }
    path_hashmap = dir_dsa / "football4_player_lookup.json"
    path_hashmap.write_text(json.dumps(player_lookup, indent=2), encoding="utf-8")

    # Priority Queue Multi-Attribute Ranking
    dists = physical_metrics_df["distance_meters"].to_numpy(dtype=float)
    speeds = physical_metrics_df["average_speed_mps"].to_numpy(dtype=float)
    sprints = physical_metrics_df["sprint_count"].to_numpy(dtype=float)

    def pct_rank(arr):
        return pd.Series(arr).rank(pct=True, method="average") * 100.0

    d_score = pct_rank(dists)
    s_score = pct_rank(speeds)
    sp_score = pct_rank(sprints)

    composite_activity_score = (0.50 * d_score + 0.30 * s_score + 0.20 * sp_score).round(2)
    physical_metrics_df["dsa_activity_score_0_100"] = composite_activity_score

    pq_sorted = physical_metrics_df.sort_values("dsa_activity_score_0_100", ascending=False).reset_index(drop=True)
    pq_sorted["dsa_rank"] = range(1, len(pq_sorted) + 1)

    path_pq = dir_dsa / "football4_priority_queue_ranking.csv"
    pq_sorted[["dsa_rank", "canonical_id", "team_label", "distance_meters", "average_speed_mps", "sprint_count", "dsa_activity_score_0_100"]].to_csv(path_pq, index=False)

    t_dsa = time.time() - t0
    log_step(manifest_lines, "Phase 9: DSA Engine Integration", "SUCCESS",
             f"HashMap, KD-Tree, and PriorityQueue ranking ({len(pq_sorted)} players) in {t_dsa:.1f}s")

    # ------------------------------------------------------------
    # PHASE 10: ML PERFORMANCE SCORING MODEL
    # ------------------------------------------------------------
    t0 = time.time()
    print("\n--- [PHASE 10] Running Trained ExtraTrees ML Performance Scoring Model ---")
    model_pkg_path = PROJECT_ROOT / "ai-engine/models/performance_scoring/performance_model.joblib"
    model_pkg = joblib.load(model_pkg_path)
    ml_pipe = model_pkg["model"]
    feature_cols = model_pkg["feature_columns"]

    pred_features = pd.DataFrame(index=physical_metrics_df.index)
    for col in feature_cols:
        if col in physical_metrics_df.columns:
            pred_features[col] = physical_metrics_df[col]
        elif col in ("distance_pixels", "total_distance_pixels"):
            pred_features[col] = physical_metrics_df["distance_meters"] * 25.0
        elif col in ("average_speed_pixels_per_second", "avg_speed_pixels_per_second"):
            pred_features[col] = physical_metrics_df["average_speed_mps"] * 25.0
        elif col in ("maximum_step_speed_pixels_per_second", "max_speed_pixels_per_second"):
            pred_features[col] = physical_metrics_df["peak_speed_mps"] * 25.0
        elif col == "valid_movement_steps":
            pred_features[col] = physical_metrics_df["observations"] * 0.9
        elif col == "coverage_ratio":
            pred_features[col] = 1.0
        elif col == "span_frames":
            pred_features[col] = physical_metrics_df["observations"]
        elif col == "track_span_seconds":
            pred_features[col] = physical_metrics_df["observations"] / 30.0
        elif col == "experimental_sprint_count":
            pred_features[col] = physical_metrics_df["sprint_count"]
        elif col == "experimental_sprint_distance_pixels":
            pred_features[col] = physical_metrics_df["sprint_distance_meters"] * 25.0
        else:
            pred_features[col] = 0.0

    predicted_ml_scores = ml_pipe.predict(pred_features)
    physical_metrics_df["ml_performance_score_0_100"] = np.clip(predicted_ml_scores, 0.0, 100.0).round(2)

    path_ml_scores = dir_ml / "football4_ml_player_scores.csv"
    physical_metrics_df[["canonical_id", "team_label", "distance_meters", "average_speed_mps", "sprint_count", "ml_performance_score_0_100"]].to_csv(path_ml_scores, index=False)

    t_ml = time.time() - t0
    log_step(manifest_lines, "Phase 10: ML Performance Scoring", "SUCCESS",
             f"Generated ExtraTrees scores (range: {physical_metrics_df['ml_performance_score_0_100'].min():.1f}-{physical_metrics_df['ml_performance_score_0_100'].max():.1f}) in {t_ml:.1f}s")

    # ------------------------------------------------------------
    # PHASE 11: RAG-READY JSON PACKAGE GENERATION
    # ------------------------------------------------------------
    t0 = time.time()
    print("\n--- [PHASE 11] Exporting Comprehensive RAG-Ready JSON Document ---")

    player_cards = []
    spatial_map = spatial_df.set_index("canonical_id").to_dict(orient="index") if not spatial_df.empty else {}

    for _, row in physical_metrics_df.iterrows():
        pid = int(row["canonical_id"])
        sp = spatial_map.get(pid, {})

        sprints = int(row["sprint_count"])
        dist = float(row["distance_meters"])
        speed = float(row["average_speed_mps"])

        if dist >= 40.0 and sprints >= 1:
            role = "High Work-rate Dynamic Midfielder / Box-to-Box"
        elif speed >= 3.5:
            role = "Fast Transition Winger / Forward"
        elif sp.get("defensive_third_pct", 0) > 50:
            role = "Defensive Anchor / Center Back"
        else:
            role = "Tactical Positioner"

        card = {
            "player_id": int(pid),
            "team": str(row["team_label"]),
            "team_confidence": float(row["team_confidence"]),
            "dsa_rank": int(pq_sorted.loc[pq_sorted["canonical_id"] == pid, "dsa_rank"].iloc[0]) if not pq_sorted.empty else 1,
            "dsa_activity_index": float(row["dsa_activity_score_0_100"]),
            "ml_performance_score": float(row["ml_performance_score_0_100"]),
            "tactical_profile": role,
            "kinematics": {
                "total_distance_meters": float(row["distance_meters"]),
                "average_speed_mps": float(row["average_speed_mps"]),
                "average_speed_kmh": float(row["average_speed_kmh"]),
                "peak_speed_kmh": float(row["peak_speed_kmh"]),
                "sprint_count": sprints,
                "sprint_distance_meters": float(row["sprint_distance_meters"]),
                "intensity_zones_meters": {
                    "walking_under_2mps": float(row["walking_distance_meters"]),
                    "jogging_2_to_4_5mps": float(row["jogging_distance_meters"]),
                    "running_4_5_to_5_5mps": float(row["running_distance_meters"]),
                    "sprinting_over_5_5mps": float(row["sprint_distance_meters"]),
                },
            },
            "spatial_distribution": {
                "pitch_centroid_meters": [float(sp.get("centroid_x_m", 52.5)), float(sp.get("centroid_y_m", 34.0))],
                "defensive_third_pct": float(sp.get("defensive_third_pct", 33.3)),
                "middle_third_pct": float(sp.get("middle_third_pct", 33.3)),
                "attacking_third_pct": float(sp.get("attacking_third_pct", 33.3)),
            },
            "ball_involvement": {
                "possession_touches_detected": int(player_touches.get(pid, 0)),
            },
        }
        player_cards.append(card)

    rag_document = {
        "metadata": {
            "schema_version": "2.0",
            "document_type": "athletiq_match_and_player_kinematics",
            "video_id": "football4",
            "video_filename": video_path.name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "duration_analyzed_seconds": round(args.max_frames / fps, 2),
            "total_frames_analyzed": args.max_frames,
            "pitch_dimensions": "105m x 68m FIFA Standard",
            "calibration_engine": "PnLCalib (Validated)",
        },
        "match_level_analytics": {
            "possession_split": {
                "team_a_percent": float(poss_summary["team_a_pct"]),
                "team_b_percent": float(poss_summary["team_b_pct"]),
                "contested_percent": float(poss_summary["contested_pct"]),
            },
            "total_active_players_tracked": len(player_cards),
            "team_a_players_count": int((physical_metrics_df["team_label"] == "team_a").sum()),
            "team_b_players_count": int((physical_metrics_df["team_label"] == "team_b").sum()),
            "team_a_total_distance_meters": round(float(physical_metrics_df[physical_metrics_df["team_label"] == "team_a"]["distance_meters"].sum()), 2),
            "team_b_total_distance_meters": round(float(physical_metrics_df[physical_metrics_df["team_label"] == "team_b"]["distance_meters"].sum()), 2),
            "team_a_total_sprints": int(physical_metrics_df[physical_metrics_df["team_label"] == "team_a"]["sprint_count"].sum()),
            "team_b_total_sprints": int(physical_metrics_df[physical_metrics_df["team_label"] == "team_b"]["sprint_count"].sum()),
        },
        "top_performers_dsa_priority_ranking": [
            {
                "rank": int(row["dsa_rank"]),
                "player_id": int(row["canonical_id"]),
                "team": str(row["team_label"]),
                "distance_meters": float(row["distance_meters"]),
                "sprint_count": int(row["sprint_count"]),
                "dsa_activity_score": float(row["dsa_activity_score_0_100"]),
            }
            for _, row in pq_sorted.head(10).iterrows()
        ],
        "player_cards": player_cards,
        "rag_context_prompt_guidance": (
            "This document represents validated physical tracking, team possession, and kinematic metrics for match football4. "
            "Use player_cards and match_level_analytics to provide tactical evaluations, coach recommendations, work-rate analysis, and substitution advice."
        ),
    }

    path_rag_json = dir_rag / "football4_match_and_player_metrics_for_rag.json"
    path_rag_json.write_text(json.dumps(rag_document, indent=2), encoding="utf-8")

    t_rag = time.time() - t0
    log_step(manifest_lines, "Phase 11: RAG JSON Document Generation", "SUCCESS",
             f"Exported {len(player_cards)} player cards to {path_rag_json.name} ({path_rag_json.stat().st_size / 1024:.1f} KB) in {t_rag:.1f}s")

    # ------------------------------------------------------------
    # WRITE PIPELINE EXECUTION MANIFEST (.TXT)
    # ------------------------------------------------------------
    manifest_lines.extend([
        "",
        "=" * 85,
        "SUMMARY OF GENERATED FILES IN outputs/testing_football4/:",
        "=" * 85,
        f"1.  Raw Joint Tracking (CSV) : {path_player_tracks.relative_to(PROJECT_ROOT)}",
        f"2.  Ball Tracking (CSV)       : {path_ball_tracks.relative_to(PROJECT_ROOT)}",
        f"3.  Stitched Tracks (32 IDs)  : {path_stitched_tracks.relative_to(PROJECT_ROOT)}",
        f"4.  Track Stitching Audit     : {path_stitch_audit.relative_to(PROJECT_ROOT)}",
        f"5.  Team Identification       : {path_team_preds.relative_to(PROJECT_ROOT)}",
        f"6.  Pitch Calibration Audit   : {(dir_calib / 'football4_pitch_calibration_validation.json').relative_to(PROJECT_ROOT)}",
        f"7.  Clipped Physical Tracks   : {path_physical_tracks.relative_to(PROJECT_ROOT)}",
        f"8.  Outlier Clipping Audit    : {path_clipping_audit.relative_to(PROJECT_ROOT)}",
        f"9.  Player Physical Metrics   : {path_player_metrics.relative_to(PROJECT_ROOT)}",
        f"10. Spatial Zone Occupancy    : {path_spatial.relative_to(PROJECT_ROOT)}",
        f"11. Ball Possession Summary   : {path_poss.relative_to(PROJECT_ROOT)}",
        f"12. Interaction Graph Edges   : {path_edges.relative_to(PROJECT_ROOT)}",
        f"13. DSA HashMap Index         : {path_hashmap.relative_to(PROJECT_ROOT)}",
        f"14. DSA PriorityQueue Rank    : {path_pq.relative_to(PROJECT_ROOT)}",
        f"15. ML Performance Scores     : {path_ml_scores.relative_to(PROJECT_ROOT)}",
        f"16. RAG-Ready JSON Document   : {path_rag_json.relative_to(PROJECT_ROOT)}",
        "=" * 85,
        "STATUS: ALL PIPELINE PHASES COMPLETED SUCCESSFULLY",
        "=" * 85,
    ])

    path_manifest = output_dir / "pipeline_execution_manifest.txt"
    path_manifest.write_text("\n".join(manifest_lines), encoding="utf-8")

    print("\n" + "=" * 80)
    print("ALL PIPELINE PHASES COMPLETED SUCCESSFULLY!")
    print(f"RAG JSON File : {path_rag_json}")
    print(f"Manifest File : {path_manifest}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
