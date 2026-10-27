
#!/usr/bin/env python3
"""Run AthletiQ player and ball tracking on the same video."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


# Repository layout:
# AthletiQ/
#   ai-engine/
#     scripts/run_joint_tracking.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
AI_ENGINE = PROJECT_ROOT / "ai-engine"

if str(AI_ENGINE) not in sys.path:
    sys.path.insert(0, str(AI_ENGINE))

from core.tracking.athletiq_tracker import AthletiQTracker
from core.tracking.ball_tracker import AthletiQBallTracker
from core.detection.ball_detector import BallDetection


def resolve_path(value: str) -> Path:
    """Resolve relative paths against the repository root."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def model_names(model: YOLO) -> dict[int, str]:
    names = model.names

    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}

    if isinstance(names, (list, tuple)):
        return {index: str(value) for index, value in enumerate(names)}

    return {}


def parse_class_ids(value: str | None) -> set[int] | None:
    if value is None or not value.strip():
        return None

    return {
        int(part.strip())
        for part in value.split(",")
        if part.strip()
    }


def choose_player_classes(
    model: YOLO,
    override: str | None,
) -> set[int]:
    """Use explicit class IDs or infer player classes from model labels."""
    if override is not None:
        selected = parse_class_ids(override)
        if not selected:
            raise ValueError("--player-classes must contain class IDs")
        return selected

    names = model_names(model)
    keywords = (
        "player",
        "person",
        "goalkeeper",
        "goal keeper",
        "referee",
        "official",
    )

    matched = {
        class_id
        for class_id, name in names.items()
        if any(word in name.lower() for word in keywords)
    }

    if matched:
        return matched

    if len(names) == 1:
        return set(names)

    raise ValueError(
        "Could not infer player classes from model labels. "
        "Supply --player-classes explicitly."
    )


def choose_ball_classes(
    model: YOLO,
    override: str | None,
) -> set[int]:
    """Use explicit class IDs or infer ball classes from model labels."""
    if override is not None:
        selected = parse_class_ids(override)
        if not selected:
            raise ValueError("--ball-classes must contain class IDs")
        return selected

    names = model_names(model)
    matched = {
        class_id
        for class_id, name in names.items()
        if "ball" in name.lower()
    }

    if matched:
        return matched

    if len(names) == 1:
        return set(names)

    # The AthletiQ champion ball detector is a single-class model.
    raise ValueError(
        "Could not infer the ball class from model labels. "
        "Supply --ball-classes explicitly."
    )


def get_player_detections(
    result,
    allowed_classes: set[int],
) -> np.ndarray:
    """Return player detections in the format expected by AthletiQTracker."""
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return np.empty((0, 5), dtype=np.float32)

    coordinates = boxes.xyxy.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()
    class_ids = boxes.cls.detach().cpu().numpy().astype(int)

    keep = np.isin(class_ids, list(allowed_classes))

    if not np.any(keep):
        return np.empty((0, 5), dtype=np.float32)

    # Required format: [x1, y1, x2, y2, confidence]
    return np.column_stack(
        (coordinates[keep], confidences[keep])
    ).astype(np.float32)


def get_ball_detections(
    result,
    allowed_classes: set[int],
) -> list[BallDetection]:
    """Convert YOLO output to the existing BallDetection dataclass."""
    candidates = []
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return candidates

    coordinates = boxes.xyxy.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()
    class_ids = boxes.cls.detach().cpu().numpy().astype(int)

    for coords, confidence, class_id in zip(
        coordinates, confidences, class_ids
    ):
        if int(class_id) not in allowed_classes:
            continue

        x1, y1, x2, y2 = map(float, coords)

        candidates.append(
            BallDetection(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=float(confidence),
            )
        )

    return candidates


def track_value(track, *keys, default=None):
    """Read a value from a track dictionary without assuming every key exists."""
    for key in keys:
        if key in track and track[key] is not None:
            return track[key]
    return default


def normalize_bbox(track: dict):
    bbox = track_value(track, "bbox", "xyxy")

    if bbox is None:
        return None

    if isinstance(bbox, dict):
        if all(key in bbox for key in ("x1", "y1", "x2", "y2")):
            return tuple(
                float(bbox[key])
                for key in ("x1", "y1", "x2", "y2")
            )

        bbox = bbox.get("xyxy")

        if bbox is None:
            return None

    try:
        values = list(bbox)
    except TypeError:
        return None

    if len(values) < 4:
        return None

    return tuple(float(value) for value in values[:4])


def normalize_center(track: dict, bbox):
    center = track.get("center")

    if center is not None:
        try:
            return float(center[0]), float(center[1])
        except (TypeError, IndexError, ValueError):
            pass

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    return None, None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run AthletiQ player and ball trackers "
            "in one synchronized video loop."
        )
    )

    parser.add_argument(
        "--input",
        default="datasets/football/raw/videos/Football.mp4",
    )
    parser.add_argument(
        "--player-model",
        default="ai-engine/models/football_player_detector.pt",
    )
    parser.add_argument(
        "--ball-model",
        default="ai-engine/models/athletiq_ball_detector_champion_1280.pt",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/joint_tracking",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Ultralytics device, e.g. 0 or cpu",
    )
    parser.add_argument("--player-conf", type=float, default=0.10)
    parser.add_argument("--ball-conf", type=float, default=0.05)
    parser.add_argument("--player-imgsz", type=int, default=1280)
    parser.add_argument("--ball-imgsz", type=int, default=1280)
    parser.add_argument("--player-classes", default=None)
    parser.add_argument("--ball-classes", default=None)
    parser.add_argument("--reid-checkpoint", default=None)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum frames to process; 0 means the full video",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing output files",
    )

    args = parser.parse_args()

    if args.max_frames < 0:
        parser.error("--max-frames cannot be negative")

    input_path = resolve_path(args.input)
    player_model_path = resolve_path(args.player_model)
    ball_model_path = resolve_path(args.ball_model)
    output_dir = resolve_path(args.output_dir)

    for path in (input_path, player_model_path, ball_model_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    player_csv = output_dir / f"{stem}_player_tracks.csv"
    ball_csv = output_dir / f"{stem}_ball_tracks.csv"
    joint_csv = output_dir / f"{stem}_joint_frames.csv"
    output_video = output_dir / f"{stem}_joint_tracking.mp4"

    output_files = (
        player_csv,
        ball_csv,
        joint_csv,
        output_video,
    )

    if not args.overwrite:
        existing = [str(path) for path in output_files if path.exists()]
        if existing:
            raise FileExistsError(
                "Output files already exist. Use --overwrite to replace: "
                + ", ".join(existing)
            )

    print(f"Repository: {PROJECT_ROOT}")
    print(f"Input video: {input_path}")
    print(f"Device: {args.device}")

    # Load player model.
    print(f"\nLoading player detector: {player_model_path}")
    player_model = YOLO(str(player_model_path))
    player_names = model_names(player_model)
    print(f"Player model labels: {player_names}")

    player_class_ids = choose_player_classes(
        player_model,
        args.player_classes,
    )
    print(f"Player class IDs: {sorted(player_class_ids)}")

    # Load ball model.
    print(f"\nLoading ball detector: {ball_model_path}")
    ball_model = YOLO(str(ball_model_path))
    ball_names = model_names(ball_model)
    print(f"Ball model labels: {ball_names}")

    ball_class_ids = choose_ball_classes(
        ball_model,
        args.ball_classes,
    )
    print(f"Ball class IDs: {sorted(ball_class_ids)}")

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))

    if not math.isfinite(fps) or fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Could not read video dimensions.")

    player_tracker = None
    video_writer = None

    frames_processed = 0
    player_rows = 0
    unique_player_ids = set()

    ball_states = {
        "detected": 0,
        "predicted": 0,
        "lost": 0,
    }

    player_fields = [
        "frame_id",
        "time_seconds",
        "canonical_id",
        "mixsort_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "center_x",
        "center_y",
        "confidence",
        "stitched",
        "identity_confidence",
    ]

    ball_fields = [
        "frame_id",
        "time_seconds",
        "x",
        "y",
        "confidence",
        "state",
        "missed_frames",
    ]

    joint_fields = [
        "frame_id",
        "time_seconds",
        "player_count",
        "player_ids_json",
        "ball_x",
        "ball_y",
        "ball_confidence",
        "ball_state",
    ]

    try:
        tracker_kwargs = {
            "project_root": PROJECT_ROOT,
            "frame_rate": max(1, int(round(fps))),
        }

        if args.reid_checkpoint:
            tracker_kwargs["reid_checkpoint"] = str(
                resolve_path(args.reid_checkpoint)
            )

        player_tracker = AthletiQTracker(**tracker_kwargs)
        ball_tracker = AthletiQBallTracker()

        print(
            "OSNet Re-ID: "
            + (
                "enabled"
                if player_tracker.reid_checkpoint is not None
                else "disabled (checkpoint not found)"
            )
        )

        with (
            player_csv.open("w", newline="", encoding="utf-8") as pf,
            ball_csv.open("w", newline="", encoding="utf-8") as bf,
            joint_csv.open("w", newline="", encoding="utf-8") as jf,
        ):
            player_writer = csv.DictWriter(
                pf, fieldnames=player_fields
            )
            ball_writer = csv.DictWriter(
                bf, fieldnames=ball_fields
            )
            joint_writer = csv.DictWriter(
                jf, fieldnames=joint_fields
            )

            player_writer.writeheader()
            ball_writer.writeheader()
            joint_writer.writeheader()

            while True:
                if (
                    args.max_frames > 0
                    and frames_processed >= args.max_frames
                ):
                    break

                ok, frame = cap.read()

                if not ok:
                    break

                frame_id = frames_processed
                timestamp = frame_id / fps

                # -------------------------------------------------
                # PLAYER DETECTION AND TRACKING
                # -------------------------------------------------
                player_result = player_model.predict(
                    source=frame,
                    conf=args.player_conf,
                    imgsz=args.player_imgsz,
                    device=args.device,
                    verbose=False,
                )[0]

                detections = get_player_detections(
                    player_result,
                    player_class_ids,
                )

                player_tracks = player_tracker.update(
                    detections,
                    frame,
                )

                # -------------------------------------------------
                # BALL DETECTION AND TRACKING
                # -------------------------------------------------
                ball_result = ball_model.predict(
                    source=frame,
                    conf=args.ball_conf,
                    imgsz=args.ball_imgsz,
                    device=args.device,
                    verbose=False,
                )[0]

                ball_candidates = get_ball_detections(
                    ball_result,
                    ball_class_ids,
                )

                ball_track = ball_tracker.update(
                    ball_candidates,
                    frame_id,
                )

                if ball_track is None:
                    ball_state = "lost"
                    ball_x = None
                    ball_y = None
                    ball_confidence = 0.0
                    ball_missed = 0
                else:
                    ball_state = ball_track.state
                    ball_x = float(ball_track.x)
                    ball_y = float(ball_track.y)
                    ball_confidence = float(ball_track.confidence)
                    ball_missed = int(ball_track.missed_frames)

                ball_states[ball_state] = (
                    ball_states.get(ball_state, 0) + 1
                )

                ball_writer.writerow({
                    "frame_id": frame_id,
                    "time_seconds": round(timestamp, 4),
                    "x": ball_x,
                    "y": ball_y,
                    "confidence": ball_confidence,
                    "state": ball_state,
                    "missed_frames": ball_missed,
                })

                # -------------------------------------------------
                # DRAW AND SAVE PLAYER TRACKS
                # -------------------------------------------------
                annotated = frame.copy()
                player_ids_this_frame = []

                for raw_track in player_tracks or []:
                    if not isinstance(raw_track, dict):
                        continue

                    track = raw_track
                    bbox = normalize_bbox(track)
                    center_x, center_y = normalize_center(
                        track, bbox
                    )

                    canonical_id = track_value(
                        track, "canonical_id"
                    )
                    mixsort_id = track_value(
                        track, "mixsort_id", "track_id"
                    )
                    confidence = track_value(
                        track, "confidence", "score"
                    )
                    stitched = track_value(
                        track, "stitched", default=False
                    )
                    identity_confidence = track_value(
                        track, "identity_confidence"
                    )

                    if canonical_id is not None:
                        canonical_text = str(canonical_id)
                        player_ids_this_frame.append(canonical_text)
                        unique_player_ids.add(canonical_text)

                    player_writer.writerow({
                        "frame_id": frame_id,
                        "time_seconds": round(timestamp, 4),
                        "canonical_id": canonical_id,
                        "mixsort_id": mixsort_id,
                        "x1": None if bbox is None else round(bbox[0], 3),
                        "y1": None if bbox is None else round(bbox[1], 3),
                        "x2": None if bbox is None else round(bbox[2], 3),
                        "y2": None if bbox is None else round(bbox[3], 3),
                        "center_x": (
                            None if center_x is None
                            else round(center_x, 3)
                        ),
                        "center_y": (
                            None if center_y is None
                            else round(center_y, 3)
                        ),
                        "confidence": confidence,
                        "stitched": stitched,
                        "identity_confidence": identity_confidence,
                    })

                    player_rows += 1

                    if bbox is not None:
                        x1, y1, x2, y2 = (
                            int(round(value)) for value in bbox
                        )

                        cv2.rectangle(
                            annotated,
                            (x1, y1),
                            (x2, y2),
                            (255, 180, 0),
                            2,
                        )

                        label_id = (
                            f"P{canonical_id}"
                            if canonical_id is not None
                            else f"T{mixsort_id}"
                        )

                        cv2.putText(
                            annotated,
                            label_id,
                            (x1, max(20, y1 - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 180, 0),
                            2,
                        )

                # Draw the ball: green = detected, yellow = predicted.
                if (
                    ball_track is not None
                    and ball_state in ("detected", "predicted")
                ):
                    bx = int(round(ball_x))
                    by = int(round(ball_y))

                    color = (
                        (0, 255, 0)
                        if ball_state == "detected"
                        else (0, 220, 255)
                    )

                    cv2.circle(
                        annotated, (bx, by), 9, color, 2
                    )
                    cv2.drawMarker(
                        annotated,
                        (bx, by),
                        color,
                        cv2.MARKER_CROSS,
                        18,
                        2,
                    )
                    cv2.putText(
                        annotated,
                        f"BALL: {ball_state}",
                        (max(5, bx + 10), max(20, by - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

                joint_writer.writerow({
                    "frame_id": frame_id,
                    "time_seconds": round(timestamp, 4),
                    "player_count": len(player_tracks or []),
                    "player_ids_json": json.dumps(
                        player_ids_this_frame
                    ),
                    "ball_x": ball_x,
                    "ball_y": ball_y,
                    "ball_confidence": ball_confidence,
                    "ball_state": ball_state,
                })

                if video_writer is None:
                    video_writer = cv2.VideoWriter(
                        str(output_video),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )

                    if not video_writer.isOpened():
                        raise RuntimeError(
                            f"Could not create output video: "
                            f"{output_video}"
                        )

                video_writer.write(annotated)

                frames_processed += 1

                if frames_processed % 10 == 0:
                    print(
                        f"Processed {frames_processed} frames..."
                    )

    finally:
        cap.release()

        if video_writer is not None:
            video_writer.release()

        if player_tracker is not None:
            try:
                player_tracker.cleanup()
            except Exception as exc:
                print(
                    f"Warning during tracker cleanup: {exc}",
                    file=sys.stderr,
                )

    if frames_processed == 0:
        raise RuntimeError(
            "No frames processed. Check the video and --max-frames."
        )

    print("\n===== AthletiQ Joint Tracking Summary =====")
    print(f"Frames processed: {frames_processed}")
    print(f"Player track rows: {player_rows}")
    print(f"Unique canonical player IDs: {len(unique_player_ids)}")
    print(f"Ball states: {ball_states}")
    print(f"Player CSV: {player_csv}")
    print(f"Ball CSV: {ball_csv}")
    print(f"Joint CSV: {joint_csv}")
    print(f"Annotated video: {output_video}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
