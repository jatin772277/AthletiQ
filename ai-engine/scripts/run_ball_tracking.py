
import argparse
import csv
import sys
from pathlib import Path

import cv2

# Make the ai-engine directory importable when this script
# is launched from the repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]
AI_ENGINE_DIR = REPO_ROOT / "ai-engine"

sys.path.insert(0, str(AI_ENGINE_DIR))

from core.detection.ball_detector import AthletiQBallDetector
from core.tracking.ball_tracker import AthletiQBallTracker


DEFAULT_MODEL = (
    REPO_ROOT
    / "ai-engine"
    / "models"
    / "athletiq_ball_detector_champion_1280.pt"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run AthletiQ ball detection and tracking on a video."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input football video.",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Path to the trained YOLO ball detector.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs" / "ball_tracking"),
        help="Directory for the output video and CSV.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.05,
        help="Minimum YOLO detection confidence.",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=5,
        help="Maximum missed frames before the track is lost.",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=100.0,
        help="Base association distance in pixels.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, e.g. '0', 'cpu', or omit for automatic.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame limit for a quick test.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the annotated video while processing.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)

    video_stem = input_path.stem
    output_video = output_dir / f"{video_stem}_ball_tracked.mp4"
    output_csv = output_dir / f"{video_stem}_ball_tracks.csv"

    detector = AthletiQBallDetector(
        model_path=model_path,
        confidence=args.conf,
        device=args.device,
    )

    tracker = AthletiQBallTracker(
        max_distance=args.max_distance,
        max_gap=args.max_gap,
    )

    capture = cv2.VideoCapture(str(input_path))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Could not read video dimensions")

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_video}")

    frame_id = 0
    detected_frames = 0
    predicted_frames = 0
    lost_frames = 0

    try:
        with output_csv.open("w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "frame_id",
                    "timestamp_seconds",
                    "detection_count",
                    "x",
                    "y",
                    "confidence",
                    "state",
                    "missed_frames",
                ],
            )
            csv_writer.writeheader()

            while True:
                if (
                    args.max_frames is not None
                    and frame_id >= args.max_frames
                ):
                    break

                success, frame = capture.read()
                if not success:
                    break

                detections = detector.detect(frame)
                track = tracker.update(detections, frame_id)

                # Draw candidate detections lightly for debugging.
                for detection in detections:
                    x1, y1, x2, y2 = map(
                        int,
                        detection.bbox,
                    )
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (150, 150, 150),
                        1,
                    )

                if track is not None:
                    x = int(round(track.x))
                    y = int(round(track.y))

                    if track.state == "detected":
                        detected_frames += 1
                        color = (0, 255, 0)
                        label = f"BALL {track.confidence:.2f}"
                    else:
                        predicted_frames += 1
                        color = (0, 165, 255)
                        label = f"BALL PREDICTED ({track.missed_frames})"

                    cv2.circle(frame, (x, y), 8, color, -1)
                    cv2.circle(frame, (x, y), 13, color, 2)
                    cv2.putText(
                        frame,
                        label,
                        (max(5, x + 10), max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

                    csv_writer.writerow(
                        {
                            "frame_id": frame_id,
                            "timestamp_seconds": round(frame_id / fps, 4),
                            "detection_count": len(detections),
                            "x": round(track.x, 2),
                            "y": round(track.y, 2),
                            "confidence": round(track.confidence, 4),
                            "state": track.state,
                            "missed_frames": track.missed_frames,
                        }
                    )
                else:
                    lost_frames += 1

                    cv2.putText(
                        frame,
                        "BALL LOST",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    csv_writer.writerow(
                        {
                            "frame_id": frame_id,
                            "timestamp_seconds": round(frame_id / fps, 4),
                            "detection_count": len(detections),
                            "x": "",
                            "y": "",
                            "confidence": "",
                            "state": "lost",
                            "missed_frames": "",
                        }
                    )

                writer.write(frame)

                if args.show:
                    cv2.imshow("AthletiQ Ball Tracker", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        frame_id += 1
                        break

                frame_id += 1

                if frame_id % 100 == 0:
                    print(f"Processed {frame_id} frames...")

    
    finally:
        capture.release()
        writer.release()

        if args.show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


    print("\n===== AthletiQ Ball Tracking Summary =====")
    print(f"Frames processed: {frame_id}")
    print(f"Detected frames:  {detected_frames}")
    print(f"Predicted frames: {predicted_frames}")
    print(f"Lost frames:      {lost_frames}")
    print(f"Annotated video:  {output_video}")
    print(f"Tracking CSV:     {output_csv}")


if __name__ == "__main__":
    main()
