#!/usr/bin/env python3
"""Interactive pitch calibration for AthletiQ.

Click identifiable pitch landmarks in the video frame, then enter their
known pitch coordinates in metres. The script computes an image-to-pitch
homography and saves the calibration as JSON.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


WINDOW_NAME = "AthletiQ Pitch Calibration"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate one football camera view using pitch landmarks."
    )
    parser.add_argument(
        "--input",
        default="datasets/football/raw/videos/football2.mp4",
        help="Input video path",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index to calibrate (default: 0)",
    )
    parser.add_argument(
        "--length",
        type=float,
        default=105.0,
        help="Assumed pitch length in metres (default: 105)",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=68.0,
        help="Assumed pitch width in metres (default: 68)",
    )
    parser.add_argument(
        "--output",
        default="outputs/calibration/football2_calibration_frame0.json",
        help="Output JSON path",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    video_path = Path(args.input)
    output_path = Path(args.output)

    if not video_path.is_file():
        sys.exit(f"ERROR: Video not found: {video_path}")

    if args.length <= 0 or args.width <= 0:
        sys.exit("ERROR: Pitch length and width must be positive.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"ERROR: Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    if args.frame < 0 or args.frame >= total_frames:
        cap.release()
        sys.exit(
            f"ERROR: Frame must be between 0 and {total_frames - 1}."
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        sys.exit(f"ERROR: Could not read frame {args.frame}.")

    height, width = frame.shape[:2]
    selected_points = []

    print("\n========== ATHLETIQ PITCH CALIBRATION ==========")
    print(f"Video:       {video_path}")
    print(f"Frame:       {args.frame}")
    print(f"Timestamp:   {args.frame / fps:.3f} seconds" if fps > 0 else "Timestamp: unknown")
    print(f"Resolution:  {width} x {height}")
    print(f"Pitch size:  {args.length} m x {args.width} m")
    print("\nClick at least FOUR identifiable pitch landmarks.")
    print("Good choices: line intersections, penalty-area corners,")
    print("or the halfway line meeting the touchline.")
    print("Avoid players, shadows, and points whose pitch position is uncertain.")
    print("Press ENTER when finished; press R to reset; ESC to cancel.\n")

    display = frame.copy()

    def redraw():
        nonlocal display
        display = frame.copy()

        for i, (x, y) in enumerate(selected_points, start=1):
            cv2.circle(display, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(
                display,
                str(i),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        instructions = (
            f"Landmarks: {len(selected_points)} | "
            "Click=add  ENTER=finish  R=reset  ESC=cancel"
        )
        cv2.rectangle(display, (0, 0), (width, 32), (0, 0, 0), -1)
        cv2.putText(
            display,
            instructions,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_points.append((int(x), int(y)))
            redraw()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    redraw()

    while True:
        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 10):  # Enter
            if len(selected_points) < 4:
                print("Select at least four landmarks before finishing.")
                continue
            break
        elif key in (ord("r"), ord("R")):
            selected_points.clear()
            redraw()
            print("Landmarks reset.")
        elif key == 27:  # Escape
            cv2.destroyAllWindows()
            sys.exit("Calibration cancelled.")

    cv2.destroyAllWindows()

    image_points = np.asarray(selected_points, dtype=np.float64)
    pitch_points = []

    print("\nFor each clicked landmark, enter its pitch coordinates in metres.")
    print(f"Coordinate system: x = 0 to {args.length} (length)")
    print(f"                   y = 0 to {args.width} (width)")
    print("Choose a consistent origin, for example one pitch corner.")
    print("Use actual known positions, not estimates based only on screen pixels.\n")

    for i, (ix, iy) in enumerate(selected_points, start=1):
        while True:
            try:
                entry = input(
                    f"Landmark {i} (image x={ix}, y={iy}) -> pitch x,y in metres: "
                ).strip()
                wx, wy = map(float, entry.replace(",", " ").split())

                if not (
                    0 <= wx <= args.length and
                    0 <= wy <= args.width
                ):
                    print("Coordinates must fall within the configured pitch dimensions.")
                    continue

                pitch_points.append([wx, wy])
                break
            except ValueError:
                print("Enter two numbers, for example: 52.5, 34")

    pitch_points = np.asarray(pitch_points, dtype=np.float64)

    if np.unique(image_points, axis=0).shape[0] < 4:
        sys.exit("ERROR: Image landmarks must be distinct.")

    if np.unique(pitch_points, axis=0).shape[0] < 4:
        sys.exit("ERROR: At least four distinct pitch coordinates are required.")

    # Reject a degenerate point layout before computing the homography.
    if np.linalg.matrix_rank(
        np.column_stack([image_points, np.ones(len(image_points))])
    ) < 3:
        sys.exit("ERROR: Image landmarks are degenerate or collinear.")

    homography, inlier_mask = cv2.findHomography(
        image_points,
        pitch_points,
        method=0,
    )

    if homography is None or not np.isfinite(homography).all():
        sys.exit("ERROR: Could not compute a valid homography.")

    projected = cv2.perspectiveTransform(
        image_points.reshape(-1, 1, 2),
        homography,
    ).reshape(-1, 2)

    errors = np.linalg.norm(projected - pitch_points, axis=1)
    mean_error = float(np.mean(errors))
    median_error = float(np.median(errors))
    max_error = float(np.max(errors))

    result = {
        "project": "AthletiQ",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "video": str(video_path),
        "frame_index": args.frame,
        "timestamp_seconds": args.frame / fps if fps > 0 else None,
        "image_size": {"width": width, "height": height},
        "pitch_dimensions_metres": {
            "length": args.length,
            "width": args.width,
            "status": "assumed; verify against the actual pitch",
        },
        "coordinate_convention": {
            "image": "origin at top-left; x right, y down; pixels",
            "pitch": "x along length, y along width; metres",
            "origin": "user-defined pitch corner",
        },
        "image_points_px": image_points.tolist(),
        "pitch_points_metres": pitch_points.tolist(),
        "image_to_pitch_homography": homography.tolist(),
        "calibration_point_reprojection_error_metres": {
            "mean": mean_error,
            "median": median_error,
            "maximum": max_error,
            "note": (
                "These are errors on the same points used to fit the homography. "
                "They are not independent validation errors."
            ),
        },
        "independent_validation_performed": False,
        "warnings": [
            "Valid only for this camera view; recalibrate after significant pan or zoom.",
            "Homography is for points on the pitch ground plane, not airborne ball positions.",
            "Validate on additional known landmarks before reporting physical player metrics.",
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n========== CALIBRATION RESULT ==========")
    print(f"Landmarks:             {len(image_points)}")
    print(f"Mean fit error:        {mean_error:.4f} m")
    print(f"Median fit error:      {median_error:.4f} m")
    print(f"Maximum fit error:     {max_error:.4f} m")
    print(f"Saved calibration:     {output_path}")
    print("\nIMPORTANT: A low fit error alone does not prove accurate calibration.")
    print("Validate against additional known pitch landmarks before using metres")
    print("for player speed or distance.")
    print("========================================")


if __name__ == "__main__":
    main()