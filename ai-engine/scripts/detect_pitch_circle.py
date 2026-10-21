#!/usr/bin/env python3
"""Detect and diagnose the projected centre circle in an AthletiQ frame.

This tool produces diagnostic geometry only. It does not claim to calibrate
the complete pitch or produce physical player metrics.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="outputs/calibration/football2_frame0.jpg",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/calibration",
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        default=[170, 220, 620, 350],
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Region of interest in original-image pixels",
    )
    args = parser.parse_args()

    image_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Cannot read input image: {image_path}")

    height, width = image.shape[:2]
    x1, y1, x2, y2 = args.roi

    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise SystemExit(
            f"ROI {args.roi} is outside image bounds {width}x{height}"
        )

    crop = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # White pitch paint is generally bright and low-saturation.
    # Watermarks and players can also pass this filter, so candidates
    # must be inspected rather than trusted automatically.
    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 125], dtype=np.uint8),
        np.array([180, 115, 255], dtype=np.uint8),
    )

    edges = cv2.Canny(gray, 35, 100)
    combined = cv2.bitwise_or(edges, white_mask)
    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )

    contours, _ = cv2.findContours(
        combined,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )

    candidates = []
    for contour in contours:
        if len(contour) < 20:
            continue

        try:
            (cx, cy), (axis_a, axis_b), angle = cv2.fitEllipse(contour)
        except cv2.error:
            continue

        major = max(axis_a, axis_b)
        minor = min(axis_a, axis_b)

        # Loose geometric filters: the projected centre circle should
        # be a broad ellipse in this broadcast view.
        if major < 100 or minor < 15:
            continue
        if major > 500 or minor > 200:
            continue
        if minor / major < 0.08:
            continue

        contour_area = abs(cv2.contourArea(contour))
        if contour_area < 50:
            continue

        # Prefer candidates centred near the middle of the ROI and
        # with a broad horizontal extent.
        expected_cx = (x2 - x1) / 2
        expected_cy = (y2 - y1) / 2
        centre_penalty = (
            abs(cx - expected_cx) / max(1, x2 - x1)
            + abs(cy - expected_cy) / max(1, y2 - y1)
        )
        score = (
            (major / max(1, minor))
            * min(len(contour), 500)
            / (1 + 4 * centre_penalty)
        )

        candidates.append({
            "ellipse": ((float(cx), float(cy)),
                        (float(axis_a), float(axis_b)),
                        float(angle)),
            "score": float(score),
            "contour_points": int(len(contour)),
            "contour_area_px2": float(contour_area),
        })

    candidates.sort(key=lambda item: item["score"], reverse=True)

    preview = image.copy()
    cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 180, 0), 1)

    for rank, candidate in enumerate(candidates[:10], start=1):
        (cx, cy), (axis_a, axis_b), angle = candidate["ellipse"]
        centre = (int(round(cx + x1)), int(round(cy + y1)))
        axes = (
            max(1, int(round(axis_a / 2))),
            max(1, int(round(axis_b / 2))),
        )
        color = (0, 255, 0) if rank == 1 else (0, 180, 255)
        cv2.ellipse(preview, centre, axes, angle, 0, 360, color, 2)
        cv2.putText(
            preview,
            f"#{rank}",
            (centre[0] + 5, centre[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    preview_path = output_dir / "football2_circle_detection_preview.jpg"
    edges_path = output_dir / "football2_circle_detection_edges.jpg"
    json_path = output_dir / "football2_circle_detection.json"

    cv2.imwrite(str(preview_path), preview)
    cv2.imwrite(str(edges_path), combined)

    report = {
        "input_image": str(image_path),
        "image_size_px": {"width": width, "height": height},
        "roi_xyxy": [x1, y1, x2, y2],
        "method": "thresholding, Canny edges, contour ellipse fitting",
        "candidate_count": len(candidates),
        "candidates": [
            {
                "rank": rank,
                "center_px": [
                    candidate["ellipse"][0][0] + x1,
                    candidate["ellipse"][0][1] + y1,
                ],
                "axis_lengths_px": list(candidate["ellipse"][1]),
                "angle_degrees": candidate["ellipse"][2],
                "contour_points": candidate["contour_points"],
                "score": candidate["score"],
            }
            for rank, candidate in enumerate(candidates[:10], start=1)
        ],
        "full_pitch_calibration_validated": False,
        "physical_metrics_enabled": False,
        "warning": (
            "Ellipse candidates can fit players, watermark edges, or other "
            "contours. Inspect the overlay. This is not a pitch homography."
        ),
    }

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("========== PITCH CIRCLE DETECTION ==========")
    print(f"Image:       {image_path}")
    print(f"ROI:         {args.roi}")
    print(f"Candidates:  {len(candidates)}")
    print(f"Preview:     {preview_path}")
    print(f"Edge map:    {edges_path}")
    print(f"JSON report: {json_path}")

    if candidates:
        best = candidates[0]
        (cx, cy), (a, b), angle = best["ellipse"]
        print(
            "Top candidate: "
            f"centre=({cx + x1:.1f}, {cy + y1:.1f}), "
            f"axes=({a:.1f}, {b:.1f}), angle={angle:.1f}°"
        )
    else:
        print("No candidates passed the filters.")
        print("Try adjusting --roi to include more of the centre circle.")

    print("Full pitch calibration: NOT validated")
    print("===========================================")


if __name__ == "__main__":
    main()
