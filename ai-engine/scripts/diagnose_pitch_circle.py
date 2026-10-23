#!/usr/bin/env python3
"""Diagnose centre-circle landmarks without claiming a full pitch calibration."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--points",
        default="outputs/calibration/football2_circle_points.json",
    )
    parser.add_argument(
        "--output",
        default="outputs/calibration/football2_circle_diagnostic.json",
    )
    parser.add_argument("--radius", type=float, default=9.15)
    args = parser.parse_args()

    points_path = Path(args.points)
    if not points_path.exists():
        raise SystemExit(
            f"Missing {points_path}. Create it using the instructions provided."
        )

    points = json.loads(points_path.read_text(encoding="utf-8"))
    if not isinstance(points, list) or len(points) < 5:
        raise SystemExit("Provide at least five image points as a JSON list.")

    xy = np.array([[p["x"], p["y"]] for p in points], dtype=np.float32)

    if not np.isfinite(xy).all():
        raise SystemExit("Points contain invalid coordinates.")

    if len(np.unique(xy, axis=0)) < 5:
        raise SystemExit("Please provide five distinct points.")

    # fitEllipse requires at least five points. Every supplied point must
    # lie on the circle boundary; the circle centre must NOT be included.
    ellipse = cv2.fitEllipse(xy.reshape(-1, 1, 2))
    (cx, cy), (axis_a, axis_b), angle = ellipse

    semi_a = axis_a / 2.0
    semi_b = axis_b / 2.0

    if min(semi_a, semi_b) < 1:
        raise SystemExit("Fitted ellipse is degenerate.")

    theta = np.deg2rad(angle)
    dx = xy[:, 0] - cx
    dy = xy[:, 1] - cy

    # Transform points into the fitted ellipse's local coordinate system.
    local_x = dx * np.cos(theta) + dy * np.sin(theta)
    local_y = -dx * np.sin(theta) + dy * np.cos(theta)

    normalized_radius = np.sqrt(
        (local_x / semi_a) ** 2 + (local_y / semi_b) ** 2
    )
    residual_px_approx = (
        np.abs(normalized_radius - 1.0) * (semi_a + semi_b) / 2.0
    )

    result = {
        "purpose": "Centre-circle image geometry diagnostic only",
        "input_points_px": xy.tolist(),
        "fitted_ellipse": {
            "center_px": [float(cx), float(cy)],
            "full_axis_lengths_px": [
                float(axis_a),
                float(axis_b),
            ],
            "angle_degrees": float(angle),
        },
        "approximate_boundary_residual_px": {
            "mean": float(np.mean(residual_px_approx)),
            "maximum": float(np.max(residual_px_approx)),
        },
        "assumed_real_circle_radius_metres": args.radius,
        "limitations": [
            "Every input point must lie on the circle boundary.",
            "Do not include the circle centre as a boundary point.",
            "Ellipse fit does not establish a unique image-to-pitch homography.",
            "No physical player distances or speeds are produced.",
            "Perspective, occlusion, and clicking error can affect the fit.",
        ],
        "full_pitch_calibration_validated": False,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("========== CIRCLE DIAGNOSTIC ==========")
    print(f"Fitted centre (pixels): ({cx:.2f}, {cy:.2f})")
    print(f"Ellipse axes (pixels): {axis_a:.2f}, {axis_b:.2f}")
    print(
        "Mean approximate boundary residual: "
        f"{np.mean(residual_px_approx):.2f} px"
    )
    print(f"Saved: {output_path}")
    print("Full pitch calibration: NOT validated")
    print("=======================================")


if __name__ == "__main__":
    main()
