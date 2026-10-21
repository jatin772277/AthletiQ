"""Validated image-to-pitch homography utilities for AthletiQ.

Image points use pixels (x right, y down).
Pitch points use metres (x along pitch length, y along pitch width).

A successful fit is not automatically a validated calibration. Independent
validation landmarks must be supplied before physical metrics are trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


Point2D = tuple[float, float]


@dataclass
class CalibrationResult:
    homography: np.ndarray
    fit_errors_m: list[float]
    validation_errors_m: list[float]
    mean_validation_error_m: float | None
    max_validation_error_m: float | None
    validated: bool
    reason: str


def _points(values: Sequence[Sequence[float]], name: str) -> np.ndarray:
    points = np.asarray(values, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2).")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} contains non-finite coordinates.")

    return points


def _project(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        points.astype(np.float64).reshape(-1, 1, 2),
        homography,
    ).reshape(-1, 2)


def calibrate_homography(
    image_points_px: Sequence[Sequence[float]],
    pitch_points_m: Sequence[Sequence[float]],
    validation_image_points_px: Sequence[Sequence[float]],
    validation_pitch_points_m: Sequence[Sequence[float]],
    *,
    max_mean_validation_error_m: float = 1.5,
    max_validation_error_m: float = 3.0,
) -> CalibrationResult:
    """Fit a homography and assess it on separate known landmarks.

    At least four non-collinear fitting points and three independent,
    non-collinear validation points are required. Validation thresholds
    are initial engineering
    thresholds, not universal guarantees of accuracy.
    """
    image = _points(image_points_px, "image_points_px")
    pitch = _points(pitch_points_m, "pitch_points_m")
    if len(validation_image_points_px) < 3:
        raise ValueError(
            "At least three independent validation points are required."
        )
    if len(validation_pitch_points_m) < 3:
        raise ValueError(
            "At least three independent validation points are required."
        )

    val_image = _points(
        validation_image_points_px, "validation_image_points_px"
    )
    val_pitch = _points(
        validation_pitch_points_m, "validation_pitch_points_m"
    )

    if len(image) < 4:
        raise ValueError("At least four fitting correspondences are required.")
    if len(image) != len(pitch):
        raise ValueError("Fitting image and pitch point counts must match.")
    if len(val_image) < 3:
        raise ValueError("At least three independent validation points are required.")
    if len(val_image) != len(val_pitch):
        raise ValueError("Validation image and pitch point counts must match.")
    if len(np.unique(val_image, axis=0)) < 3:
        raise ValueError("Validation image points must be distinct.")
    if len(np.unique(val_pitch, axis=0)) < 3:
        raise ValueError("Validation pitch points must be distinct.")
    if np.linalg.matrix_rank(val_image - val_image.mean(axis=0)) < 2:
        raise ValueError("Validation image points are collinear.")
    if np.linalg.matrix_rank(val_pitch - val_pitch.mean(axis=0)) < 2:
        raise ValueError("Validation pitch points are collinear.")
    if len(np.unique(image, axis=0)) < 4:
        raise ValueError("Fitting image points must be distinct.")
    if len(np.unique(pitch, axis=0)) < 4:
        raise ValueError("Fitting pitch points must be distinct.")

    # Check that points span a 2-D area rather than lying on a single line.
    if np.linalg.matrix_rank(image - image.mean(axis=0)) < 2:
        raise ValueError("Fitting image points are collinear.")
    if np.linalg.matrix_rank(pitch - pitch.mean(axis=0)) < 2:
        raise ValueError("Fitting pitch points are collinear.")

    homography, _ = cv2.findHomography(
        image.astype(np.float64),
        pitch.astype(np.float64),
        method=0,
    )

    if homography is None or not np.isfinite(homography).all():
        raise ValueError("Could not compute a finite homography.")

    fit_projected = _project(image, homography)
    fit_errors = np.linalg.norm(fit_projected - pitch, axis=1)

    val_projected = _project(val_image, homography)
    val_errors = np.linalg.norm(val_projected - val_pitch, axis=1)

    mean_error = float(np.mean(val_errors))
    max_error = float(np.max(val_errors))

    validated = (
        mean_error <= max_mean_validation_error_m
        and max_error <= max_validation_error_m
    )

    reason = (
        "Independent validation errors passed configured thresholds."
        if validated
        else "Independent validation errors exceed configured thresholds."
    )

    return CalibrationResult(
        homography=homography,
        fit_errors_m=fit_errors.tolist(),
        validation_errors_m=val_errors.tolist(),
        mean_validation_error_m=mean_error,
        max_validation_error_m=max_error,
        validated=validated,
        reason=reason,
    )


def image_to_pitch(
    image_points_px: Sequence[Sequence[float]],
    homography: np.ndarray,
) -> np.ndarray:
    """Transform image points to pitch coordinates in metres."""
    points = _points(image_points_px, "image_points_px")

    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("homography must be a finite 3x3 matrix.")

    return _project(points, matrix)
