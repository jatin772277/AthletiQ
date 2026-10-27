import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "core"
    / "spatial"
    / "pitch_calibration.py"
)

SPEC = importlib.util.spec_from_file_location(
    "pitch_calibration", MODULE_PATH
)
pitch_calibration = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pitch_calibration
SPEC.loader.exec_module(pitch_calibration)

calibrate_homography = pitch_calibration.calibrate_homography
image_to_pitch = pitch_calibration.image_to_pitch


def test_valid_homography_passes_independent_validation():
    # Synthetic correspondences generated from a known mapping.
    image_points = [
        [100, 100],
        [500, 100],
        [500, 300],
        [100, 300],
        [300, 200],
    ]

    pitch_points = [
        [0, 0],
        [100, 0],
        [100, 50],
        [0, 50],
        [50, 25],
    ]

    validation_image = [[200, 150], [400, 250], [300, 120]]
    validation_pitch = [[25, 12.5], [75, 37.5], [50, 5]]

    result = calibrate_homography(
        image_points,
        pitch_points,
        validation_image,
        validation_pitch,
        max_mean_validation_error_m=0.01,
        max_validation_error_m=0.01,
    )

    assert result.validated
    assert result.mean_validation_error_m < 0.01
    assert result.max_validation_error_m < 0.01


def test_collinear_points_are_rejected():
    points = [[0, 0], [1, 1], [2, 2], [3, 3]]
    validation_image = [[0, 0], [1, 0], [0, 1]]
    validation_pitch = [[0, 0], [1, 0], [0, 1]]

    with pytest.raises(ValueError, match="collinear"):
        calibrate_homography(
            points,
            points,
            validation_image,
            validation_pitch,
        )


def test_missing_validation_points_are_rejected():
    points = [[0, 0], [4, 0], [4, 4], [0, 4]]

    with pytest.raises(ValueError, match="validation point"):
        calibrate_homography(
            points,
            points,
            [],
            [],
        )


def test_image_points_transform_correctly():
    image_points = [[0, 0], [4, 0], [4, 4], [0, 4]]
    pitch_points = [[0, 0], [40, 0], [40, 20], [0, 20]]

    result = calibrate_homography(
        image_points,
        pitch_points,
        [[2, 2], [1, 1], [3, 2]],
        [[20, 10], [10, 5], [30, 10]],
        max_mean_validation_error_m=0.01,
        max_validation_error_m=0.01,
    )

    transformed = image_to_pitch([[2, 2]], result.homography)

    np.testing.assert_allclose(
        transformed,
        [[20, 10]],
        atol=1e-5,
    )



def test_too_few_validation_points_are_rejected():
    points = [[0, 0], [4, 0], [4, 4], [0, 4]]

    with pytest.raises(ValueError, match="three independent validation points"):
        calibrate_homography(
            points,
            points,
            [[1, 1], [2, 2]],
            [[1, 1], [2, 2]],
        )


def test_collinear_validation_points_are_rejected():
    image = [[0, 0], [4, 0], [4, 4], [0, 4]]
    pitch = [[0, 0], [40, 0], [40, 40], [0, 40]]

    with pytest.raises(ValueError, match="Validation image points are collinear"):
        calibrate_homography(
            image,
            pitch,
            [[1, 1], [2, 2], [3, 3]],
            [[10, 10], [20, 20], [30, 30]],
        )
