
import pytest

from core.detection.ball_detector import BallDetection
from core.tracking.ball_tracker import AthletiQBallTracker


def detection(x, y, confidence=0.9, size=10):
    """Create a synthetic detection centered at (x, y)."""
    half = size / 2
    return BallDetection(
        x1=x - half,
        y1=y - half,
        x2=x + half,
        y2=y + half,
        confidence=confidence,
    )


def test_initializes_from_highest_confidence_detection():
    tracker = AthletiQBallTracker()

    result = tracker.update(
        [
            detection(100, 100, 0.7),
            detection(200, 150, 0.95),
        ],
        frame_id=0,
    )

    assert result is not None
    assert result.state == "detected"
    assert result.center == (200, 150)
    assert result.confidence == pytest.approx(0.95)


def test_tracks_a_moving_ball():
    tracker = AthletiQBallTracker(
        max_distance=100,
        velocity_smoothing=0.0,
    )

    first = tracker.update([detection(100, 100)], frame_id=0)
    second = tracker.update([detection(110, 100)], frame_id=1)

    assert first is not None
    assert second is not None
    assert second.state == "detected"
    assert second.center == (110, 100)


def test_predicts_position_when_detection_is_missing():
    tracker = AthletiQBallTracker(
        max_distance=100,
        max_gap=2,
        velocity_smoothing=0.0,
    )

    tracker.update([detection(100, 100)], frame_id=0)
    tracker.update([detection(110, 100)], frame_id=1)

    result = tracker.update([], frame_id=2)

    assert result is not None
    assert result.state == "predicted"
    assert result.x == pytest.approx(120)
    assert result.y == pytest.approx(100)
    assert result.missed_frames == 1


def test_marks_track_lost_after_max_gap():
    tracker = AthletiQBallTracker(max_gap=2)

    tracker.update([detection(100, 100)], frame_id=0)

    assert tracker.update([], frame_id=1) is not None
    assert tracker.update([], frame_id=2) is not None
    assert tracker.update([], frame_id=3) is None


def test_rejects_a_detection_too_far_from_prediction():
    tracker = AthletiQBallTracker(max_distance=50, max_gap=2)

    tracker.update([detection(100, 100)], frame_id=0)

    result = tracker.update(
        [detection(500, 500)],
        frame_id=1,
    )

    assert result is not None
    assert result.state == "predicted"
    assert result.center == (100, 100)


def test_reset_clears_previous_tracking_state():
    tracker = AthletiQBallTracker()

    tracker.update([detection(100, 100)], frame_id=0)
    tracker.reset()

    result = tracker.update(
        [detection(300, 200, 0.8)],
        frame_id=0,
    )

    assert result is not None
    assert result.center == (300, 200)
    assert result.state == "detected"


def test_frame_ids_must_increase():
    tracker = AthletiQBallTracker()

    tracker.update([detection(100, 100)], frame_id=5)

    with pytest.raises(ValueError):
        tracker.update([detection(110, 100)], frame_id=5)
