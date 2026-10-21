
from dataclasses import dataclass
from typing import Optional
import math

from core.detection.ball_detector import BallDetection


@dataclass
class BallTrack:
    """Ball position and tracking state for one video frame."""

    frame_id: int
    x: float
    y: float
    confidence: float
    state: str  # detected, predicted, or lost
    missed_frames: int = 0

    @property
    def center(self) -> tuple[float, float]:
        return self.x, self.y


class AthletiQBallTracker:
    """
    Single-ball tracker with observation-based motion estimation.

    Predictions never become observations. Velocity is updated only
    when a real detection is accepted. Recovery gradually expands
    its search region after prolonged tracking loss.
    """

    def __init__(
        self,
        max_distance: float = 100.0,
        max_gap: int = 5,
        velocity_smoothing: float = 0.65,
        reacquire_distance: float = 160.0,
    ) -> None:
        if max_distance <= 0:
            raise ValueError("max_distance must be positive")
        if max_gap < 0:
            raise ValueError("max_gap cannot be negative")
        if not 0.0 <= velocity_smoothing < 1.0:
            raise ValueError("velocity_smoothing must be in [0, 1)")
        if reacquire_distance <= 0:
            raise ValueError("reacquire_distance must be positive")

        self.max_distance = float(max_distance)
        self.max_gap = int(max_gap)
        self.velocity_smoothing = float(velocity_smoothing)
        self.reacquire_distance = float(reacquire_distance)

        self.reset()

    def reset(self) -> None:
        """Clear tracker state before processing a new video."""
        self._last_track: Optional[BallTrack] = None

        # Last actual YOLO observation; predictions never overwrite it.
        self._last_observed_frame: Optional[int] = None
        self._last_observed_x: Optional[float] = None
        self._last_observed_y: Optional[float] = None

        self._vx = 0.0
        self._vy = 0.0
        self._lost = False
        self._last_frame_id: Optional[int] = None

    def _make_track(
        self,
        frame_id: int,
        x: float,
        y: float,
        confidence: float,
        state: str,
        missed_frames: int = 0,
    ) -> BallTrack:
        track = BallTrack(
            frame_id=frame_id,
            x=float(x),
            y=float(y),
            confidence=float(confidence),
            state=state,
            missed_frames=missed_frames,
        )
        self._last_track = track
        return track

    def _start_track(
        self,
        detection: BallDetection,
        frame_id: int,
    ) -> BallTrack:
        """Initialize or reacquire the track from a real detection."""
        x, y = detection.center

        self._last_observed_frame = frame_id
        self._last_observed_x = x
        self._last_observed_y = y

        # Reset velocity after reacquisition because the previous
        # trajectory may no longer describe the ball's current motion.
        self._vx = 0.0
        self._vy = 0.0
        self._lost = False

        return self._make_track(
            frame_id,
            x,
            y,
            detection.confidence,
            "detected",
        )

    def update(
        self,
        detections: list[BallDetection],
        frame_id: int,
    ) -> Optional[BallTrack]:
        """
        Associate detections with the tracked ball.

        Returns:
            A detected or predicted BallTrack, or None when the ball
            cannot be located confidently enough.
        """
        if frame_id < 0:
            raise ValueError("frame_id cannot be negative")

        if (
            self._last_frame_id is not None
            and frame_id <= self._last_frame_id
        ):
            raise ValueError("frame_id must increase on every update")

        self._last_frame_id = frame_id
        detections = list(detections)

        # First valid observation initializes the tracker.
        if self._last_observed_frame is None:
            if not detections:
                return None

            chosen = max(detections, key=lambda d: d.confidence)
            return self._start_track(chosen, frame_id)

        observed_frame = self._last_observed_frame
        observed_x = self._last_observed_x
        observed_y = self._last_observed_y

        assert observed_x is not None
        assert observed_y is not None

        elapsed = frame_id - observed_frame

        # Predict from the last REAL observation.
        predicted_x = observed_x + self._vx * elapsed
        predicted_y = observed_y + self._vy * elapsed

        # ---------------------------------------------------------
        # Reacquisition after the previous trajectory was lost
        # ---------------------------------------------------------
        if self._lost:
            extra_gap = max(0, elapsed - self.max_gap)

            recovery_distance = min(
                self.reacquire_distance
                + self.max_distance * math.sqrt(extra_gap),
                self.max_distance * 4.0,
            )

            candidates = []

            for detection in detections:
                x, y = detection.center

                distance_observed = math.hypot(
                    x - observed_x,
                    y - observed_y,
                )
                distance_predicted = math.hypot(
                    x - predicted_x,
                    y - predicted_y,
                )

                # Allow recovery near either the last observation or
                # the motion prediction, with a bounded search radius.
                distance = min(
                    distance_observed,
                    distance_predicted,
                )

                if distance <= recovery_distance:
                    score = distance - 5.0 * detection.confidence
                    candidates.append((score, detection))

            if not candidates:
                return None

            _, chosen = min(candidates, key=lambda item: item[0])
            return self._start_track(chosen, frame_id)

        # ---------------------------------------------------------
        # Normal tracking: associate detections with the prediction
        # ---------------------------------------------------------
        allowed_distance = min(
            self.max_distance * math.sqrt(elapsed),
            self.max_distance * 2.5,
        )

        candidates = []

        for detection in detections:
            x, y = detection.center

            distance = math.hypot(
                x - predicted_x,
                y - predicted_y,
            )

            if distance <= allowed_distance:
                # Distance is the primary signal; confidence breaks
                # close ties without dominating spatial association.
                score = distance - 5.0 * detection.confidence
                candidates.append((score, detection))

        if candidates:
            _, chosen = min(candidates, key=lambda item: item[0])
            x, y = chosen.center

            # Estimate velocity using only real observations.
            measured_vx = (x - observed_x) / elapsed
            measured_vy = (y - observed_y) / elapsed

            alpha = self.velocity_smoothing

            self._vx = (
                alpha * self._vx
                + (1.0 - alpha) * measured_vx
            )
            self._vy = (
                alpha * self._vy
                + (1.0 - alpha) * measured_vy
            )

            self._last_observed_frame = frame_id
            self._last_observed_x = x
            self._last_observed_y = y

            return self._make_track(
                frame_id,
                x,
                y,
                chosen.confidence,
                "detected",
            )

        # ---------------------------------------------------------
        # No suitable detection: predict briefly, then mark as lost
        # ---------------------------------------------------------
        missed_frames = elapsed

        if missed_frames > self.max_gap:
            self._lost = True
            return None

        return self._make_track(
            frame_id,
            predicted_x,
            predicted_y,
            0.0,
            "predicted",
            missed_frames=missed_frames,
        )
