
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import math


@dataclass
class TrackState:
    """
    Persistent identity state for one canonical player.

    Velocity is estimated from a short trajectory window rather
    than a single frame-to-frame displacement.
    """

    canonical_id: int
    current_track_id: int

    position: Tuple[float, float]
    previous_position: Optional[Tuple[float, float]] = None

    velocity: Tuple[float, float] = (0.0, 0.0)

    bbox: Optional[List[float]] = None

    last_frame: int = 0
    missed_frames: int = 0

    team_id: Optional[int] = None
    identity_confidence: float = 1.0

    # Latest appearance embedding from the production Re-ID model.
    # Expected shape: (512,)
    reid_embedding: Optional[object] = None

    track_history: List[int] = field(default_factory=list)

    # Recent (frame, x, y) observations.
    position_history: List[Tuple[int, float, float]] = field(
        default_factory=list
    )

    # Number of observations used for velocity estimation.
    velocity_window: int = 5

    def update(
        self,
        position: Tuple[float, float],
        bbox: Optional[List[float]],
        frame_id: int,
        track_id: int,
        identity_confidence: Optional[float] = None,
        reid_embedding=None,
    ):
        """
        Update the canonical track with a new observation.
        """

        self.previous_position = self.position
        self.position = position

        self.bbox = bbox
        self.current_track_id = track_id
        self.last_frame = frame_id
        self.missed_frames = 0

        if identity_confidence is not None:
            self.identity_confidence = identity_confidence

        # ----------------------------------------------------
        # Re-ID appearance embedding
        # ----------------------------------------------------

        if reid_embedding is not None:
            self.reid_embedding = reid_embedding

        # ----------------------------------------------------
        # Track ID history
        # ----------------------------------------------------

        if (
            not self.track_history
            or self.track_history[-1] != track_id
        ):
            self.track_history.append(track_id)

        # ----------------------------------------------------
        # Position history
        # ----------------------------------------------------

        self.position_history.append(
            (
                frame_id,
                float(position[0]),
                float(position[1]),
            )
        )

        # Keep only recent observations.
        self.position_history = self.position_history[
            -self.velocity_window:
        ]

        # ----------------------------------------------------
        # Robust velocity estimation
        # ----------------------------------------------------

        self.velocity = self._estimate_velocity()

    def _estimate_velocity(self):
        """
        Estimate velocity using linear regression over the
        recent trajectory window.

        This is much more stable than:

            current_position - previous_position

        because a single noisy detection can otherwise produce
        an unrealistically large velocity.
        """

        if len(self.position_history) < 2:
            return (0.0, 0.0)

        frames = [
            float(item[0])
            for item in self.position_history
        ]

        xs = [
            float(item[1])
            for item in self.position_history
        ]

        ys = [
            float(item[2])
            for item in self.position_history
        ]

        # ----------------------------------------------------
        # Mean-center frame values
        # ----------------------------------------------------

        mean_frame = sum(frames) / len(frames)

        centered_frames = [
            f - mean_frame
            for f in frames
        ]

        denominator = sum(
            value * value
            for value in centered_frames
        )

        if denominator <= 1e-12:
            return (0.0, 0.0)

        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)

        velocity_x = sum(
            (f - mean_frame) * (x - mean_x)
            for f, x in zip(frames, xs)
        ) / denominator

        velocity_y = sum(
            (f - mean_frame) * (y - mean_y)
            for f, y in zip(frames, ys)
        ) / denominator

        return (
            float(velocity_x),
            float(velocity_y),
        )

    def mark_missing(self):
        """
        Mark this canonical identity as temporarily missing.
        """
        self.missed_frames += 1

    def speed(self):
        """
        Current speed in pixels/frame.
        """
        return math.sqrt(
            self.velocity[0] ** 2
            + self.velocity[1] ** 2
        )
