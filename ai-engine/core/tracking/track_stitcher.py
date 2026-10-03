from typing import Dict, List, Tuple
import math

from track_state import TrackState


class TrackStitcher:
    """
    Canonical identity recovery layer on top of MixSort.

    Lifecycle:

        ACTIVE
          ↓
        LOST
          ↓
        trajectory-based matching
          ↓
        ACTIVE again

    Important:
    - Active identities are NEVER candidates for stitching.
    - New MixSort IDs can only reconnect to LOST identities.
    - Prediction uses the robust velocity stored in TrackState.
    - Velocity is capped before prediction to prevent noisy jumps.
    """

    def __init__(
        self,
        max_distance: float = 100.0,
        max_frame_gap: int = 30,
        velocity_weight: float = 0.30,
        min_stitch_confidence: float = 0.70,
        max_prediction_speed: float = 30.0,
    ):

        self.max_distance = max_distance
        self.max_frame_gap = max_frame_gap
        self.velocity_weight = velocity_weight
        self.min_stitch_confidence = min_stitch_confidence

        # Maximum velocity allowed for trajectory prediction.
        self.max_prediction_speed = max_prediction_speed

        # canonical_id -> TrackState
        self.states: Dict[int, TrackState] = {}

        # MixSort ID -> canonical ID
        self.track_to_canonical: Dict[int, int] = {}

        self.next_canonical_id = 1

        self.active_ids = set()
        self.lost_ids = set()

        self.total_stitches = 0
        self.total_new_players = 0

    # =========================================================
    # HELPERS
    # =========================================================

    @staticmethod
    def _center_from_bbox(bbox):
        x1, y1, x2, y2 = bbox

        return (
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
        )

    def _predict_position(
        self,
        state: TrackState,
        frame_id: int,
    ) -> Tuple[float, float]:

        gap = frame_id - state.last_frame

        vx, vy = state.velocity

        speed = math.sqrt(
            vx * vx + vy * vy
        )

        # -----------------------------------------------------
        # Cap prediction velocity.
        # -----------------------------------------------------

        if speed > self.max_prediction_speed:

            scale = (
                self.max_prediction_speed
                / speed
            )

            vx *= scale
            vy *= scale

        predicted_x = (
            state.position[0]
            + vx * gap
        )

        predicted_y = (
            state.position[1]
            + vy * gap
        )

        return (
            predicted_x,
            predicted_y,
        )

    def _distance(
        self,
        p1,
        p2,
    ):

        return math.sqrt(
            (p1[0] - p2[0]) ** 2
            +
            (p1[1] - p2[1]) ** 2
        )

    def _velocity_similarity(
        self,
        state: TrackState,
        new_position,
        frame_gap,
    ):
        """
        Compare observed displacement with the previous
        velocity direction.

        Returns [0, 1].
        """

        if frame_gap <= 0:
            return 1.0

        dx = (
            new_position[0]
            - state.position[0]
        )

        dy = (
            new_position[1]
            - state.position[1]
        )

        observed_speed = math.sqrt(
            dx * dx + dy * dy
        )

        if observed_speed < 1e-6:
            return 1.0

        vx, vy = state.velocity

        previous_speed = math.sqrt(
            vx * vx
            + vy * vy
        )

        if previous_speed < 1e-6:
            return 0.5

        # Expected displacement direction
        expected_x = vx / previous_speed
        expected_y = vy / previous_speed

        # Observed displacement direction
        observed_x = dx / observed_speed
        observed_y = dy / observed_speed

        cosine = (
            expected_x * observed_x
            +
            expected_y * observed_y
        )

        cosine = max(
            -1.0,
            min(1.0, cosine)
        )

        # Map [-1, 1] -> [0, 1]
        return (
            cosine + 1.0
        ) / 2.0

    # =========================================================
    # MATCHING
    # =========================================================

    def _find_best_lost_match(
        self,
        position,
        frame_id,
    ):

        best_id = None
        best_confidence = 0.0

        for canonical_id in list(
            self.lost_ids
        ):

            state = self.states[
                canonical_id
            ]

            gap = (
                frame_id
                - state.last_frame
            )

            if gap <= 0:
                continue

            if gap > self.max_frame_gap:
                continue

            # -------------------------------------------------
            # Predicted position
            # -------------------------------------------------

            predicted = self._predict_position(
                state,
                frame_id,
            )

            prediction_distance = self._distance(
                predicted,
                position,
            )

            # -------------------------------------------------
            # Current-position distance
            # -------------------------------------------------

            current_distance = self._distance(
                state.position,
                position,
            )

            # -------------------------------------------------
            # Reject clearly impossible matches.
            # -------------------------------------------------

            if (
                prediction_distance
                > self.max_distance
            ):
                continue

            # -------------------------------------------------
            # Position confidence
            # -------------------------------------------------

            position_confidence = max(
                0.0,
                1.0
                - (
                    prediction_distance
                    / self.max_distance
                ),
            )

            # -------------------------------------------------
            # Time confidence
            # -------------------------------------------------

            time_confidence = max(
                0.0,
                1.0
                - (
                    (gap - 1)
                    / self.max_frame_gap
                ),
            )

            # -------------------------------------------------
            # Velocity direction consistency
            # -------------------------------------------------

            velocity_confidence = (
                self._velocity_similarity(
                    state,
                    position,
                    gap,
                )
            )

            # -------------------------------------------------
            # Final confidence
            # -------------------------------------------------

            confidence = (
                0.60
                * position_confidence
                +
                self.velocity_weight
                * velocity_confidence
                +
                0.10
                * time_confidence
            )

            if confidence > best_confidence:

                best_confidence = confidence
                best_id = canonical_id

        return (
            best_id,
            best_confidence,
        )

    # =========================================================
    # UPDATE
    # =========================================================

    def update(
        self,
        mixsort_tracks: List[dict],
        frame_id: int,
    ):

        current_mixsort_ids = {
            int(track["track_id"])
            for track in mixsort_tracks
        }

        # =====================================================
        # STEP 1
        # Move disappeared ACTIVE identities to LOST.
        # =====================================================

        for canonical_id in list(
            self.active_ids
        ):

            state = self.states[
                canonical_id
            ]

            if (
                state.current_track_id
                not in current_mixsort_ids
            ):

                state.mark_missing()

                self.active_ids.discard(
                    canonical_id
                )

                self.lost_ids.add(
                    canonical_id
                )

        results = []

        # =====================================================
        # STEP 2
        # Process current MixSort tracks.
        # =====================================================

        for track in mixsort_tracks:

            mixsort_id = int(
                track["track_id"]
            )

            bbox = [
                float(v)
                for v in track["bbox"]
            ]

            position = self._center_from_bbox(
                bbox
            )

            score = float(
                track.get("score", 0.0)
            )

            stitched = False
            identity_confidence = 1.0

            # -------------------------------------------------
            # Existing MixSort → canonical mapping
            # -------------------------------------------------

            if mixsort_id in self.track_to_canonical:

                canonical_id = (
                    self.track_to_canonical[
                        mixsort_id
                    ]
                )

                state = self.states[
                    canonical_id
                ]

                state.update(
                    position=position,
                    bbox=bbox,
                    frame_id=frame_id,
                    track_id=mixsort_id,
                    identity_confidence=1.0,
                )

                self.active_ids.add(
                    canonical_id
                )

                self.lost_ids.discard(
                    canonical_id
                )

            else:

                # =============================================
                # New MixSort ID
                # =============================================

                (
                    matched_id,
                    confidence,
                ) = self._find_best_lost_match(
                    position,
                    frame_id,
                )

                if (
                    matched_id is not None
                    and confidence
                    >= self.min_stitch_confidence
                ):

                    # =========================================
                    # Successful stitch
                    # =========================================

                    canonical_id = matched_id

                    state = self.states[
                        canonical_id
                    ]

                    state.update(
                        position=position,
                        bbox=bbox,
                        frame_id=frame_id,
                        track_id=mixsort_id,
                        identity_confidence=confidence,
                    )

                    self.track_to_canonical[
                        mixsort_id
                    ] = canonical_id

                    self.active_ids.add(
                        canonical_id
                    )

                    self.lost_ids.discard(
                        canonical_id
                    )

                    self.total_stitches += 1

                    stitched = True
                    identity_confidence = (
                        confidence
                    )

                else:

                    # =========================================
                    # Completely new canonical identity
                    # =========================================

                    canonical_id = (
                        self.next_canonical_id
                    )

                    self.next_canonical_id += 1

                    state = TrackState(
                        canonical_id=canonical_id,
                        current_track_id=mixsort_id,
                        position=position,
                        bbox=bbox,
                        last_frame=frame_id,
                    )

                    state.update(
                        position=position,
                        bbox=bbox,
                        frame_id=frame_id,
                        track_id=mixsort_id,
                        identity_confidence=1.0,
                    )

                    self.states[
                        canonical_id
                    ] = state

                    self.track_to_canonical[
                        mixsort_id
                    ] = canonical_id

                    self.active_ids.add(
                        canonical_id
                    )

                    self.total_new_players += 1

            # -------------------------------------------------
            # Output
            # -------------------------------------------------

            results.append({
                "mixsort_id": mixsort_id,
                "canonical_id": canonical_id,
                "bbox": bbox,
                "center": position,
                "score": score,
                "stitched": stitched,
                "identity_confidence": (
                    identity_confidence
                ),
            })

        # =====================================================
        # STEP 3
        # Remove identities that have been lost too long.
        # =====================================================

        stale_ids = []

        for canonical_id in list(
            self.lost_ids
        ):

            state = self.states[
                canonical_id
            ]

            gap = (
                frame_id
                - state.last_frame
            )

            if gap > self.max_frame_gap:

                stale_ids.append(
                    canonical_id
                )

        for canonical_id in stale_ids:

            self.lost_ids.discard(
                canonical_id
            )

            # Remove the state.
            del self.states[
                canonical_id
            ]

            # Remove any old MixSort mappings.
            stale_mappings = [
                mixsort_id
                for mixsort_id, cid
                in self.track_to_canonical.items()
                if cid == canonical_id
            ]

            for mixsort_id in stale_mappings:

                del self.track_to_canonical[
                    mixsort_id
                ]

        return results

    # =========================================================
    # STATISTICS
    # =========================================================

    def statistics(self):

        return {
            "active_players": len(
                self.active_ids
            ),

            "lost_players": len(
                self.lost_ids
            ),

            "canonical_players_created": (
                self.next_canonical_id - 1
            ),

            "total_stitches": (
                self.total_stitches
            ),

            "total_new_players": (
                self.total_new_players
            ),
        }

    # =========================================================
    # RESET
    # =========================================================

    def reset(self):

        self.states.clear()
        self.track_to_canonical.clear()

        self.active_ids.clear()
        self.lost_ids.clear()

        self.next_canonical_id = 1

        self.total_stitches = 0
        self.total_new_players = 0