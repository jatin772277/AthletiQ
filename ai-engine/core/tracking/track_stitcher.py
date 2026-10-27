from typing import Dict, List, Tuple
from pathlib import Path
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from .track_state import TrackState
from .reid_model import OSNetReID


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
        reid_checkpoint=None,
        reid_threshold: float = 0.70,
    ):

        self.max_distance = max_distance
        self.max_frame_gap = max_frame_gap
        self.velocity_weight = velocity_weight
        self.min_stitch_confidence = min_stitch_confidence

        # Maximum velocity allowed for trajectory prediction.
        self.max_prediction_speed = max_prediction_speed

        # -----------------------------------------------------
        # Production Re-ID
        # -----------------------------------------------------

        self.reid_threshold = reid_threshold

        self.reid_model = None

        if reid_checkpoint is not None:
            self.reid_model = OSNetReID(
                checkpoint_path=reid_checkpoint,
                device="cuda",
            )

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
    # PRODUCTION RE-ID MATCHING
    # =========================================================

    def _extract_reid_embedding(
        self,
        frame,
        bbox,
    ):
        """
        Extract a normalized OSNet embedding from one player.
        """

        if self.reid_model is None:
            return None

        return self.reid_model.extract_from_bbox(
            frame,
            bbox,
        )

    def _reid_similarity(
        self,
        embedding_a,
        embedding_b,
    ):
        """
        Compute cosine similarity between two Re-ID embeddings.
        """

        if (
            embedding_a is None
            or embedding_b is None
        ):
            return None

        return self.reid_model.cosine_similarity(
            embedding_a,
            embedding_b,
        )

    def _build_reid_candidates(
        self,
        mixsort_tracks,
        frame,
        frame_id,
    ):
        """
        Build candidate matches between new MixSort IDs
        and LOST canonical identities.

        Candidate generation uses the validated spatial rule:

            prediction_distance <= max_distance

        Re-ID is then used as the appearance gate.
        """

        candidates = []

        # -----------------------------------------------------
        # Only tracks that do not already have a mapping
        # are considered new identities.
        # -----------------------------------------------------

        new_tracks = [
            track
            for track in mixsort_tracks
            if int(track["track_id"])
            not in self.track_to_canonical
        ]

        for track in new_tracks:

            mixsort_id = int(track["track_id"])

            bbox = [
                float(v)
                for v in track["bbox"]
            ]

            position = self._center_from_bbox(
                bbox
            )

            # -------------------------------------------------
            # Extract NEW track appearance once.
            # -------------------------------------------------

            new_embedding = self._extract_reid_embedding(
                frame,
                bbox,
            )

            if new_embedding is None:
                continue

            # -------------------------------------------------
            # Compare against LOST identities only.
            # -------------------------------------------------

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

                # ---------------------------------------------
                # Motion prediction
                # ---------------------------------------------

                predicted = self._predict_position(
                    state,
                    frame_id,
                )

                prediction_distance = self._distance(
                    predicted,
                    position,
                )

                # ---------------------------------------------
                # Hard spatial gate
                # ---------------------------------------------

                if (
                    prediction_distance
                    > self.max_distance
                ):
                    continue

                # ---------------------------------------------
                # Re-ID gate
                # ---------------------------------------------

                if state.reid_embedding is None:
                    continue

                reid_score = self._reid_similarity(
                    state.reid_embedding,
                    new_embedding,
                )

                if reid_score is None:
                    continue

                if (
                    reid_score
                    < self.reid_threshold
                ):
                    continue

                # ---------------------------------------------
                # Motion confidence
                # ---------------------------------------------

                position_confidence = max(
                    0.0,
                    1.0
                    - (
                        prediction_distance
                        / self.max_distance
                    ),
                )

                velocity_confidence = (
                    self._velocity_similarity(
                        state,
                        position,
                        gap,
                    )
                )

                time_confidence = max(
                    0.0,
                    1.0
                    - (
                        (gap - 1)
                        / self.max_frame_gap
                    ),
                )

                # ---------------------------------------------
                # Combined score
                #
                # Re-ID is the strongest identity signal,
                # while motion remains a hard constraint.
                # ---------------------------------------------

                motion_score = (
                    0.60
                    * position_confidence
                    +
                    self.velocity_weight
                    * velocity_confidence
                    +
                    0.10
                    * time_confidence
                )

                combined_score = (
                    0.70 * reid_score
                    +
                    0.30 * motion_score
                )

                candidates.append({
                    "mixsort_id": mixsort_id,
                    "canonical_id": canonical_id,
                    "bbox": bbox,
                    "position": position,
                    "embedding": new_embedding,
                    "gap": gap,
                    "prediction_distance": prediction_distance,
                    "reid_score": reid_score,
                    "motion_score": motion_score,
                    "combined_score": combined_score,
                })

        return candidates

    def _hungarian_match(
        self,
        candidates,
    ):
        """
        Perform one-to-one global assignment.

        Returns:
            list of accepted candidate dictionaries.
        """

        if not candidates:
            return []

        new_ids = sorted({
            candidate["mixsort_id"]
            for candidate in candidates
        })

        canonical_ids = sorted({
            candidate["canonical_id"]
            for candidate in candidates
        })

        new_index = {
            track_id: index
            for index, track_id
            in enumerate(new_ids)
        }

        canonical_index = {
            canonical_id: index
            for index, canonical_id
            in enumerate(canonical_ids)
        }

        # Cost matrix.
        cost_matrix = np.full(
            (
                len(new_ids),
                len(canonical_ids),
            ),
            1e6,
            dtype=np.float32,
        )

        candidate_lookup = {}

        for candidate in candidates:

            i = new_index[
                candidate["mixsort_id"]
            ]

            j = canonical_index[
                candidate["canonical_id"]
            ]

            # Hungarian minimizes cost.
            cost = (
                1.0
                - candidate["combined_score"]
            )

            cost_matrix[i, j] = cost

            candidate_lookup[
                (
                    candidate["mixsort_id"],
                    candidate["canonical_id"],
                )
            ] = candidate

        rows, cols = linear_sum_assignment(
            cost_matrix
        )

        assignments = []

        for row, col in zip(rows, cols):

            if cost_matrix[row, col] >= 1e6:
                continue

            mixsort_id = new_ids[row]
            canonical_id = canonical_ids[col]

            candidate = candidate_lookup.get(
                (
                    mixsort_id,
                    canonical_id,
                )
            )

            if candidate is not None:
                assignments.append(candidate)

        return assignments

    # =========================================================
    # UPDATE
    # =========================================================

    def update(
        self,
        mixsort_tracks: List[dict],
        frame_id: int,
        frame=None,
    ):
        """
        Update canonical identities using MixSort + motion + Re-ID.

        Parameters
        ----------
        mixsort_tracks : list of dict
            MixSort outputs.

        frame_id : int
            Current frame number.

        frame : np.ndarray, optional
            Current BGR video frame.

        Pipeline
        --------
        1. Move disappeared ACTIVE identities to LOST.
        2. Update existing MixSort -> canonical mappings.
        3. Generate candidates for all NEW MixSort IDs.
        4. Perform global Hungarian assignment.
        5. Apply accepted stitches.
        6. Create canonical identities for unmatched new tracks.
        7. Remove stale LOST identities.
        """

        if mixsort_tracks is None:
            mixsort_tracks = []

        # =====================================================
        # STEP 1
        # Move disappeared ACTIVE identities to LOST.
        # =====================================================

        current_mixsort_ids = {
            int(track["track_id"])
            for track in mixsort_tracks
        }

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

        # =====================================================
        # STEP 2
        # Separate existing and NEW MixSort tracks.
        # =====================================================

        existing_tracks = []
        new_tracks = []

        for track in mixsort_tracks:

            mixsort_id = int(
                track["track_id"]
            )

            if mixsort_id in self.track_to_canonical:
                existing_tracks.append(track)
            else:
                new_tracks.append(track)

        # =====================================================
        # STEP 3
        # Update existing identities.
        #
        # We deliberately keep their previous Re-ID embedding
        # until a new identity transition is confirmed.
        # =====================================================

        results = []

        for track in existing_tracks:

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

            results.append({
                "mixsort_id": mixsort_id,
                "canonical_id": canonical_id,
                "bbox": bbox,
                "center": position,
                "score": score,
                "stitched": False,
                "identity_confidence": 1.0,
            })

        # =====================================================
        # STEP 4
        # Build Re-ID embeddings for NEW MixSort tracks.
        #
        # This is done once per new MixSort ID.
        # =====================================================

        new_track_embeddings = {}

        if (
            frame is not None
            and self.reid_model is not None
        ):

            for track in new_tracks:

                mixsort_id = int(
                    track["track_id"]
                )

                bbox = [
                    float(v)
                    for v in track["bbox"]
                ]

                embedding = (
                    self._extract_reid_embedding(
                        frame,
                        bbox,
                    )
                )

                if embedding is not None:
                    new_track_embeddings[
                        mixsort_id
                    ] = embedding

        # =====================================================
        # STEP 5
        # Generate candidates for ALL new tracks.
        #
        # We temporarily expose the embeddings through the
        # track dictionaries so the candidate engine can use
        # the same frame-level candidate pool.
        # =====================================================

        candidate_tracks = []

        for track in new_tracks:

            mixsort_id = int(
                track["track_id"]
            )

            if mixsort_id not in new_track_embeddings:
                continue

            candidate_track = dict(track)

            candidate_track["_reid_embedding"] = (
                new_track_embeddings[
                    mixsort_id
                ]
            )

            candidate_tracks.append(
                candidate_track
            )

        # =====================================================
        # STEP 6
        # Build candidates manually using the precomputed
        # embeddings.
        #
        # This avoids running OSNet repeatedly for every
        # LOST identity.
        # =====================================================

        candidates = []

        for track in candidate_tracks:

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

            new_embedding = (
                track["_reid_embedding"]
            )

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

                if state.reid_embedding is None:
                    continue

                predicted = self._predict_position(
                    state,
                    frame_id,
                )

                prediction_distance = self._distance(
                    predicted,
                    position,
                )

                # Validated hard spatial gate.
                if (
                    prediction_distance
                    > self.max_distance
                ):
                    continue

                reid_score = self._reid_similarity(
                    state.reid_embedding,
                    new_embedding,
                )

                if reid_score is None:
                    continue

                # Validated Re-ID gate.
                if (
                    reid_score
                    < self.reid_threshold
                ):
                    continue

                position_confidence = max(
                    0.0,
                    1.0
                    - (
                        prediction_distance
                        / self.max_distance
                    ),
                )

                velocity_confidence = (
                    self._velocity_similarity(
                        state,
                        position,
                        gap,
                    )
                )

                time_confidence = max(
                    0.0,
                    1.0
                    - (
                        (gap - 1)
                        / self.max_frame_gap
                    ),
                )

                motion_score = (
                    0.60
                    * position_confidence
                    +
                    self.velocity_weight
                    * velocity_confidence
                    +
                    0.10
                    * time_confidence
                )

                combined_score = (
                    0.70 * reid_score
                    +
                    0.30 * motion_score
                )

                candidates.append({
                    "mixsort_id": mixsort_id,
                    "canonical_id": canonical_id,
                    "bbox": bbox,
                    "position": position,
                    "embedding": new_embedding,
                    "gap": gap,
                    "prediction_distance": prediction_distance,
                    "reid_score": reid_score,
                    "motion_score": motion_score,
                    "combined_score": combined_score,
                })

        # =====================================================
        # STEP 7
        # Global one-to-one Hungarian assignment.
        # =====================================================

        assignments = self._hungarian_match(
            candidates
        )

        assignment_by_mixsort = {
            candidate["mixsort_id"]: candidate
            for candidate in assignments
        }

        # =====================================================
        # STEP 8
        # Process every NEW MixSort track.
        # =====================================================

        for track in new_tracks:

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

            embedding = new_track_embeddings.get(
                mixsort_id
            )

            assignment = assignment_by_mixsort.get(
                mixsort_id
            )

            stitched = False

            # -------------------------------------------------
            # Successful global stitch
            # -------------------------------------------------

            if assignment is not None:

                canonical_id = (
                    assignment["canonical_id"]
                )

                confidence = float(
                    assignment["combined_score"]
                )

                state = self.states[
                    canonical_id
                ]

                state.update(
                    position=position,
                    bbox=bbox,
                    frame_id=frame_id,
                    track_id=mixsort_id,
                    identity_confidence=confidence,
                    reid_embedding=embedding,
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
                identity_confidence = confidence

            # -------------------------------------------------
            # No valid stitch → create NEW canonical identity
            # -------------------------------------------------

            else:

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
                    reid_embedding=embedding,
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

                identity_confidence = 1.0

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
        # STEP 9
        # Remove identities that have been LOST too long.
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

            if canonical_id in self.states:
                del self.states[
                    canonical_id
                ]

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