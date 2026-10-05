from pathlib import Path
from typing import Optional

from .mixsort_tracker import AthletiQMixSort
from .track_stitcher import TrackStitcher


class AthletiQTracker:
    """
    Single production entry point for AthletiQ player tracking.

    Pipeline:
        detections
            -> MixSort
            -> motion + OSNet Re-ID identity stitching
            -> Hungarian assignment
            -> canonical player IDs

    Input detections:
        numpy array of shape (N, 5):
        [x1, y1, x2, y2, confidence]

    Input frame:
        BGR OpenCV image.

    Output:
        list of dictionaries containing canonical_id, MixSort ID,
        bbox, center, confidence and stitching status.
    """

    def __init__(
        self,
        project_root,
        frame_rate: int = 25,
        mixsort_config: Optional[str] = None,
        reid_checkpoint: Optional[str] = None,
        reid_threshold: float = 0.70,
        max_distance: float = 100.0,
        max_frame_gap: int = 30,
        velocity_weight: float = 0.30,
        min_stitch_confidence: float = 0.70,
        max_prediction_speed: float = 30.0,
    ):
        self.project_root = Path(project_root)
        self.frame_rate = frame_rate

        # ------------------------------------------------------------
        # Final MixSort tracker
        # ------------------------------------------------------------
        self.mixsort = AthletiQMixSort(
            project_root=self.project_root,
            frame_rate=frame_rate,
            config_path=mixsort_config,
        )

        # ------------------------------------------------------------
        # Final production Re-ID checkpoint
        # ------------------------------------------------------------
        if reid_checkpoint is None:
            reid_checkpoint = (
                self.project_root
                / "ai-engine"
                / "models"
                / "reid"
                / "osnet_x1_0_msmt17_combineall.pth"
            )

        self.reid_checkpoint = Path(reid_checkpoint)

        # ------------------------------------------------------------
        # Identity stitching
        #
        # TrackStitcher performs:
        #   motion prediction
        #   OSNet Re-ID
        #   validated spatial gating
        #   Hungarian one-to-one assignment
        # ------------------------------------------------------------
        self.stitcher = TrackStitcher(
            max_distance=max_distance,
            max_frame_gap=max_frame_gap,
            velocity_weight=velocity_weight,
            min_stitch_confidence=min_stitch_confidence,
            max_prediction_speed=max_prediction_speed,
            reid_checkpoint=str(self.reid_checkpoint),
            reid_threshold=reid_threshold,
        )

        self.frame_id = 0

    def update(self, detections, frame):
        """
        Process one video frame.

        Parameters
        ----------
        detections : np.ndarray
            Shape (N, 5):
            [x1, y1, x2, y2, confidence]

        frame : np.ndarray
            BGR OpenCV frame.

        Returns
        -------
        list[dict]
            Canonical player tracks.
        """

        mixsort_tracks = self.mixsort.update(
            detections=detections,
            frame=frame,
        )

        tracks = self.stitcher.update(
            mixsort_tracks=mixsort_tracks,
            frame_id=self.frame_id,
            frame=frame,
        )

        self.frame_id += 1

        return tracks

    def reset(self):
        """Reset MixSort, stitching state and frame counter."""
        self.mixsort.reset()
        self.stitcher.reset()
        self.frame_id = 0

    def statistics(self):
        """Return current identity-stitching statistics."""
        return self.stitcher.statistics()

    def get_config(self):
        """Return the frozen production tracking configuration."""
        return {
            "tracker": "MixSort + OSNet Re-ID + Hungarian Stitching",
            "frame_rate": self.frame_rate,
            "mixsort": self.mixsort.get_config(),
            "stitching": {
                "reid_threshold": self.stitcher.reid_threshold,
                "max_distance": self.stitcher.max_distance,
                "max_frame_gap": self.stitcher.max_frame_gap,
                "velocity_weight": self.stitcher.velocity_weight,
                "min_stitch_confidence": (
                    self.stitcher.min_stitch_confidence
                ),
                "max_prediction_speed": (
                    self.stitcher.max_prediction_speed
                ),
                "reid_checkpoint": str(self.reid_checkpoint),
            },
        }

    def cleanup(self):
        """Release MixSort and Re-ID resources."""
        self.mixsort.cleanup()

        reid_model = getattr(self.stitcher, "reid_model", None)

        if reid_model is not None:
            model = getattr(reid_model, "model", None)

            if model is not None:
                try:
                    del model
                except Exception:
                    pass

        self.stitcher = None
        self.mixsort = None

        try:
            import gc
            import torch

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
