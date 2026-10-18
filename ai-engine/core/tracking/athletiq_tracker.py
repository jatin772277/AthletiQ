
from pathlib import Path
from typing import Optional

from .mixsort_tracker import AthletiQMixSort
from .track_stitcher import TrackStitcher


class AthletiQTracker:
    """
    Single entry point for AthletiQ player tracking.

    Pipeline:
        detections -> MixSort -> identity stitching -> canonical IDs

    Input detections:
        NumPy array with shape (N, 5):
        [x1, y1, x2, y2, confidence]

    Input frame:
        BGR OpenCV image.
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

        self.mixsort = AthletiQMixSort(
            project_root=self.project_root,
            frame_rate=frame_rate,
            config_path=mixsort_config,
        )

        # Re-ID is optional. Do not try to load a missing checkpoint.
        if reid_checkpoint is None:
            default_checkpoint = (
                self.project_root
                / "ai-engine"
                / "models"
                / "reid"
                / "osnet_x1_0_msmt17_combineall.pth"
            )
            if default_checkpoint.is_file():
                reid_checkpoint = str(default_checkpoint)

        if reid_checkpoint is not None:
            checkpoint_path = Path(reid_checkpoint).expanduser()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Specified Re-ID checkpoint does not exist: "
                    f"{checkpoint_path}"
                )
            self.reid_checkpoint = checkpoint_path
        else:
            self.reid_checkpoint = None

        self.stitcher = TrackStitcher(
            max_distance=max_distance,
            max_frame_gap=max_frame_gap,
            velocity_weight=velocity_weight,
            min_stitch_confidence=min_stitch_confidence,
            max_prediction_speed=max_prediction_speed,
            reid_checkpoint=(
                str(self.reid_checkpoint)
                if self.reid_checkpoint is not None
                else None
            ),
            reid_threshold=reid_threshold,
        )

        self.frame_id = 0

    def update(self, detections, frame):
        """Process one frame and return canonical player tracks."""
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
        """Reset MixSort, stitching state, and frame counter."""
        self.mixsort.reset()
        self.stitcher.reset()
        self.frame_id = 0

    def statistics(self):
        """Return identity-stitching statistics."""
        return self.stitcher.statistics()

    def get_config(self):
        """Return the active tracking configuration."""
        return {
            "tracker": "MixSort + optional OSNet Re-ID + Hungarian Stitching",
            "frame_rate": self.frame_rate,
            "mixsort": self.mixsort.get_config(),
            "stitching": {
                "reid_enabled": self.reid_checkpoint is not None,
                "reid_checkpoint": (
                    str(self.reid_checkpoint)
                    if self.reid_checkpoint is not None
                    else None
                ),
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
            },
        }

    def cleanup(self):
        """Release tracker resources and clear the CUDA cache."""
        if self.mixsort is not None:
            self.mixsort.cleanup()

        reid_model = (
            getattr(self.stitcher, "reid_model", None)
            if self.stitcher is not None
            else None
        )

        if reid_model is not None:
            model = getattr(reid_model, "model", None)
            if model is not None:
                del model

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
