
from pathlib import Path
from types import SimpleNamespace
import sys
import gc

import numpy as np
import torch
import yaml

# ----------------------------------------------------------------------
# PyTorch compatibility for the trusted MixSort checkpoint
#
# PyTorch 2.6+ defaults torch.load() to weights_only=True.
# MixSort's checkpoint contains legacy Python objects such as
# AverageMeter, so the official trusted checkpoint must be loaded
# with weights_only=False.
# ----------------------------------------------------------------------

_original_torch_load = torch.load


def _mixsort_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = _mixsort_torch_load


class AthletiQMixSort:
    """
    Reusable MixSort tracker for AthletiQ.

    Final configuration:
        track_thresh = 0.6
        track_buffer = 30
        match_thresh = 0.9
        iou_thresh = 0.3
        alpha = 0.8
    """

    def __init__(
        self,
        project_root,
        frame_rate=25,
        config_path=None,
    ):
        self.project_root = Path(project_root)
        self.frame_rate = frame_rate

        self.mixsort_dir = (
            self.project_root
            / "ai-engine"
            / "third_party"
            / "MixSort"
        )

        if config_path is None:
            config_path = (
                self.project_root
                / "ai-engine"
                / "models"
                / "mixsort_final.yaml"
            )

        self.config_path = Path(config_path)

        # ------------------------------------------------------------------
        # NumPy compatibility for older MixSort code
        # ------------------------------------------------------------------
        if not hasattr(np, "float"):
            np.float = float

        if not hasattr(np, "int"):
            np.int = int

        # ------------------------------------------------------------------
        # Add MixSort and MixViT to Python path
        # ------------------------------------------------------------------
        mixsort_path = str(self.mixsort_dir)
        mixvit_path = str(self.mixsort_dir / "MixViT")

        # MixSort imports the top-level `yolox` package.
        if mixsort_path not in sys.path:
            sys.path.insert(0, mixsort_path)

        # MixViT internally imports `lib.*`.
        # Therefore MixViT itself must be on sys.path.
        if mixvit_path not in sys.path:
            sys.path.insert(0, mixvit_path)

        # ------------------------------------------------------------------
        # Load final AthletiQ configuration
        # ------------------------------------------------------------------
        with open(self.config_path, "r") as f:
            cfg = yaml.safe_load(f)

        tracker_cfg = cfg["tracker"]

        self.track_thresh = tracker_cfg["track_thresh"]
        self.track_buffer = tracker_cfg["track_buffer"]
        self.match_thresh = tracker_cfg["match_thresh"]
        self.iou_thresh = tracker_cfg["iou_thresh"]
        self.alpha = tracker_cfg["alpha"]

        # ------------------------------------------------------------------
        # Import MixSort
        # ------------------------------------------------------------------
        from yolox.mixsort_tracker.mixsort_tracker import MIXTracker

        self.MIXTracker = MIXTracker

        # ------------------------------------------------------------------
        # MixSort arguments
        # ------------------------------------------------------------------
        args = SimpleNamespace(
            track_thresh=self.track_thresh,
            track_buffer=self.track_buffer,
            match_thresh=self.match_thresh,
            iou_thresh=self.iou_thresh,
            alpha=self.alpha,
            radius=0,
            script="mixformer_deit",
            config="track",
            local_rank=0,
            mot20=False,
        )

        self.args = args

        # ------------------------------------------------------------------
        # Create tracker
        # ------------------------------------------------------------------
        self.tracker = MIXTracker(
            self.args,
            frame_rate=self.frame_rate
        )

        self.frame_id = 0

    # ----------------------------------------------------------------------
    # Update tracker
    # ----------------------------------------------------------------------
    def update(self, detections, frame):
        """
        Update MixSort with detector outputs.

        Parameters
        ----------
        detections : np.ndarray
            Shape: (N, 5)
            [x1, y1, x2, y2, confidence]

        frame : np.ndarray
            BGR image.

        Returns
        -------
        list of dict
            [
                {
                    "track_id": int,
                    "bbox": [x1, y1, x2, y2],
                    "score": float
                },
                ...
            ]
        """

        if detections is None:
            detections = np.empty((0, 5), dtype=np.float32)

        detections = np.asarray(
            detections,
            dtype=np.float32
        ).copy()

        h, w = frame.shape[:2]

        img_info = (h, w)
        img_size = (h, w)

        online_targets = self.tracker.update(
            detections,
            img_info,
            img_size,
            frame
        )

        results = []

        for target in online_targets:

            tlwh = target.tlwh

            x, y, bw, bh = tlwh

            x1 = float(x)
            y1 = float(y)
            x2 = float(x + bw)
            y2 = float(y + bh)

            results.append({
                "track_id": int(target.track_id),
                "bbox": [x1, y1, x2, y2],
                "score": float(target.score),
            })

        self.frame_id += 1

        return results

    # ----------------------------------------------------------------------
    # Reset tracker
    # ----------------------------------------------------------------------
    def reset(self):
        """
        Create a fresh tracker.
        """

        self.tracker = self.MIXTracker(
            self.args,
            frame_rate=self.frame_rate
        )

        self.frame_id = 0

    # ----------------------------------------------------------------------
    # Cleanup
    # ----------------------------------------------------------------------
    def cleanup(self):
        """
        Release tracker and GPU memory.
        """

        self.tracker = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # ----------------------------------------------------------------------
    # Configuration
    # ----------------------------------------------------------------------
    def get_config(self):

        return {
            "track_thresh": self.track_thresh,
            "track_buffer": self.track_buffer,
            "match_thresh": self.match_thresh,
            "iou_thresh": self.iou_thresh,
            "alpha": self.alpha,
            "frame_rate": self.frame_rate,
        }
