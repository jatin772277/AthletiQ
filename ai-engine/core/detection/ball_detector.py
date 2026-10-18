
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class BallDetection:
    """A single ball detection in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def center(self) -> tuple[float, float]:
        """Return the center of the bounding box."""
        return (
            (self.x1 + self.x2) / 2.0,
            (self.y1 + self.y2) / 2.0,
        )

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Return bounding box as (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)


class AthletiQBallDetector:
    """
    AthletiQ's YOLO-based football detector.

    This class detects the ball in individual frames.
    Tracking the ball across frames is handled separately.
    """

    def __init__(
        self,
        model_path: str | Path,
        confidence: float = 0.05,
        iou_threshold: float = 0.7,
        device: Optional[str] = None,
    ) -> None:
        self.model_path = Path(model_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Ball detector model not found: {self.model_path}"
            )

        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")

        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.device = device

        self.model = YOLO(str(self.model_path))

    def detect(self, frame: np.ndarray) -> list[BallDetection]:
        """
        Detect candidate footballs in one BGR image.

        Args:
            frame: Image as a NumPy array, usually from OpenCV.

        Returns:
            Ball detections sorted by confidence, highest first.
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a NumPy array")

        if frame.size == 0:
            raise ValueError("frame cannot be empty")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must have shape (height, width, 3)")

        prediction_args = {
            "source": frame,
            "imgsz": 1280,
            "conf": self.confidence,
            "iou": self.iou_threshold,
            "max_det": 30,
            "verbose": False,
        }

        if self.device is not None:
            prediction_args["device"] = self.device

        results = self.model.predict(**prediction_args)
        detections: list[BallDetection] = []

        if not results:
            return detections

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        coordinates = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()

        for coords, score in zip(coordinates, confidences):
            x1, y1, x2, y2 = map(float, coords)

            detections.append(
                BallDetection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=float(score),
                )
            )

        detections.sort(
            key=lambda detection: detection.confidence,
            reverse=True,
        )

        return detections
