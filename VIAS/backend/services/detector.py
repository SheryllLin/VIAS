from __future__ import annotations

import time

import cv2

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency fallback
    YOLO = None

from backend.config import get_settings, resolve_path
from backend.models.schemas import DetectionRecord
from backend.utils.device import get_preferred_device
from backend.utils.logging import logger


class YOLOv10Detector:
    def __init__(self) -> None:
        settings = get_settings()
        self.conf_threshold = settings.thresholds.detector_confidence
        self.weights_path = resolve_path(settings.models.yolov10_weights)
        self.device = get_preferred_device()
        self.model = YOLO(str(self.weights_path)) if YOLO is not None and self.weights_path.exists() else None
        if self.model is None:
            logger.warning("YOLOv10 weights not found at %s. Falling back to HOG person detector.", self.weights_path)
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame, frame_id: int) -> tuple[list[DetectionRecord], float]:
        start = time.perf_counter()
        if self.model is not None:
            results = self.model.predict(frame, classes=[0], conf=self.conf_threshold, verbose=False, device=self.device)
            detections = [
                DetectionRecord(
                    frame_id=frame_id,
                    person_bbox=[float(x) for x in box.xyxy[0].tolist()],
                    confidence=float(box.conf[0]),
                    detector="yolov10",
                )
                for result in results
                for box in result.boxes
            ]
        else:
            rects, weights = self.hog.detectMultiScale(frame, winStride=(8, 8))
            detections = [
                DetectionRecord(
                    frame_id=frame_id,
                    person_bbox=[float(x), float(y), float(x + w), float(y + h)],
                    confidence=float(weight),
                    detector="hog",
                )
                for (x, y, w, h), weight in zip(rects, weights, strict=False)
            ]
        elapsed = max(time.perf_counter() - start, 1e-6)
        return detections, 1.0 / elapsed
