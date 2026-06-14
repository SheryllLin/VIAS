from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency fallback
    YOLO = None

from backend.config import get_settings, resolve_path
from backend.models.schemas import DetectionRecord, TrackRecord
from backend.utils.device import get_preferred_device


@dataclass
class TrackerState:
    track_id: int
    bbox: np.ndarray
    score: float
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    active: bool = True

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.bbox[0] + self.bbox[2]) / 2.0, (self.bbox[1] + self.bbox[3]) / 2.0], dtype=np.float32)


class ByteTrackService:
    """
    Lightweight ByteTrack-inspired tracker.

    This keeps the project local and dependency-light while following the
    core ByteTrack idea of associating high-confidence detections first and
    then rescuing lost tracks with lower-confidence detections.
    """

    def __init__(self) -> None:
        settings = get_settings()
        tracker_settings = settings.tracker
        self.high_threshold = tracker_settings.high_confidence_threshold
        self.low_threshold = tracker_settings.low_confidence_threshold
        self.new_track_threshold = tracker_settings.new_track_threshold
        self.match_iou_threshold = tracker_settings.match_iou_threshold
        self.max_time_lost = tracker_settings.max_time_lost
        self.device = get_preferred_device()
        self.weights_path = resolve_path(settings.models.yolov10_weights)
        self.native_model = YOLO(str(self.weights_path)) if YOLO is not None and self.weights_path.exists() else None
        self.next_track_id = 1
        self.tracked: dict[int, TrackerState] = {}
        self.lost: dict[int, TrackerState] = {}

    @property
    def native_tracking_enabled(self) -> bool:
        return self.native_model is not None

    def track(self, frame, frame_id: int) -> tuple[list[DetectionRecord], list[TrackRecord], float]:
        if self.native_model is None:
            raise RuntimeError("Native ByteTrack integration is not available.")
        import time

        start = time.perf_counter()
        results = self.native_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=self.low_threshold,
            verbose=False,
            device=self.device,
        )
        detections: list[DetectionRecord] = []
        tracks: list[TrackRecord] = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
            for idx, box in enumerate(boxes):
                bbox = [float(x) for x in box.xyxy[0].tolist()]
                confidence = float(box.conf[0]) if box.conf is not None else 0.0
                detections.append(
                    DetectionRecord(
                        frame_id=frame_id,
                        person_bbox=bbox,
                        confidence=confidence,
                        detector="yolov10-bytetrack",
                    )
                )
                track_id = ids[idx] if ids[idx] is not None else self.next_track_id + idx
                tracks.append(
                    TrackRecord(
                        track_id=int(track_id),
                        frame_id=frame_id,
                        bbox=bbox,
                        confidence=confidence,
                        detection_confidence=confidence,
                        track_age=0,
                        lost_frames=0,
                        source="bytetrack-native",
                        velocity=[0.0, 0.0],
                    )
                )
        elapsed = max(time.perf_counter() - start, 1e-6)
        return detections, tracks, 1.0 / elapsed

    def update(self, detections: list[DetectionRecord], frame_id: int) -> list[TrackRecord]:
        high_detections = [det for det in detections if det.confidence >= self.high_threshold]
        low_detections = [det for det in detections if self.low_threshold <= det.confidence < self.high_threshold]

        self._predict_tracks()

        matches, unmatched_tracks, unmatched_detections = self._associate(self.tracked, high_detections)
        output = self._apply_matches(matches, high_detections, frame_id, source="high-confidence")

        rescued_matches, lost_unmatched, _ = self._associate_subset(unmatched_tracks, low_detections)
        output.extend(self._apply_matches(rescued_matches, low_detections, frame_id, source="low-confidence"))

        self._mark_unmatched_tracks(lost_unmatched)
        self._create_new_tracks([high_detections[idx] for idx in unmatched_detections], frame_id, output)
        self._prune_lost_tracks()
        return output

    def _predict_tracks(self) -> None:
        for pool in (self.tracked, self.lost):
            for state in pool.values():
                state.bbox = self._advance_bbox(state.bbox, state.velocity)
                state.age += 1
                state.time_since_update += 1

    def _associate(
        self, track_pool: dict[int, TrackerState], detections: list[DetectionRecord]
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        track_ids = list(track_pool.keys())
        if not track_ids or not detections:
            return [], track_ids, list(range(len(detections)))
        cost_matrix = self._build_cost_matrix([track_pool[track_id] for track_id in track_ids], detections)
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        matches: list[tuple[int, int]] = []
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for row, col in zip(row_idx, col_idx, strict=False):
            if 1.0 - cost_matrix[row, col] < self.match_iou_threshold:
                continue
            matches.append((track_ids[row], col))
            used_tracks.add(track_ids[row])
            used_detections.add(col)
        unmatched_tracks = [track_id for track_id in track_ids if track_id not in used_tracks]
        unmatched_detections = [idx for idx in range(len(detections)) if idx not in used_detections]
        return matches, unmatched_tracks, unmatched_detections

    def _associate_subset(
        self, track_ids: list[int], detections: list[DetectionRecord]
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        subset = {track_id: self.tracked[track_id] for track_id in track_ids if track_id in self.tracked}
        return self._associate(subset, detections)

    def _apply_matches(
        self, matches: list[tuple[int, int]], detections: list[DetectionRecord], frame_id: int, source: str
    ) -> list[TrackRecord]:
        records: list[TrackRecord] = []
        for track_id, det_idx in matches:
            detection = detections[det_idx]
            new_bbox = np.array(detection.person_bbox, dtype=np.float32)
            state = self.tracked.get(track_id) or self.lost.pop(track_id)
            previous_center = state.center
            updated_center = np.array([(new_bbox[0] + new_bbox[2]) / 2.0, (new_bbox[1] + new_bbox[3]) / 2.0], dtype=np.float32)
            state.velocity = updated_center - previous_center
            state.bbox = new_bbox
            state.score = detection.confidence
            state.hits += 1
            state.time_since_update = 0
            state.active = True
            self.tracked[track_id] = state
            records.append(
                TrackRecord(
                    track_id=track_id,
                    frame_id=frame_id,
                    bbox=detection.person_bbox,
                    confidence=self._track_confidence(state),
                    detection_confidence=detection.confidence,
                    track_age=state.age,
                    lost_frames=state.time_since_update,
                    source=source,
                    velocity=[float(state.velocity[0]), float(state.velocity[1])],
                )
            )
        return records

    def _mark_unmatched_tracks(self, unmatched_track_ids: list[int]) -> None:
        for track_id in unmatched_track_ids:
            if track_id not in self.tracked:
                continue
            state = self.tracked.pop(track_id)
            state.active = False
            self.lost[track_id] = state

    def _create_new_tracks(self, detections: list[DetectionRecord], frame_id: int, output: list[TrackRecord]) -> None:
        for detection in detections:
            if detection.confidence < self.new_track_threshold:
                continue
            bbox = np.array(detection.person_bbox, dtype=np.float32)
            track_id = self.next_track_id
            self.next_track_id += 1
            state = TrackerState(track_id=track_id, bbox=bbox, score=detection.confidence)
            self.tracked[track_id] = state
            output.append(
                TrackRecord(
                    track_id=track_id,
                    frame_id=frame_id,
                    bbox=detection.person_bbox,
                    confidence=self._track_confidence(state),
                    detection_confidence=detection.confidence,
                    track_age=state.age,
                    lost_frames=state.time_since_update,
                    source="new-track",
                    velocity=[0.0, 0.0],
                )
            )

    def _prune_lost_tracks(self) -> None:
        expired = [track_id for track_id, state in self.lost.items() if state.time_since_update > self.max_time_lost]
        for track_id in expired:
            del self.lost[track_id]

    def _build_cost_matrix(self, states: list[TrackerState], detections: list[DetectionRecord]) -> np.ndarray:
        matrix = np.ones((len(states), len(detections)), dtype=np.float32)
        for i, state in enumerate(states):
            predicted_bbox = self._advance_bbox(state.bbox, state.velocity)
            for j, detection in enumerate(detections):
                det_box = np.array(detection.person_bbox, dtype=np.float32)
                iou = self._iou(predicted_bbox, det_box)
                center_distance = np.linalg.norm(self._center(predicted_bbox) - self._center(det_box))
                normalized_distance = center_distance / max(self._bbox_diagonal(predicted_bbox), 1.0)
                matrix[i, j] = 1.0 - (0.8 * iou + 0.2 * max(0.0, 1.0 - normalized_distance))
        return matrix

    @staticmethod
    def _advance_bbox(bbox: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        return np.array(
            [bbox[0] + velocity[0], bbox[1] + velocity[1], bbox[2] + velocity[0], bbox[3] + velocity[1]],
            dtype=np.float32,
        )

    @staticmethod
    def _center(bbox: np.ndarray) -> np.ndarray:
        return np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0], dtype=np.float32)

    @staticmethod
    def _bbox_diagonal(bbox: np.ndarray) -> float:
        return float(np.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))

    @staticmethod
    def _track_confidence(state: TrackerState) -> float:
        stability = min(1.0, state.hits / max(state.age, 1))
        return float(0.7 * state.score + 0.3 * stability)

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union = area_a + area_b - inter + 1e-6
        return float(inter / union)
