from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.spatial.distance import euclidean

from backend.models.schemas import PoseRecord, TMSRecord
from backend.services.faiss_store import FAISSStore


class TemporalMotionSignatureService:
    def __init__(self, store: FAISSStore, window_size: int = 60) -> None:
        self.store = store
        self.window_size = window_size
        self.pose_windows: dict[int, list[PoseRecord]] = defaultdict(list)

    def update(self, pose_record: PoseRecord) -> TMSRecord | None:
        window = self.pose_windows[pose_record.track_id]
        window.append(pose_record)
        if len(window) < self.window_size:
            return None
        window[:] = window[-self.window_size :]
        vector = self._compute_vector(window)
        matches = self.store.search("tms_embeddings", vector, top_k=1)
        score = matches[0]["score"] if matches else 0.0
        self.store.add("tms_embeddings", vector.reshape(1, -1), [{"track_id": pose_record.track_id}])
        return TMSRecord(
            track_id=pose_record.track_id,
            tms_vector=vector.tolist(),
            similarity_score=float(score),
        )

    def _compute_vector(self, window: list[PoseRecord]) -> np.ndarray:
        keypoints = np.array([record.keypoints for record in window], dtype=np.float32)
        left_ankle = keypoints[:, 27, :2]
        right_ankle = keypoints[:, 28, :2]
        left_wrist = keypoints[:, 15, :2]
        right_wrist = keypoints[:, 16, :2]
        nose = keypoints[:, 0, :2]
        shoulders = keypoints[:, [11, 12], :2].mean(axis=1)
        hips = keypoints[:, [23, 24], :2].mean(axis=1)
        stride = np.linalg.norm(left_ankle - right_ankle, axis=1)
        arm_swing = np.linalg.norm(left_wrist - right_wrist, axis=1)
        velocity = np.diff(hips, axis=0, prepend=hips[:1])
        posture = np.linalg.norm(shoulders - hips, axis=1)
        vector = np.array(
            [
                stride.mean(),
                stride.std(),
                self._periodicity(stride),
                self._cadence(stride),
                arm_swing.mean(),
                arm_swing.std(),
                self._cadence(arm_swing),
                self._periodicity(arm_swing),
                self._cadence(nose[:, 1]),
                np.std(nose[:, 1]),
                np.mean(np.linalg.norm(velocity, axis=1)),
                np.std(np.linalg.norm(velocity, axis=1)),
                posture.mean(),
                np.mean(np.abs(np.diff(posture, prepend=posture[:1]))),
                velocity[:, 0].mean(),
                velocity[:, 1].mean(),
            ],
            dtype=np.float32,
        )
        return vector

    @staticmethod
    def _periodicity(signal: np.ndarray) -> float:
        centered = signal - signal.mean()
        spectrum = np.fft.rfft(centered)
        magnitude = np.abs(spectrum)
        return float(magnitude[1:].max()) if magnitude.size > 1 else 0.0

    @staticmethod
    def _cadence(signal: np.ndarray) -> float:
        diffs = np.abs(np.diff(signal))
        return float(diffs.mean()) if diffs.size else 0.0

    def dtw_distance(self, vector_a: np.ndarray, vector_b: np.ndarray) -> float:
        n, m = len(vector_a), len(vector_b)
        dtw = np.full((n + 1, m + 1), np.inf)
        dtw[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = euclidean([vector_a[i - 1]], [vector_b[j - 1]])
                dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
        return float(dtw[n, m])
