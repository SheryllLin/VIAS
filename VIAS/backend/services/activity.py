from __future__ import annotations

import numpy as np

from backend.models.schemas import ActivityRecord, PoseRecord


class STGCNPlusPlusService:
    CLASSES = ["Walking", "Standing", "Sitting", "Waving", "Hand Raising"]

    def classify(self, pose_record: PoseRecord, timestamp: float) -> ActivityRecord:
        keypoints = np.array(pose_record.keypoints, dtype=np.float32)
        wrists = keypoints[[15, 16], 1]
        hips = keypoints[[23, 24], 1]
        ankles = keypoints[[27, 28], 1]
        if wrists.mean() < keypoints[0, 1]:
            label = "Hand Raising"
            confidence = 0.78
        elif np.abs(wrists[0] - wrists[1]) > 0.15:
            label = "Waving"
            confidence = 0.74
        elif np.abs(ankles[0] - ankles[1]) > 0.08:
            label = "Walking"
            confidence = 0.81
        elif hips.mean() > 0.7:
            label = "Sitting"
            confidence = 0.69
        else:
            label = "Standing"
            confidence = 0.76
        return ActivityRecord(
            track_id=pose_record.track_id,
            activity=label,
            confidence=confidence,
            timestamp=timestamp,
        )
