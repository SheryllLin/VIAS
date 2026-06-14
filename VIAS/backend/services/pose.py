from __future__ import annotations

try:
    import mediapipe as mp  # type: ignore
except ImportError:  # pragma: no cover - optional dependency fallback
    mp = None

from backend.models.schemas import PoseRecord


class MediaPipePoseService:
    def __init__(self) -> None:
        try:
            self.pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1) if mp is not None else None
        except Exception:  # pragma: no cover - optional dependency fallback
            self.pose = None

    def extract(self, frame, track_id: int, frame_id: int) -> PoseRecord | None:
        if self.pose is None:
            return None
        result = self.pose.process(frame[:, :, ::-1])
        if not result.pose_landmarks:
            return None
        keypoints = [
            [landmark.x, landmark.y, landmark.z]
            for landmark in result.pose_landmarks.landmark
        ]
        return PoseRecord(track_id=track_id, frame_id=frame_id, keypoints=keypoints)
