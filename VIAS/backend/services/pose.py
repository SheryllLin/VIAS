from __future__ import annotations

import logging

try:
    import mediapipe as mp  # type: ignore
except ImportError:  # pragma: no cover - optional dependency fallback
    mp = None

from backend.models.schemas import PoseRecord

logger = logging.getLogger(__name__)


class MediaPipePoseService:
    """
    MediaPipe Pose Estimation Service
    
    Extracts 33 body keypoints (landmarks) from video frames.
    Each landmark includes (x, y, z, visibility) where:
    - x, y: Normalized coordinates [0-1]
    - z: Depth relative to hips
    - visibility: Confidence [0-1] that landmark is visible
    """
    
    # MediaPipe Pose landmark indices
    LANDMARK_COUNT = 33
    
    def __init__(self) -> None:
        """Initialize MediaPipe Pose estimator."""
        self.pose = None
        self._extraction_count = 0
        self._failure_count = 0
        
        if mp is None:
            logger.warning("MediaPipe module not installed. Pose estimation unavailable.")
            return
            
        try:
            # Configure MediaPipe Pose
            # static_image_mode=False: Optimized for video (tracks across frames)
            # model_complexity=1: Balance of accuracy and speed (0=lite, 1=full, 2=heavy)
            # smooth_landmarks=True: Temporal smoothing for video
            # min_detection_confidence=0.5: Minimum confidence for initial detection
            # min_tracking_confidence=0.5: Minimum confidence for tracking
            self.pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("MediaPipe Pose initialized successfully (model_complexity=1)")
            
        except AttributeError as exc:
            logger.error(f"MediaPipe Pose initialization failed - AttributeError: {exc}")
            self.pose = None
        except Exception as exc:
            logger.error(f"MediaPipe Pose initialization failed - Unexpected error: {exc}")
            self.pose = None
    
    @property
    def available(self) -> bool:
        """Check if MediaPipe Pose is available and initialized."""
        return self.pose is not None

    def extract(self, frame, track_id: int, frame_id: int) -> PoseRecord | None:
        """
        Extract pose landmarks from a video frame.
        
        Args:
            frame: BGR image (OpenCV format)
            track_id: ID of the tracked person
            frame_id: Current frame number
            
        Returns:
            PoseRecord with 33 keypoints, or None if extraction fails
        """
        if self.pose is None:
            return None
        
        try:
            # Convert BGR to RGB (MediaPipe expects RGB)
            rgb_frame = frame[:, :, ::-1]
            
            # Run pose estimation
            result = self.pose.process(rgb_frame)
            
            # Check if landmarks were detected
            if not result.pose_landmarks:
                self._failure_count += 1
                if self._failure_count % 100 == 0:  # Log every 100 failures
                    logger.debug(f"No pose landmarks detected (track={track_id}, frame={frame_id}). Total failures: {self._failure_count}")
                return None
            
            # Extract landmarks
            landmarks = result.pose_landmarks.landmark
            
            # Validate landmark count
            if len(landmarks) != self.LANDMARK_COUNT:
                logger.warning(
                    f"Unexpected landmark count: {len(landmarks)} (expected {self.LANDMARK_COUNT}) "
                    f"for track={track_id}, frame={frame_id}"
                )
                return None
            
            # Extract keypoints with visibility
            keypoints = [
                [landmark.x, landmark.y, landmark.z, landmark.visibility]
                for landmark in landmarks
            ]
            
            self._extraction_count += 1
            
            # Log success periodically
            if self._extraction_count == 1:
                logger.info(f"First successful pose extraction (track={track_id}, frame={frame_id})")
            elif self._extraction_count % 1000 == 0:
                logger.info(
                    f"Pose extraction stats: {self._extraction_count} successful, "
                    f"{self._failure_count} failures"
                )
            
            return PoseRecord(track_id=track_id, frame_id=frame_id, keypoints=keypoints)
            
        except AttributeError as exc:
            logger.error(f"MediaPipe attribute error during extraction: {exc}")
            return None
        except Exception as exc:
            logger.error(f"Unexpected error during pose extraction (track={track_id}, frame={frame_id}): {exc}")
            return None
    
    def get_stats(self) -> dict[str, int]:
        """Get extraction statistics."""
        return {
            "total_extractions": self._extraction_count,
            "total_failures": self._failure_count,
            "success_rate": (
                self._extraction_count / (self._extraction_count + self._failure_count)
                if (self._extraction_count + self._failure_count) > 0
                else 0.0
            )
        }
