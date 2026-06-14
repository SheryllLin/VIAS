from __future__ import annotations

import uuid
from pathlib import Path

import cv2

from backend.config import get_settings, resolve_path
from backend.database.repository import EventRepository
from backend.models.schemas import EventRecord, VideoProcessingResponse
from backend.services.activity import STGCNPlusPlusService
from backend.services.analytics import AnalyticsService
from backend.services.behavior import BehaviorDiscoveryService
from backend.services.detector import YOLOv10Detector
from backend.services.faiss_store import FAISSStore
from backend.services.pose import MediaPipePoseService
from backend.services.reid import MultiTierReIDService
from backend.services.tms import TemporalMotionSignatureService
from backend.services.tracker import ByteTrackService


class VideoAnalyticsPipeline:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.repository = EventRepository()
        self.store = FAISSStore()
        self.tracker = ByteTrackService()
        self.detector = None if self.tracker.native_tracking_enabled else YOLOv10Detector()
        self.pose = MediaPipePoseService()
        self.reid = MultiTierReIDService(self.store)
        self.tms = TemporalMotionSignatureService(self.store, self.settings.video.tms_window)
        self.activity = STGCNPlusPlusService()
        self.behavior = BehaviorDiscoveryService(self.store)
        self.analytics = AnalyticsService(self.repository)

    def process_video(self, video_path: Path) -> VideoProcessingResponse:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        video_id = str(uuid.uuid4())
        output_path = resolve_path(self.settings.paths.outputs_dir) / f"{video_id}.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        frame_id = 0
        total_fps = []
        detections_count = 0
        tracks_seen: set[int] = set()
        activities_count = 0
        events_count = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_id % self.settings.video.frame_stride != 0:
                frame_id += 1
                continue
            if self.tracker.native_tracking_enabled:
                detections, tracks, inference_fps = self.tracker.track(frame, frame_id)
            else:
                if self.detector is None:
                    raise RuntimeError("Detector is unavailable.")
                detections, inference_fps = self.detector.detect(frame, frame_id)
                tracks = self.tracker.update(detections, frame_id)
            total_fps.append(inference_fps)
            detections_count += len(detections)
            timestamp = frame_id / fps
            for track in tracks:
                tracks_seen.add(track.track_id)
                pose_record = self.pose.extract(frame, track.track_id, frame_id)
                identity = self.reid.identify(
                    track_id=track.track_id,
                    frame_id=frame_id,
                    pose_available=pose_record is not None,
                    frame=frame,
                    bbox=track.bbox,
                )
                self.repository.add_track(
                    track,
                    timestamp,
                    identity.person_id,
                    max(track.confidence, identity.face_confidence, identity.body_confidence, identity.tms_confidence),
                )
                tms_record = self.tms.update(pose_record) if pose_record else None
                if tms_record:
                    self.repository.add_tms_vector(tms_record)
                if pose_record:
                    activity = self.activity.classify(pose_record, timestamp)
                    self.repository.add_activity(activity)
                    activities_count += 1
                    behavior_summary = self.behavior.update(activity, tms_record)
                    event = EventRecord(
                        event_type="activity",
                        timestamp=timestamp,
                        track_id=track.track_id,
                        identity=identity.person_id,
                        confidence=activity.confidence,
                        metadata={"activity": activity.activity, "behavior_summary": behavior_summary["summary"]},
                    )
                    self.repository.add_event(event)
                    events_count += 1
                    self._draw_overlay(frame, track, identity.person_id, activity.activity, activity.confidence, timestamp, identity.match_source)
                else:
                    self._draw_overlay(frame, track, identity.person_id, "Unknown", 0.0, timestamp, identity.match_source)
            writer.write(frame)
            frame_id += 1

        cap.release()
        writer.release()
        avg_fps = sum(total_fps) / len(total_fps) if total_fps else 0.0
        return VideoProcessingResponse(
            video_id=video_id,
            output_video_path=str(output_path),
            detections=detections_count,
            tracks=len(tracks_seen),
            activities=activities_count,
            events=events_count,
            fps=avg_fps,
        )

    @staticmethod
    def _draw_overlay(
        frame,
        track,
        identity: str | None,
        activity: str,
        confidence: float,
        timestamp: float,
        match_source: str,
    ) -> None:
        x1, y1, x2, y2 = [int(v) for v in track.bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)
        label = f"ID {track.track_id} | {identity or 'unknown'} | {match_source} | {activity} | {confidence:.2f} | trk {track.confidence:.2f}"
        cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"{timestamp:.2f}s", (x1, min(frame.shape[0] - 10, y2 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
