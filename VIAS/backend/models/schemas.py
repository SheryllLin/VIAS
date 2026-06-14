from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DetectionRecord(BaseModel):
    frame_id: int
    person_bbox: list[float]
    confidence: float
    detector: str = "yolov10"


class TrackRecord(BaseModel):
    track_id: int
    frame_id: int
    bbox: list[float]
    confidence: float = 0.0
    detection_confidence: float = 0.0
    track_age: int = 0
    lost_frames: int = 0
    source: str = "tracked"
    velocity: list[float] = Field(default_factory=lambda: [0.0, 0.0])


class PoseRecord(BaseModel):
    track_id: int
    frame_id: int
    keypoints: list[list[float]]


class IdentityRecord(BaseModel):
    track_id: int
    person_id: str | None = None
    face_confidence: float = 0.0
    body_confidence: float = 0.0
    tms_confidence: float = 0.0
    match_source: str = "none"


class ActivityRecord(BaseModel):
    track_id: int
    activity: str
    confidence: float
    timestamp: float


class TMSRecord(BaseModel):
    track_id: int
    tms_vector: list[float] = Field(min_length=16, max_length=16)
    similarity_score: float


class EventRecord(BaseModel):
    event_type: str
    timestamp: float
    track_id: int
    confidence: float
    location: str = "default"
    identity: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    query: str


class SearchPersonRequest(BaseModel):
    person_id: str | None = None
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    sql: str | None = None
    results: list[dict[str, Any]] = Field(default_factory=list)


class VideoProcessingResponse(BaseModel):
    video_id: str
    output_video_path: str
    detections: int
    tracks: int
    activities: int
    events: int
    fps: float


class AnalyticsSummary(BaseModel):
    detection_count: int
    tracked_persons: int
    activity_distribution: dict[str, int]
    reid_statistics: dict[str, int]
    avg_fps: float
    timeline: list[dict[str, Any]] = Field(default_factory=list)
