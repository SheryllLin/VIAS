from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings


ROOT_DIR = Path(__file__).resolve().parents[1]


class PathSettings(BaseModel):
    uploads_dir: str
    outputs_dir: str
    sqlite_path: str
    faiss_dir: str
    model_dir: str


class ThresholdSettings(BaseModel):
    face_match: float
    body_match: float
    detector_confidence: float
    activity_confidence: float


class VideoSettings(BaseModel):
    fps_window: int
    tms_window: int
    frame_stride: int


class RuntimeSettings(BaseModel):
    device: str


class TrackerSettings(BaseModel):
    high_confidence_threshold: float
    low_confidence_threshold: float
    new_track_threshold: float
    match_iou_threshold: float
    max_time_lost: int


class ModelSettings(BaseModel):
    yolov10_weights: str
    arcface_weights: str
    arcface_model_name: str
    osnet_weights: str
    osnet_model_name: str
    stgcn_weights: str
    ollama_model: str


class DatabaseSettings(BaseModel):
    top_k: int


class AppConfig(BaseModel):
    app: dict[str, Any]
    runtime: RuntimeSettings
    paths: PathSettings
    video: VideoSettings
    tracker: TrackerSettings
    thresholds: ThresholdSettings
    models: ModelSettings
    database: DatabaseSettings


class EnvSettings(BaseSettings):
    vias_config_path: str = str(ROOT_DIR / "configs" / "settings.yaml")


@lru_cache
def get_settings() -> AppConfig:
    env = EnvSettings()
    with open(env.vias_config_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return AppConfig.model_validate(data)


def resolve_path(relative_path: str) -> Path:
    return ROOT_DIR / relative_path
