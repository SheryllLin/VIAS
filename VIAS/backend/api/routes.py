from __future__ import annotations

import shutil
import uuid
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.config import get_settings, resolve_path
from backend.database.repository import EventRepository
from backend.models.schemas import QueryRequest, QueryResponse, SearchPersonRequest
from backend.services.analytics import AnalyticsService
from backend.services.behavior import BehaviorDiscoveryService
from backend.services.datasets import DatasetRegistry
from backend.services.embedding_utils import deterministic_embedding
from backend.services.faiss_store import FAISSStore
from backend.services.model_registry import ModelRegistry
from backend.services.query_engine import NLQueryEngine
from backend.services.reid import MultiTierReIDService
from backend.services.video_pipeline import VideoAnalyticsPipeline


router = APIRouter()
settings = get_settings()


@lru_cache
def get_repository() -> EventRepository:
    return EventRepository()


@lru_cache
def get_store() -> FAISSStore:
    return FAISSStore()


@lru_cache
def get_behavior_service() -> BehaviorDiscoveryService:
    return BehaviorDiscoveryService(get_store())


@lru_cache
def get_analytics_service() -> AnalyticsService:
    return AnalyticsService(get_repository())


@lru_cache
def get_query_engine() -> NLQueryEngine:
    return NLQueryEngine(get_repository(), get_behavior_service())


@lru_cache
def get_reid_service() -> MultiTierReIDService:
    return MultiTierReIDService(get_store())


@lru_cache
def get_pipeline() -> VideoAnalyticsPipeline:
    return VideoAnalyticsPipeline()


@lru_cache
def get_dataset_registry() -> DatasetRegistry:
    return DatasetRegistry()


@lru_cache
def get_model_registry() -> ModelRegistry:
    return ModelRegistry()


def _save_upload(file: UploadFile, subdir: str) -> Path:
    target_dir = resolve_path(subdir)
    target_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "").suffix or ".bin"
    target_path = target_dir / f"{uuid.uuid4()}{suffix}"
    with target_path.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    return target_path


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "VIAS"}


@router.post("/upload-video")
def upload_video(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name.")
    path = _save_upload(file, settings.paths.uploads_dir)
    result = get_pipeline().process_video(path)
    return result.model_dump()


@router.post("/upload-reference-image")
def upload_reference_image(file: UploadFile = File(...), person_id: str = "reference-person") -> dict:
    path = _save_upload(file, settings.paths.uploads_dir)
    return get_reid_service().register_reference(person_id, str(path))


@router.post("/search-person")
def search_person(request: SearchPersonRequest) -> dict:
    if request.person_id:
        query = deterministic_embedding(f"face-ref:{request.person_id}:lookup", 512)
        results = get_store().search("face_embeddings", query, top_k=request.top_k)
        return {"results": [item for item in results if item.get("person_id") == request.person_id]}
    return {"results": get_repository().fetch_table("tracks")[: request.top_k]}


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    return get_query_engine().query(request.query)


@router.get("/activities")
def activities() -> list[dict]:
    return get_repository().fetch_table("activities")


@router.get("/events")
def events() -> list[dict]:
    return get_repository().fetch_table("events")


@router.get("/tracks")
def tracks() -> list[dict]:
    return get_repository().fetch_table("tracks")


@router.post("/behavior-search")
def behavior_search(request: QueryRequest) -> dict:
    return {"results": get_behavior_service().search(request.query)}


@router.get("/analytics")
def analytics() -> dict:
    return get_analytics_service().summary().model_dump()


@router.get("/datasets")
def datasets() -> dict:
    return {"datasets": get_dataset_registry().status()}


@router.get("/models/status")
def models_status() -> dict:
    return {"models": get_model_registry().status()}
