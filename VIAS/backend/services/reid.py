from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    from insightface.app import FaceAnalysis
except Exception:  # pragma: no cover - optional dependency fallback
    FaceAnalysis = None

try:
    from torchreid.utils import FeatureExtractor
except Exception:  # pragma: no cover - optional dependency fallback
    FeatureExtractor = None

from backend.config import get_settings, resolve_path
from backend.models.schemas import IdentityRecord
from backend.services.embedding_utils import deterministic_embedding
from backend.services.faiss_store import FAISSStore
from backend.utils.device import get_preferred_device
from backend.utils.logging import logger


@dataclass
class EmbeddingResult:
    embedding: np.ndarray | None
    confidence: float


class ArcFaceService:
    def __init__(self) -> None:
        settings = get_settings()
        self.model_name = settings.models.arcface_model_name
        self.device = get_preferred_device()
        self.app = None
        if FaceAnalysis is None:
            return
        try:
            providers = ["CPUExecutionProvider"]
            self.app = FaceAnalysis(name=self.model_name, providers=providers)
            self.app.prepare(ctx_id=0, det_size=(640, 640))
        except Exception as exc:  # pragma: no cover - dependency-specific
            logger.warning("ArcFace initialization failed: %s", exc)
            self.app = None

    @property
    def available(self) -> bool:
        return self.app is not None

    def extract_from_image(self, image: np.ndarray) -> EmbeddingResult:
        if self.app is None:
            return EmbeddingResult(None, 0.0)
        try:
            faces = self.app.get(image)
        except Exception as exc:  # pragma: no cover - dependency-specific
            logger.warning("ArcFace inference failed: %s", exc)
            return EmbeddingResult(None, 0.0)
        if not faces:
            return EmbeddingResult(None, 0.0)
        best_face = max(faces, key=lambda face: float(face.det_score))
        embedding = np.asarray(best_face.embedding, dtype=np.float32)
        return EmbeddingResult(embedding, float(best_face.det_score))


class OSNetService:
    def __init__(self) -> None:
        settings = get_settings()
        self.device = get_preferred_device()
        self.weights_path = resolve_path(settings.models.osnet_weights)
        self.model_name = settings.models.osnet_model_name
        self.extractor = None
        if FeatureExtractor is None:
            return
        try:
            model_path = str(self.weights_path) if self.weights_path.exists() else ""
            self.extractor = FeatureExtractor(
                model_name=self.model_name,
                model_path=model_path or None,
                device=self.device if self.device in {"cpu", "cuda"} else "cpu",
            )
        except Exception as exc:  # pragma: no cover - dependency-specific
            logger.warning("OSNet initialization failed: %s", exc)
            self.extractor = None

    @property
    def available(self) -> bool:
        return self.extractor is not None

    def extract_from_image(self, image: np.ndarray) -> EmbeddingResult:
        if self.extractor is None:
            return EmbeddingResult(None, 0.0)
        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            embedding = self.extractor([rgb])[0]
            vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
            confidence = float(np.linalg.norm(vector) / (np.linalg.norm(vector) + 1e-6))
            return EmbeddingResult(vector, confidence)
        except Exception as exc:  # pragma: no cover - dependency-specific
            logger.warning("OSNet inference failed: %s", exc)
            return EmbeddingResult(None, 0.0)


class MultiTierReIDService:
    def __init__(self, store: FAISSStore) -> None:
        settings = get_settings()
        self.face_threshold = settings.thresholds.face_match
        self.body_threshold = settings.thresholds.body_match
        self.store = store
        self.arcface = ArcFaceService()
        self.osnet = OSNetService()

    def identify(
        self,
        track_id: int,
        frame_id: int,
        pose_available: bool,
        frame: np.ndarray | None = None,
        bbox: list[float] | None = None,
    ) -> IdentityRecord:
        face_embedding, body_embedding, extraction_metadata = self._extract_embeddings(track_id, frame_id, frame, bbox)
        face_matches = self.store.search("face_embeddings", face_embedding, top_k=1)
        body_matches = self.store.search("body_embeddings", body_embedding, top_k=1)
        face_score = face_matches[0]["score"] if face_matches else 0.0
        body_score = body_matches[0]["score"] if body_matches else 0.0
        person_id = None
        tms_confidence = 0.0
        match_source = "none"
        if face_score >= self.face_threshold:
            person_id = face_matches[0].get("person_id")
            match_source = "arcface"
        elif body_score >= self.body_threshold:
            person_id = body_matches[0].get("person_id")
            match_source = "osnet"
        elif pose_available:
            person_id = f"tms-candidate-{track_id}"
            tms_confidence = 0.5
            match_source = "tms"
        return IdentityRecord(
            track_id=track_id,
            person_id=person_id,
            face_confidence=max(float(face_score), extraction_metadata["face_confidence"]),
            body_confidence=max(float(body_score), extraction_metadata["body_confidence"]),
            tms_confidence=float(tms_confidence),
            match_source=match_source,
        )

    def register_reference(self, person_id: str, image_path: str) -> dict:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Unable to read image at {image_path}")
        face_result = self.arcface.extract_from_image(image)
        body_result = self.osnet.extract_from_image(image)
        if face_result.embedding is None:
            face_result = EmbeddingResult(deterministic_embedding(f"face-ref:{person_id}:{image_path}", 512), 0.0)
        if body_result.embedding is None:
            body_result = EmbeddingResult(deterministic_embedding(f"body-ref:{person_id}:{image_path}", 512), 0.0)
        self.store.add("face_embeddings", face_result.embedding.reshape(1, -1), [{"person_id": person_id, "image_path": image_path}])
        self.store.add("body_embeddings", body_result.embedding.reshape(1, -1), [{"person_id": person_id, "image_path": image_path}])
        return {
            "person_id": person_id,
            "image_path": image_path,
            "face_embedding": bool(face_result.embedding is not None),
            "body_embedding": bool(body_result.embedding is not None),
            "arcface_available": self.arcface.available,
            "osnet_available": self.osnet.available,
        }

    def _extract_embeddings(
        self, track_id: int, frame_id: int, frame: np.ndarray | None, bbox: list[float] | None
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        face_confidence = 0.0
        body_confidence = 0.0
        face_embedding = None
        body_embedding = None
        if frame is not None and bbox is not None:
            crop = self._crop_bbox(frame, bbox)
            if crop is not None:
                face_result = self.arcface.extract_from_image(crop)
                body_result = self.osnet.extract_from_image(crop)
                face_embedding = face_result.embedding
                body_embedding = body_result.embedding
                face_confidence = face_result.confidence
                body_confidence = body_result.confidence
        if face_embedding is None:
            face_embedding = deterministic_embedding(f"face:{track_id}:{frame_id}", 512)
        if body_embedding is None:
            body_embedding = deterministic_embedding(f"body:{track_id}:{frame_id}", 512)
        return face_embedding, body_embedding, {
            "face_confidence": float(face_confidence),
            "body_confidence": float(body_confidence),
        }

    @staticmethod
    def _crop_bbox(frame: np.ndarray, bbox: list[float]) -> np.ndarray | None:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size else None
