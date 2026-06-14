from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - optional dependency fallback
    faiss = None

from backend.config import get_settings, resolve_path


class FAISSStore:
    def __init__(self) -> None:
        settings = get_settings()
        self.base_dir = resolve_path(settings.paths.faiss_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.indices: dict[str, object] = {}
        self.metadata_paths: dict[str, Path] = {}

    def _ensure_index(self, name: str, dimension: int) -> object:
        if name not in self.indices:
            self.indices[name] = faiss.IndexFlatIP(dimension) if faiss is not None else []
            self.metadata_paths[name] = self.base_dir / f"{name}.jsonl"
        return self.indices[name]

    def add(self, name: str, embeddings: np.ndarray, payloads: list[dict]) -> None:
        index = self._ensure_index(name, embeddings.shape[1])
        normalized = embeddings.astype("float32")
        if faiss is not None:
            faiss.normalize_L2(normalized)
            index.add(normalized)  # type: ignore[union-attr]
        else:
            normalized /= np.linalg.norm(normalized, axis=1, keepdims=True) + 1e-6
            index.extend(normalized)  # type: ignore[union-attr]
        metadata_path = self.metadata_paths[name]
        with open(metadata_path, "a", encoding="utf-8") as handle:
            for payload in payloads:
                handle.write(json.dumps(payload) + "\n")

    def search(self, name: str, query: np.ndarray, top_k: int = 5) -> list[dict]:
        if name not in self.indices:
            return []
        normalized = query.astype("float32").reshape(1, -1)
        if faiss is not None:
            faiss.normalize_L2(normalized)
            scores, indices = self.indices[name].search(normalized, top_k)  # type: ignore[union-attr]
        else:
            normalized /= np.linalg.norm(normalized, axis=1, keepdims=True) + 1e-6
            vectors = np.array(self.indices[name], dtype=np.float32)
            if vectors.size == 0:
                return []
            scores = normalized @ vectors.T
            top_indices = np.argsort(scores[0])[::-1][:top_k]
            indices = np.array([top_indices], dtype=int)
            scores = np.array([[scores[0][idx] for idx in top_indices]], dtype=np.float32)
        payloads = self._load_payloads(name)
        results: list[dict] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0 or idx >= len(payloads):
                continue
            result = dict(payloads[idx])
            result["score"] = float(score)
            results.append(result)
        return results

    def _load_payloads(self, name: str) -> list[dict]:
        path = self.metadata_paths.get(name)
        if not path or not path.exists():
            return []
        with open(path, "r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
