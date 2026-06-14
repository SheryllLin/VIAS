from __future__ import annotations

from collections import defaultdict

import numpy as np

from backend.models.schemas import ActivityRecord, TMSRecord
from backend.services.embedding_utils import deterministic_embedding
from backend.services.faiss_store import FAISSStore


class BehaviorDiscoveryService:
    def __init__(self, store: FAISSStore) -> None:
        self.store = store
        self.activity_history: dict[int, list[str]] = defaultdict(list)

    def update(self, activity: ActivityRecord, tms_record: TMSRecord | None) -> dict:
        history = self.activity_history[activity.track_id]
        history.append(activity.activity)
        summary = self._summarize(activity.track_id, history, tms_record)
        embedding = deterministic_embedding(summary, 128).reshape(1, -1)
        self.store.add(
            "behavior_embeddings",
            embedding,
            [{"track_id": activity.track_id, "summary": summary}],
        )
        return {"track_id": activity.track_id, "summary": summary}

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        embedding = deterministic_embedding(query, 128)
        return self.store.search("behavior_embeddings", embedding, top_k=top_k)

    def _summarize(self, track_id: int, history: list[str], tms_record: TMSRecord | None) -> str:
        dominant = max(set(history), key=history.count)
        cadence = "repeatedly" if history.count(dominant) > 2 else "once"
        motion = "with strong gait signature" if tms_record and tms_record.similarity_score > 0.3 else "with limited motion signature"
        return f"Track {track_id} {cadence} performed {dominant.lower()} {motion}"
