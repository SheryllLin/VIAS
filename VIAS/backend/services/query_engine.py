from __future__ import annotations

from typing import Any

try:
    import ollama  # type: ignore
except ImportError:  # pragma: no cover - optional dependency fallback
    ollama = None

from backend.config import get_settings
from backend.database.repository import EventRepository
from backend.models.schemas import QueryResponse
from backend.services.behavior import BehaviorDiscoveryService


class NLQueryEngine:
    def __init__(self, repository: EventRepository, behavior_service: BehaviorDiscoveryService) -> None:
        self.repository = repository
        self.behavior_service = behavior_service
        self.model_name = get_settings().models.ollama_model

    def query(self, prompt: str) -> QueryResponse:
        lower = prompt.lower()
        sql = None
        results: list[dict[str, Any]] = []
        if "behavior" in lower or "walked" in lower or "raising" in lower:
            results = self.behavior_service.search(prompt)
            answer = self._format_behavior_answer(prompt, results)
        else:
            sql = self._generate_sql(prompt)
            try:
                results = self.repository.run_sql(sql) if sql else []
            except Exception as exc:  # pragma: no cover - defensive
                answer = f"Query execution failed: {exc}"
                self.repository.add_query(prompt, sql, answer)
                return QueryResponse(answer=answer, sql=sql, results=results)
            answer = self._summarize(prompt, results)
        self.repository.add_query(prompt, sql, answer)
        return QueryResponse(answer=answer, sql=sql, results=results)

    def _generate_sql(self, prompt: str) -> str:
        system_prompt = (
            "You translate surveillance questions to SQLite SQL. "
            "Available tables: persons, tracks, activities, tms_vectors, events, queries. "
            "Return only SQL."
        )
        try:
            if ollama is None:
                raise RuntimeError("ollama not installed")
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )
            return response["message"]["content"].strip().strip("`")
        except Exception:
            if "waved" in prompt.lower():
                return "SELECT * FROM activities WHERE LOWER(activity) = 'waving' ORDER BY timestamp DESC LIMIT 20"
            if "entered" in prompt.lower():
                return "SELECT * FROM events WHERE LOWER(event_type) = 'entry' ORDER BY timestamp DESC LIMIT 20"
            return "SELECT * FROM events ORDER BY timestamp DESC LIMIT 20"

    @staticmethod
    def _summarize(prompt: str, results: list[dict[str, Any]]) -> str:
        if not results:
            return f"No results found for: {prompt}"
        return f"Found {len(results)} result(s) for: {prompt}"

    @staticmethod
    def _format_behavior_answer(prompt: str, results: list[dict[str, Any]]) -> str:
        if not results:
            return f"No behavior matches found for: {prompt}"
        tracks = ", ".join(str(item.get("track_id")) for item in results)
        return f"Behavior search matched tracks: {tracks}"
