from __future__ import annotations

from collections import Counter

from backend.database.repository import EventRepository
from backend.models.schemas import AnalyticsSummary


class AnalyticsService:
    def __init__(self, repository: EventRepository) -> None:
        self.repository = repository

    def summary(self) -> AnalyticsSummary:
        tracks = self.repository.fetch_table("tracks")
        activities = self.repository.fetch_table("activities")
        events = self.repository.fetch_table("events")
        activity_distribution = Counter(item["activity"] for item in activities if item.get("activity"))
        known_identities = sum(1 for item in tracks if item.get("identity"))
        unknown_identities = len(tracks) - known_identities
        timeline = [
            {
                "timestamp": event.get("timestamp"),
                "track_id": event.get("track_id"),
                "activity": event.get("activity"),
                "event_type": event.get("event_type"),
            }
            for event in events[:50]
        ]
        return AnalyticsSummary(
            detection_count=len(tracks),
            tracked_persons=len({item["track_id"] for item in tracks if item.get("track_id") is not None}),
            activity_distribution=dict(activity_distribution),
            reid_statistics={"identified": known_identities, "unidentified": unknown_identities},
            avg_fps=0.0,
            timeline=timeline,
        )
