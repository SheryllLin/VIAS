from __future__ import annotations

import json
from typing import Any

from backend.database.db import get_connection
from backend.models.schemas import ActivityRecord, EventRecord, TMSRecord, TrackRecord


class EventRepository:
    def add_track(self, track: TrackRecord, timestamp: float, identity: str | None, confidence: float) -> None:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO tracks (track_id, frame_id, timestamp, bbox, identity, confidence) VALUES (?, ?, ?, ?, ?, ?)",
                (track.track_id, track.frame_id, timestamp, json.dumps(track.bbox), identity, confidence),
            )

    def add_activity(self, activity: ActivityRecord) -> None:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO activities (track_id, activity, confidence, timestamp) VALUES (?, ?, ?, ?)",
                (activity.track_id, activity.activity, activity.confidence, activity.timestamp),
            )

    def add_tms_vector(self, record: TMSRecord) -> None:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO tms_vectors (track_id, vector, similarity_score) VALUES (?, ?, ?)",
                (record.track_id, json.dumps(record.tms_vector), record.similarity_score),
            )

    def add_event(self, event: EventRecord) -> None:
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO events (event_type, timestamp, activity, location, track_id, identity, confidence, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_type,
                    event.timestamp,
                    event.metadata.get("activity"),
                    event.location,
                    event.track_id,
                    event.identity,
                    event.confidence,
                    json.dumps(event.metadata),
                ),
            )

    def add_query(self, query_text: str, generated_sql: str | None, response_text: str) -> None:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO queries (query_text, generated_sql, response_text) VALUES (?, ?, ?)",
                (query_text, generated_sql, response_text),
            )

    def fetch_table(self, table_name: str) -> list[dict[str, Any]]:
        with get_connection() as conn:
            rows = conn.execute(f"SELECT * FROM {table_name} ORDER BY id DESC LIMIT 200").fetchall()
        return [dict(row) for row in rows]

    def count_rows(self, table_name: str) -> int:
        with get_connection() as conn:
            row = conn.execute(f"SELECT COUNT(*) AS total FROM {table_name}").fetchone()
        return int(row["total"]) if row else 0

    def run_sql(self, sql: str) -> list[dict[str, Any]]:
        with get_connection() as conn:
            rows = conn.execute(sql).fetchall()
        return [dict(row) for row in rows]
