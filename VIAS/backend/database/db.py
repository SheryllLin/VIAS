from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from backend.config import get_settings, resolve_path


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT UNIQUE,
    label TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS tracks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER,
    frame_id INTEGER,
    timestamp REAL,
    bbox TEXT,
    identity TEXT,
    confidence REAL
);
CREATE TABLE IF NOT EXISTS activities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER,
    activity TEXT,
    confidence REAL,
    timestamp REAL
);
CREATE TABLE IF NOT EXISTS tms_vectors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER,
    vector TEXT,
    similarity_score REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT,
    timestamp REAL,
    activity TEXT,
    location TEXT,
    track_id INTEGER,
    identity TEXT,
    confidence REAL,
    metadata TEXT
);
CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text TEXT,
    generated_sql TEXT,
    response_text TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def get_db_path() -> Path:
    settings = get_settings()
    return resolve_path(settings.paths.sqlite_path)


def init_db() -> None:
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.commit()


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
