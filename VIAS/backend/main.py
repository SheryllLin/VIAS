from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.api.realtime_routes import router as realtime_router
from backend.database.db import init_db
from backend.utils.logging import configure_logging


configure_logging()
init_db()

app = FastAPI(title="VIAS", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
app.include_router(realtime_router)
