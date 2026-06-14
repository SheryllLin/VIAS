from __future__ import annotations

from backend.config import get_settings, resolve_path


class ModelRegistry:
    def __init__(self) -> None:
        settings = get_settings()
        self.paths = {
            "yolov10": resolve_path(settings.models.yolov10_weights),
            "arcface": resolve_path(settings.models.arcface_weights),
            "osnet": resolve_path(settings.models.osnet_weights),
            "stgcn": resolve_path(settings.models.stgcn_weights),
        }

    def status(self) -> list[dict[str, str | bool]]:
        return [
            {"name": name, "path": str(path), "present": path.exists()}
            for name, path in self.paths.items()
        ]
