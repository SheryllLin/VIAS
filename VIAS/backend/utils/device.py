from __future__ import annotations

try:
    import torch
except Exception:  # pragma: no cover - optional dependency fallback
    torch = None

from backend.config import get_settings


def get_preferred_device() -> str:
    configured = get_settings().runtime.device.lower()
    if configured != "auto":
        return configured
    if torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"
