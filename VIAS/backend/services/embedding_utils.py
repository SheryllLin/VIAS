from __future__ import annotations

import hashlib

import numpy as np


def deterministic_embedding(seed_text: str, dimension: int) -> np.ndarray:
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    values = np.frombuffer((digest * ((dimension // len(digest)) + 1))[:dimension], dtype=np.uint8)
    vector = values.astype(np.float32)
    vector = (vector - vector.mean()) / (vector.std() + 1e-6)
    return vector
