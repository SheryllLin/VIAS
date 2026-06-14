import numpy as np

from backend.services.faiss_store import FAISSStore
from backend.services.reid import MultiTierReIDService


def test_reid_falls_back_without_runtime_models() -> None:
    service = MultiTierReIDService(FAISSStore())
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    result = service.identify(track_id=1, frame_id=1, pose_available=False, frame=frame, bbox=[5, 5, 40, 60])
    assert result.track_id == 1
    assert result.match_source in {"none", "arcface", "osnet"}
