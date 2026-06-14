import numpy as np

from backend.models.schemas import PoseRecord
from backend.services.faiss_store import FAISSStore
from backend.services.tms import TemporalMotionSignatureService


def test_tms_vector_has_expected_dimension() -> None:
    service = TemporalMotionSignatureService(FAISSStore(), window_size=2)
    pose = [[0.1, 0.2, 0.0] for _ in range(33)]
    record_1 = PoseRecord(track_id=1, frame_id=1, keypoints=pose)
    record_2 = PoseRecord(track_id=1, frame_id=2, keypoints=[[0.2, 0.3, 0.0] for _ in range(33)])
    assert service.update(record_1) is None
    result = service.update(record_2)
    assert result is not None
    assert len(result.tms_vector) == 16
    assert isinstance(np.array(result.tms_vector), np.ndarray)
