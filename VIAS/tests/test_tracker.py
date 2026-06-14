from backend.models.schemas import DetectionRecord
from backend.services.tracker import ByteTrackService


def test_tracker_preserves_id_for_consistent_motion() -> None:
    tracker = ByteTrackService()
    frame_1 = [DetectionRecord(frame_id=1, person_bbox=[10, 10, 40, 80], confidence=0.9)]
    frame_2 = [DetectionRecord(frame_id=2, person_bbox=[12, 10, 42, 80], confidence=0.88)]
    first = tracker.update(frame_1, frame_id=1)
    second = tracker.update(frame_2, frame_id=2)
    assert len(first) == 1
    assert len(second) == 1
    assert first[0].track_id == second[0].track_id
