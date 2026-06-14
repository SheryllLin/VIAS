from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backend.config import resolve_path


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    task: str
    local_dir: Path
    annotation_hint: str


class DatasetRegistry:
    def __init__(self) -> None:
        base = resolve_path("datasets")
        self.specs = {
            "crowdhuman": DatasetSpec("CrowdHuman", "detection", base / "CrowdHuman", "annotation_train.odgt"),
            "coco2017": DatasetSpec("COCO 2017", "detection", base / "COCO2017", "annotations/instances_train2017.json"),
            "mot17": DatasetSpec("MOT17", "tracking", base / "MOT17", "train/*/gt/gt.txt"),
            "lfw": DatasetSpec("LFW", "face_recognition", base / "LFW", "pairs.txt"),
            "market1501": DatasetSpec("Market1501", "reid", base / "Market1501", "bounding_box_train"),
            "prw": DatasetSpec("PRW", "reid", base / "PRW", "frames"),
            "ntu_rgb_d_120": DatasetSpec("NTU RGB+D 120", "activity", base / "NTU_RGBD_120", "skeletons"),
            "cctv_action": DatasetSpec("CCTV Action Recognition Dataset", "activity", base / "CCTV_Action", "annotations"),
        }

    def status(self) -> list[dict[str, str | bool]]:
        return [
            {
                "key": key,
                "name": spec.name,
                "task": spec.task,
                "path": str(spec.local_dir),
                "present": spec.local_dir.exists(),
                "annotation_hint": spec.annotation_hint,
            }
            for key, spec in self.specs.items()
        ]
