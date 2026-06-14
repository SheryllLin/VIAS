from __future__ import annotations

import json


def evaluate() -> None:
    metrics = {
        "detection_accuracy_target": 0.90,
        "reid_accuracy_target": 0.85,
        "activity_accuracy_target": 0.80,
        "query_accuracy_target": 0.90,
        "fps_target": "15-30",
    }
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    evaluate()
