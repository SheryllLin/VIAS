from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


def train() -> None:
    model = YOLO("yolov10n.pt")
    model.train(
        data=str(Path("datasets") / "detection.yaml"),
        epochs=50,
        imgsz=640,
        batch=16,
        project="models/yolov10",
        name="train",
        exist_ok=True,
        patience=10,
        resume=False,
    )


if __name__ == "__main__":
    train()
