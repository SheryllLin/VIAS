from __future__ import annotations

import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def train() -> None:
    writer = SummaryWriter(log_dir="models/osnet/tensorboard")
    for epoch in range(1, 11):
        loss = 1.0 / epoch
        accuracy = min(0.5 + epoch * 0.04, 0.9)
        writer.add_scalar("loss/train", loss, epoch)
        writer.add_scalar("accuracy/val", accuracy, epoch)
        Path("models/osnet").mkdir(parents=True, exist_ok=True)
        with open(f"models/osnet/checkpoint_epoch_{epoch}.pth", "w", encoding="utf-8") as handle:
            handle.write(f"mock checkpoint epoch {epoch}\n")
        time.sleep(0.01)
    writer.close()


if __name__ == "__main__":
    train()
