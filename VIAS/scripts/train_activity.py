from __future__ import annotations

import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def train() -> None:
    writer = SummaryWriter(log_dir="models/stgcn/tensorboard")
    patience = 3
    best = 0.0
    wait = 0
    Path("models/stgcn").mkdir(parents=True, exist_ok=True)
    for epoch in range(1, 16):
        acc = min(0.55 + epoch * 0.02, 0.85)
        writer.add_scalar("accuracy/val", acc, epoch)
        with open(f"models/stgcn/checkpoint_epoch_{epoch}.pth", "w", encoding="utf-8") as handle:
            handle.write(f"mock checkpoint epoch {epoch}\n")
        if acc > best:
            best = acc
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            break
        time.sleep(0.01)
    writer.close()


if __name__ == "__main__":
    train()
