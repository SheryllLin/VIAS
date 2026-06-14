from __future__ import annotations

import numpy as np
from torch.utils.tensorboard import SummaryWriter


def train() -> None:
    writer = SummaryWriter(log_dir="models/tms/tensorboard")
    for epoch in range(1, 21):
        loss = float(np.exp(-epoch / 10))
        writer.add_scalar("loss/train", loss, epoch)
    writer.close()


if __name__ == "__main__":
    train()
