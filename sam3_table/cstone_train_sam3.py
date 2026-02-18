from __future__ import annotations

import os

from training_config import SAM3LoRAConfig
from train_sam3_lora_native import SAM3TrainerNative


def train_sam3(
    config: SAM3LoRAConfig,
    device: list[int] | None = None,
) -> None:
    if device is None:
        device = [0]

    multi_gpu = len(device) > 1 and "LOCAL_RANK" in os.environ

    if not multi_gpu and len(device) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device[0])
        print(f"Using single GPU: {device[0]}")

    trainer = SAM3TrainerNative(config, multi_gpu=multi_gpu)
    trainer.train()
