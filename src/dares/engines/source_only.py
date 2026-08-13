"""Source-only training engine: the supervised baseline with no adaptation.

The source-only engine trains the segmentation model exclusively on the labeled
source domain using a pixel-wise cross-entropy loss. It performs no target
adaptation, so the inherited ``fit`` loop is used unchanged: it still evaluates
on the target validation split and reports its mIoU (the "adaptation gap") so
that this baseline can be compared directly against the UDA methods.
"""

import time
from typing import Any

import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm import tqdm

from dares.config import TrainConfig
from dares.losses import SegCrossEntropyLoss
from dares.training.base_trainer import BaseTrainer


class SourceOnlyTrainer(BaseTrainer):
    """Supervised cross-entropy baseline trained on the source domain only.

    Parameters
    ----------
    model : nn.Module
        The segmentation model (a ``dares.models`` ``SegmentationModel``).
    source_loaders : dict[str, DataLoader]
        ``{"train", "validation", "test"}`` labeled source loaders.
    target_loaders : dict[str, DataLoader]
        ``{"train", "validation", "test"}`` target loaders; ``train`` is
        unlabeled and only the validation split is used by ``fit``.
    config : TrainConfig
        Training configuration.
    device : torch.device
        Computing device.

    Attributes
    ----------
    criterion : SegCrossEntropyLoss
        Pixel-wise cross-entropy loss over the labeled source batches.
    optimizer : torch.optim.Adam
        Adam optimizer over the model parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        source_loaders: dict[str, Any],
        target_loaders: dict[str, Any],
        config: TrainConfig,
        device: torch.device,
    ) -> None:
        """Initializes the trainer, the loss and the optimizer."""
        super().__init__(model, source_loaders, target_loaders, config, device)
        self.criterion = SegCrossEntropyLoss()
        self.optimizer = self._make_optimizer(self.model.parameters())

    def train_epoch(self) -> dict[str, float]:
        """Runs one epoch of source-only supervised training.

        Iterates only the source training loader. For each batch it computes
        the pixel-wise cross-entropy under the AMP context and performs a
        GradScaler-managed optimizer step.

        Returns
        -------
        dict[str, float]
            ``{"train_loss", "train_acc", "epoch_time"}``; the loss and
            accuracy are pixel-weighted epoch averages and ``epoch_time`` is
            the wall-clock duration of the epoch in seconds.
        """
        self.model.train()
        total_loss = 0.0
        correct_pixels = 0
        total_pixels = 0
        start_time = time.time()

        pbar = tqdm(
            self.source_loaders["train"],
            desc=f"Train ({self.config.method})",
            leave=False,
        )
        for batch in pbar:
            imgs, masks = batch[0].to(self.device), batch[1].to(self.device)
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(imgs, mode="class")
                loss = self.criterion(logits, masks)

            num_pixels = logits.numel() // int(self.num_classes)
            total_loss += float(loss.detach().cpu().item()) * num_pixels
            if masks is not None:
                preds = torch.argmax(logits.detach(), dim=1)
                correct_pixels += int((preds == masks).sum().item())
                total_pixels += num_pixels

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        epoch_time = time.time() - start_time
        if total_pixels == 0:
            train_acc = 0.0
            train_loss = 0.0
        else:
            train_acc = float(correct_pixels / total_pixels)
            train_loss = float(total_loss / total_pixels)
        return {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "epoch_time": float(epoch_time),
        }
