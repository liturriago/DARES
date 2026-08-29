"""FDA training engine: Fourier Domain Adaptation for UDA segmentation.

FDA (Yang & Soatto, CVPR 2020) adapts each source image to the target "style"
by swapping its low-frequency FFT amplitude with a randomly sampled target
image, then trains on cross-entropy over the adapted source plus a
Charbonnier-weighted entropy penalty on the target. No generators or
discriminators are trained.

``L = CE(x_{s->t}, y_s) + lambda_ent * L_ent(x_t)``
"""

import time
from typing import Any

import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm import tqdm

from dares.config import TrainConfig
from dares.losses.ce import SegCrossEntropyLoss
from dares.losses.fda import charbonnier_entropy, fourier_domain_adaptation
from dares.training.base_trainer import BaseTrainer


class FDATrainer(BaseTrainer):
    """Fourier Domain Adaptation engine (single scale + entropy).

    Parameters
    ----------
    model : nn.Module
        The segmentation model (a ``dares.models`` ``SegmentationModel``).
    source_loaders : dict[str, DataLoader]
        ``{"train", "validation", "test"}`` labeled source loaders.
    target_loaders : dict[str, DataLoader]
        ``{"train", "validation", "test"}`` target loaders; ``train`` is
        unlabeled.
    config : TrainConfig
        Training configuration (FDA hyperparameters under ``fda_*``).
    device : torch.device
        Computing device.
    """

    def __init__(
        self,
        model: nn.Module,
        source_loaders: dict[str, Any],
        target_loaders: dict[str, Any],
        config: TrainConfig,
        device: torch.device,
    ) -> None:
        """Initializes the criterion and optimizer."""
        super().__init__(model, source_loaders, target_loaders, config, device)
        self.criterion = SegCrossEntropyLoss()
        self.optimizer = self._make_optimizer(self.model.parameters())

        self.beta = float(config.fda_beta)
        self.lambda_entropy = float(config.fda_lambda_entropy)
        self.eta = float(config.fda_eta)

    def train_epoch(self) -> dict[str, float]:
        """Runs one epoch of FDA training.

        Returns
        -------
        dict[str, float]
            ``{"loss_total", "loss_ce", "loss_entropy", "epoch_time"}``; losses
            are per-iteration averages and ``epoch_time`` is the wall-clock
            duration in seconds.
        """
        self.model.train()
        src_iter, tgt_iter, num_batches = self._dual_iterators(
            self.source_loaders["train"], self.target_loaders["train"]
        )

        total_loss = 0.0
        ce_loss = 0.0
        ent_loss = 0.0
        start_time = time.time()

        pbar = tqdm(
            range(num_batches),
            desc=f"Train ({self.config.method})",
            leave=False,
        )
        for _ in pbar:
            imgs_s, masks_s = next(src_iter)
            imgs_t, _ = next(tgt_iter)
            imgs_s = imgs_s.to(self.device)
            masks_s = masks_s.to(self.device)
            imgs_t = imgs_t.to(self.device)

            adapted_s = fourier_domain_adaptation(imgs_s, imgs_t, self.beta)

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                logits_adapt = self.model(adapted_s, mode="class")
                loss_ce = self.criterion(logits_adapt, masks_s)
                logits_t = self.model(imgs_t, mode="class")
                loss_ent = charbonnier_entropy(logits_t, self.eta)
                total = loss_ce + self.lambda_entropy * loss_ent

            stepped = super()._backward_step(total, self.optimizer)
            if stepped:
                self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            pbar.set_postfix(loss=f"{total.item():.4f}")

            total_loss += float(total.detach().cpu().item())
            ce_loss += float(loss_ce.detach().cpu().item())
            ent_loss += float(loss_ent.detach().cpu().item())

        denom = max(num_batches, 1)
        return {
            "loss_total": total_loss / denom,
            "loss_ce": ce_loss / denom,
            "loss_entropy": ent_loss / denom,
            "epoch_time": time.time() - start_time,
        }
