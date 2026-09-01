"""CLAN training engine: category-level adversarial output-space adaptation.

Implements the core of Ruan et al. (2019), "Category-Level Adversaries for
Semantic Domain Adaptation" (CAA-Net / CLAN). The segmenter is supervised on
the labeled source domain while a multi-category discriminator acts on *slices*
of the predictions (single masked class channels) to close the source/target
output-space gap.

Each iteration runs two alternating gradient steps:

1. **Discriminator step** -- for every present class the masked class slice of
   the (detached) source prediction (masked by the real labels) and of the
   target prediction (masked by its pseudo-labels) is fed to the
   discriminator. Source slices are labeled with their own category and target
   slices with the extra "target" category.
2. **Segmenter step** -- the cross-entropy on the source plus the adversarial
   feedback pushing the target masked slices towards their pseudo-label
   categories (tricking the discriminator).
"""

import time
from typing import Any

import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm import tqdm

from dares.config import TrainConfig
from dares.losses.ce import SegCrossEntropyLoss
from dares.losses.clan import (
    CLANDiscriminator,
    clan_adversarial_loss,
    clan_discriminator_loss,
    masked_class_slices,
    target_pseudo_slices,
)
from dares.training.base_trainer import BaseTrainer


class CLANTrainer(BaseTrainer):
    """Trainer implementing category-level output-space adversarial adaptation."""

    def __init__(
        self,
        model: nn.Module,
        source_loaders: dict[str, Any],
        target_loaders: dict[str, Any],
        config: TrainConfig,
        device: torch.device,
    ) -> None:
        super().__init__(model, source_loaders, target_loaders, config, device)
        self.criterion = SegCrossEntropyLoss()
        self.discriminator = CLANDiscriminator(
            num_classes=self.num_classes,
            base=int(getattr(config, "clan_disc_base", 32)),
            num_layers=int(getattr(config, "clan_disc_layers", 3)),
        ).to(device)
        self.optimizer = self._make_optimizer(self.model.parameters())
        self.optimizer_d = self._make_optimizer(
            self.discriminator.parameters(), lr=config.lr_d
        )
        self.lambda_clan = float(config.lambda_clan)
        self.clan_threshold = float(config.clan_threshold)

    def train_epoch(self) -> dict[str, float]:
        """Runs a single CLAN training epoch over the paired domains.

        Returns
        -------
        dict[str, float]
            Weighted-mean scalar metrics: ``loss_total``, ``loss_ce``,
            ``loss_disc``, ``loss_adv_clan`` and ``epoch_time`` (seconds).
        """
        self.model.train()
        self.discriminator.train()
        epoch_start = time.time()

        src_iter, tgt_iter, num_batches = self._dual_iterators(
            self.source_loaders["train"], self.target_loaders["train"]
        )

        running_ce = 0.0
        running_disc = 0.0
        running_adv = 0.0

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

            self.optimizer.zero_grad(set_to_none=True)
            self.optimizer_d.zero_grad(set_to_none=True)

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                logits_s = self.model(imgs_s, mode="class")
                logits_t = self.model(imgs_t, mode="class")

                # Source masked slices (mask from real labels) detached.
                slices_s, labels_s = masked_class_slices(
                    logits_s.detach(), masks_s
                )
                # Target masked slices (mask from pseudo-labels) detached.
                slices_t, labels_t, _ = target_pseudo_slices(
                    logits_t.detach(), self.clan_threshold
                )

            # 1. Discriminator step.
            d_s = self.discriminator(slices_s)
            d_t = self.discriminator(slices_t)
            loss_disc = clan_discriminator_loss(
                d_s,
                labels_s,
                d_t,
                labels_t,
                self.num_classes,
                lambda_out=self.lambda_clan,
            )
            stepped = self._backward_step(
                loss_disc, self.optimizer_d, self.discriminator.parameters()
            )
            if stepped:
                self.scaler.update()

            # 2. Segmenter step: CE on source + adversarial target feedback.
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                d_t = self.discriminator(slices_t)
                loss_adv = clan_adversarial_loss(d_t, labels_t)
                loss_ce = self.criterion(logits_s, masks_s)
                total = loss_ce + self.lambda_clan * loss_adv

            stepped = self._backward_step(total, self.optimizer, self.model.parameters())
            if stepped:
                self.scaler.update()

            running_ce += float(loss_ce.item())
            running_disc += float(loss_disc.item())
            running_adv += float(loss_adv.item())
            pbar.set_postfix(loss=f"{total.item():.4f}")

        num = max(num_batches, 1)
        loss_ce = running_ce / num
        loss_disc = running_disc / num
        loss_adv = running_adv / num
        return {
            "loss_total": loss_ce + self.lambda_clan * loss_adv,
            "loss_ce": loss_ce,
            "loss_disc": loss_disc,
            "loss_adv_clan": loss_adv,
            "epoch_time": time.time() - epoch_start,
        }
