"""ADVENT training engine: Adversarial Entropy Minimization for UDA.

Implements the ADVENT method (Tsai et al. 2019, "Learning to Adapt Structured
Output Space for Semantic Segmentation"). The segmenter is supervised on the
labeled source domain while a domain discriminator classifies the entropy maps
of the predictions to align the source and target output spaces. An explicit
entropy-minimization term further encourages confident predictions on the
unlabeled target domain.
"""

import time
from typing import Any

import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm import tqdm

from dares.config import TrainConfig
from dares.losses.advent import EntropyLoss, entropy_map
from dares.losses.ce import SegCrossEntropyLoss
from dares.losses.domain import DomainDiscriminator, adversarial_loss
from dares.training.base_trainer import BaseTrainer


class ADVENTTrainer(BaseTrainer):
    """Trainer implementing adversarial entropy minimization (ADVENT).

    Each iteration runs two alternating gradient steps:

    1. **Discriminator step** -- the entropy maps of the (detached) source and
       target predictions are classified into their domain. Source maps are
       labeled ``1`` and target maps ``0``.
    2. **Segmenter step** -- the cross-entropy on the source plus the
       adversarial loss (target entropy maps pushed towards the source label)
       plus the explicit target entropy minimization term.

    Parameters
    ----------
    model : nn.Module
        The segmentation model (``dares.models`` ``SegmentationModel``).
    source_loaders : dict[str, Any]
        ``{"train", "validation", "test"}`` labeled source loaders.
    target_loaders : dict[str, Any]
        ``{"train", "validation", "test"}`` target loaders; ``train`` is
        unlabeled.
    config : TrainConfig
        Training configuration; ``lambda_adv`` and ``lambda_entropy`` weight
        the adversarial and entropy terms, ``lr_d`` optionally overrides the
        discriminator learning rate.
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
        super().__init__(model, source_loaders, target_loaders, config, device)
        self.criterion = SegCrossEntropyLoss()
        self.entropy_loss = EntropyLoss()
        self.discriminator = DomainDiscriminator(in_channels=1).to(device)
        self.optimizer = self._make_optimizer(self.model.parameters())
        self.optimizer_d = self._make_optimizer(
            self.discriminator.parameters(), lr=config.lr_d
        )

    def train_epoch(self) -> dict[str, float]:
        """Runs a single ADVENT training epoch over the paired domains.

        Returns
        -------
        dict[str, float]
            Weighted-mean scalar metrics: ``loss_total``, ``loss_ce``,
            ``loss_adv``, ``loss_ent`` and ``epoch_time`` (seconds).
        """
        self.model.train()
        self.discriminator.train()
        epoch_start = time.time()

        src_iter, tgt_iter, num_batches = self._dual_iterators(
            self.source_loaders["train"], self.target_loaders["train"]
        )

        running_ce = 0.0
        running_adv = 0.0
        running_ent = 0.0

        pbar = tqdm(range(num_batches), desc="ADVENT Train")
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

            e_s = entropy_map(logits_s)
            e_t = entropy_map(logits_t)

            d_s = self.discriminator(e_s.detach())
            d_t = self.discriminator(e_t.detach())
            loss_dis, _ = adversarial_loss(d_s, d_t)
            self.scaler.scale(loss_dis).backward()
            self.scaler.step(self.optimizer_d)

            _, loss_adv = adversarial_loss(d_s.detach(), self.discriminator(e_t))
            loss_ce = self.criterion(logits_s, masks_s)
            loss_ent = self.entropy_loss(logits_t)
            total = (
                loss_ce
                + self.config.lambda_adv * loss_adv
                + self.config.lambda_entropy * loss_ent
            )
            self.scaler.scale(total).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_ce += float(loss_ce.item())
            running_adv += float(loss_adv.item())
            running_ent += float(loss_ent.item())
            pbar.set_postfix(loss=f"{total.item():.4f}")

        num = max(num_batches, 1)
        loss_ce = running_ce / num
        loss_adv = running_adv / num
        loss_ent = running_ent / num
        return {
            "loss_total": (
                loss_ce
                + self.config.lambda_adv * loss_adv
                + self.config.lambda_entropy * loss_ent
            ),
            "loss_ce": loss_ce,
            "loss_adv": loss_adv,
            "loss_ent": loss_ent,
            "epoch_time": time.time() - epoch_start,
        }
