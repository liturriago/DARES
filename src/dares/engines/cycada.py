"""CyCADA training engine (Cycle-Consistent Adversarial Domain Adaptation).

Minimal-but-faithful implementation of CyCADA (Hoffman et al., 2018) for UDA
semantic segmentation. Combines image-level cycle-consistent translation
(source <-> target) with a task loss on the translated source and feature-level
adversarial alignment:

* ``g_st`` / ``g_ts``  ->  small pixel generators translating source <-> target.
* ``d_pix``            ->  PatchGAN image discriminator on translated images.
* ``d_feat``           ->  feature-level domain discriminator (shared with ADVENT).

Subclasses :class:`dares.training.base_trainer.BaseTrainer`; the ``fit`` loop,
AMP handling, dual-domain batching and pixel-metric evaluation are inherited.
"""

import time
from typing import Any

import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm import tqdm

from dares.config import TrainConfig
from dares.losses.ce import SegCrossEntropyLoss
from dares.losses.cycada import (
    PatchDiscriminator,
    PixelGenerator,
    cycle_consistency_loss,
    identity_loss,
    patch_adversarial_loss,
    patch_discriminator_loss,
)
from dares.losses.domain import DomainDiscriminator, adversarial_loss
from dares.training.base_trainer import BaseTrainer


class CyCADATrainer(BaseTrainer):
    """Cycle-consistent adversarial domain adaptation training engine.

    Args:
        model (nn.Module): The segmentation model (a ``dares.models``
            ``SegmentationModel``).
        source_loaders (dict[str, DataLoader]): ``{"train", "validation",
            "test"}`` labeled source loaders.
        target_loaders (dict[str, DataLoader]): ``{"train", "validation",
            "test"}`` target loaders; ``train`` is unlabeled.
        config (TrainConfig): Training configuration.
        device (torch.device): Computing device.

    Attributes:
        g_st (PixelGenerator): Source -> target pixel generator.
        g_ts (PixelGenerator): Target -> source pixel generator.
        d_pix (PatchDiscriminator): PatchGAN image discriminator.
        d_feat (DomainDiscriminator): Feature-level domain discriminator.
        criterion (SegCrossEntropyLoss): Pixel-wise cross-entropy task loss.
        optimizer (optim.Adam): Optimizer for the segmentation model.
        optimizer_gen (optim.Adam): Optimizer for both pixel generators.
        optimizer_dpix (optim.Adam): Optimizer for the PatchGAN discriminator.
        optimizer_dfeat (optim.Adam): Optimizer for the feature discriminator.
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

        self.g_st = PixelGenerator(in_channels=4).to(device)
        self.g_ts = PixelGenerator(in_channels=4).to(device)
        self.d_pix = PatchDiscriminator(in_channels=4).to(device)
        self.d_feat = DomainDiscriminator(in_channels=self.model.feature_dim).to(
            device
        )
        self.criterion = SegCrossEntropyLoss()

        self.optimizer = self._make_optimizer(self.model.parameters())
        self.optimizer_gen = self._make_optimizer(
            list(self.g_st.parameters()) + list(self.g_ts.parameters()),
            lr=config.lr_g,
        )
        self.optimizer_dpix = self._make_optimizer(
            self.d_pix.parameters(), lr=config.lr_d
        )
        self.optimizer_dfeat = self._make_optimizer(
            self.d_feat.parameters(), lr=config.lr_d
        )
        self._epoch_idx = 0

    def train_epoch(self) -> dict[str, float]:
        """Runs a single CyCADA training epoch.

        Per iteration: a joint generator + segmenter step (task loss on the
        original source AND the translated source, cycle / identity / pixel-
        adversarial terms, plus feature-adversarial once the warm-up ends),
        followed by PatchGAN and feature discriminator steps. The scaler is
        updated once after all four optimizers.

        Returns:
            dict[str, float]: Averaged scalar metrics for the epoch including
                ``"epoch_time"`` (seconds) plus the loss components
                ``"loss_total"``, ``"loss_task"``, ``"loss_cycle"``,
                ``"loss_identity"``, ``"loss_pix_adv"`` and ``"loss_feat_adv"``.
        """
        self.model.train()
        self.g_st.train()
        self.g_ts.train()
        self.d_pix.train()
        self.d_feat.train()

        src_iter, tgt_iter, num_batches = self._dual_iterators(
            self.source_loaders["train"], self.target_loaders["train"]
        )

        sums: dict[str, float] = {
            "loss_total": 0.0,
            "loss_task": 0.0,
            "loss_cycle": 0.0,
            "loss_identity": 0.0,
            "loss_pix_adv": 0.0,
            "loss_feat_adv": 0.0,
        }
        epoch_start = time.time()

        pbar = tqdm(
            range(num_batches),
            desc=f"Train ({self.config.method})",
            leave=False,
        )
        for _ in pbar:
            self.optimizer.zero_grad()
            self.optimizer_gen.zero_grad()
            self.optimizer_dpix.zero_grad()
            self.optimizer_dfeat.zero_grad()

            imgs_s, masks_s = next(src_iter)
            imgs_t, _ = next(tgt_iter)
            imgs_s = imgs_s.to(self.device)
            imgs_t = imgs_t.to(self.device)
            masks_s = masks_s.to(self.device) if masks_s is not None else None

            # --- Generator + segmenter step ---------------------------------
            # Feature alignment is disabled during the warm-up (backbone
            # frozen, model not yet segmenting): aligning its features only
            # pushes it to a trivial domain-invariant solution.
            feat_lambda = (
                0.0
                if (
                    self.config.warmup_epochs is not None
                    and self._epoch_idx < self.config.warmup_epochs
                )
                else self.config.lambda_feat
            )
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                x_st = self.g_st(imgs_s)
                x_ts = self.g_ts(imgs_t)

                loss_cycle = cycle_consistency_loss(
                    self.g_st, self.g_ts, imgs_s, imgs_t
                )
                loss_idt = identity_loss(self.g_st, self.g_ts, imgs_s, imgs_t)

                # Task loss anchored on BOTH the original and the translated
                # source, so the segmenter learns from real source imagery and
                # the generator cannot "hack" the task.
                feats_s, logits_s = self.model(imgs_s, mode="both")
                logits_st = self.model(x_st, mode="class")
                loss_task = self.criterion(
                    logits_s, masks_s
                ) + self.criterion(logits_st, masks_s)

                loss_pix_adv = patch_adversarial_loss(
                    self.d_pix(x_st)
                ) + patch_adversarial_loss(self.d_pix(x_ts))

                feats_t, _ = self.model(imgs_t, mode="both")
                _, loss_feat_adv = adversarial_loss(
                    self.d_feat(feats_s), self.d_feat(feats_t)
                )

                total = (
                    loss_task
                    + self.config.lambda_cycle * loss_cycle
                    + self.config.lambda_identity * loss_idt
                    + self.config.lambda_pixel * loss_pix_adv
                    + feat_lambda * loss_feat_adv
                )

            self.scaler.scale(total).backward()
            self.scaler.step(self.optimizer)
            self.scaler.step(self.optimizer_gen)

            # --- PatchGAN discriminator step ---------------------------------
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                d_real = self.d_pix(torch.cat([imgs_s, imgs_t], dim=0))
                d_fake = self.d_pix(
                    torch.cat([x_st, x_ts], dim=0).detach()
                )
                loss_dpix = patch_discriminator_loss(d_real, d_fake)
            self.scaler.scale(loss_dpix).backward()
            self.scaler.step(self.optimizer_dpix)

            # --- Feature discriminator step ----------------------------------
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                loss_dfeat, _ = adversarial_loss(
                    self.d_feat(feats_s.detach()),
                    self.d_feat(feats_t.detach()),
                )
            self.scaler.scale(loss_dfeat).backward()
            self.scaler.step(self.optimizer_dfeat)

            self.scaler.update()

            sums["loss_total"] += float(total.detach())
            sums["loss_task"] += float(loss_task.detach())
            sums["loss_cycle"] += float(loss_cycle.detach())
            sums["loss_identity"] += float(loss_idt.detach())
            sums["loss_pix_adv"] += float(loss_pix_adv.detach())
            sums["loss_feat_adv"] += float(loss_feat_adv.detach())
            pbar.set_postfix(loss=f"{total.item():.4f}")

        metrics = {key: value / num_batches for key, value in sums.items()}
        metrics["epoch_time"] = time.time() - epoch_start
        self._epoch_idx += 1
        return metrics
