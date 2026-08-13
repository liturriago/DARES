"""DARES training engine: source cross-entropy plus alpha-Renyi alignment.

The DARES objective combines the supervised segmentation loss on the source
domain with the class-conditional Renyi mutual-information alignment on
unlabeled target batches:

``L_DARES = L_CE(D_s) - lambda * sum_c I2tilde(Ks_c; Ktilde_t_c)``

The alignment term is *subtracted* because the mutual-information estimator is
maximized during training (Eq. 5 of the paper).
"""

import time
from typing import Any

import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm import tqdm

from dares.config import TrainConfig
from dares.losses.ce import SegCrossEntropyLoss
from dares.losses.renyi import RenyiLoss
from dares.training.base_trainer import BaseTrainer


class DARESTrainer(BaseTrainer):
    """Unsupervised domain adaptation via class-conditional Renyi alignment.

    Parameters
    ----------
    model : nn.Module
        The segmentation model (a ``dares.models`` ``SegmentationModel``).
    source_loaders : dict[str, DataLoader]
        ``{"train", "validation", "test"}`` labeled source loaders.
    target_loaders : dict[str, DataLoader]
        ``{"train", "validation", "test"}`` target loaders; ``train`` is
        unlabeled and only its pseudo-labels are used for alignment.
    config : TrainConfig
        Training configuration.
    device : torch.device
        Computing device.

    Attributes
    ----------
    criterion : SegCrossEntropyLoss
        Pixel-wise cross-entropy over the labeled source batches.
    renyi_loss : RenyiLoss
        Class-conditional alpha-Renyi alignment over the target batches.
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
        """Initializes the trainer, the losses and the optimizer."""
        super().__init__(model, source_loaders, target_loaders, config, device)
        self.criterion = SegCrossEntropyLoss()
        self.renyi_loss = RenyiLoss(
            self.num_classes,
            tau=config.tau,
            n_max=config.n_max,
            sigma=config.sigma,
            alpha=config.alpha,
        )
        self.optimizer = self._make_optimizer(self.model.parameters())
        self._epoch_idx = 0

    def train_epoch(self) -> dict[str, float]:
        """Runs one epoch of DARES training.

        Pairs source and target batches with ``_dual_iterators`` and for each
        pair computes the source cross-entropy plus the class-conditional
        Renyi alignment. The mutual-information term is maximized by
        minimizing ``L_CE - lambda * I2tilde``; during the warmup phase
        (``epoch_idx < warmup_epochs``) the alignment weight is ``0.0``.

        Returns
        -------
        dict[str, float]
            ``{"loss_total", "loss_ce", "loss_renyi", "lambda_active",
            "valid_classes", "epoch_time"}``; the loss keys are epoch averages
            and ``epoch_time`` is the wall-clock duration in seconds.
        """
        self.model.train()
        src_iter, tgt_iter, num_batches = self._dual_iterators(
            self.source_loaders["train"], self.target_loaders["train"]
        )

        warmup = (
            self.config.warmup_epochs is not None
            and self._epoch_idx < self.config.warmup_epochs
        )
        lambda_active = 0.0 if warmup else float(self.config.lambda_renyi)

        total_loss = 0.0
        ce_loss = 0.0
        renyi_loss = 0.0
        valid_classes = 0
        start_time = time.time()

        for _ in tqdm(range(num_batches), desc="DARES train", leave=False):
            imgs_s, masks_s = next(src_iter)
            imgs_t, _ = next(tgt_iter)
            imgs_s = imgs_s.to(self.device)
            masks_s = masks_s.to(self.device)
            imgs_t = imgs_t.to(self.device)

            with autocast(
                device_type=self.device.type, enabled=self.use_amp
            ):
                feats_s, logits_s = self.model(imgs_s, mode="both")
                feats_t, logits_t = self.model(imgs_t, mode="both")
                loss_ce = self.criterion(logits_s, masks_s)

            # The Renyi alignment is computed outside autocast (and internally
            # in float32): the Gram-matrix trace normalizations are precision
            # sensitive and must not run in float16.
            with autocast(device_type=self.device.type, enabled=False):
                alignment, rmetrics = self.renyi_loss(
                    feats_s, masks_s, feats_t, logits_t
                )
            total = loss_ce - lambda_active * alignment

            self.scaler.scale(total).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            total_loss += float(total.detach().cpu().item())
            ce_loss += float(loss_ce.detach().cpu().item())
            renyi_loss += float(alignment.detach().cpu().item())
            valid_classes += int(rmetrics["valid_classes"])

        self._epoch_idx += 1
        epoch_time = time.time() - start_time

        num_batches = max(num_batches, 1)
        return {
            "loss_total": float(total_loss / num_batches),
            "loss_ce": float(ce_loss / num_batches),
            "loss_renyi": float(renyi_loss / num_batches),
            "lambda_active": float(lambda_active),
            "valid_classes": float(valid_classes / num_batches),
            "epoch_time": float(epoch_time),
        }
