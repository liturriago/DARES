"""CBST trainer engine: class-balanced self-training for UDA segmentation.

Implements CBST (Zou et al., 2018). Each iteration the model is trained with a
supervised pixel-wise cross-entropy on the labeled source domain plus a masked
cross-entropy on the unlabeled target domain. Target pseudo-labels are
refreshed every iteration with a class-balanced top-ratio selection policy, and
for ``n_self_training_rounds > 1`` the pseudo-labels are recomputed and an
extra optimization step is taken each round.
"""

import time
from typing import Any

import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm import tqdm

from dares.config import TrainConfig
from dares.losses.cbst import CBSTPseudoLabeling, CBSTSelfTrainingLoss
from dares.losses.ce import SegCrossEntropyLoss
from dares.training.base_trainer import BaseTrainer


class CBSTTrainer(BaseTrainer):
    """Class-balanced self-training engine.

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
        Training configuration (CBST hyperparameters under ``lambda_self``,
        ``pseudo_threshold``, ``pseudo_topk_ratio`` and
        ``n_self_training_rounds``).
    device : torch.device
        Computing device.

    Attributes
    ----------
    criterion : SegCrossEntropyLoss
        Pixel-wise cross-entropy loss over the labeled source batches.
    pseudo_labeler : CBSTPseudoLabeling
        Class-balanced pseudo-label / weight generator for the target domain.
    self_loss : CBSTSelfTrainingLoss
        Masked self-training loss over the target pseudo-labels.
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
        """Initializes the losses, the pseudo-labeler and the optimizer."""
        super().__init__(model, source_loaders, target_loaders, config, device)
        self.criterion = SegCrossEntropyLoss()
        self.pseudo_labeler = CBSTPseudoLabeling(
            self.num_classes,
            topk_ratio=config.pseudo_topk_ratio,
            threshold=config.pseudo_threshold,
        )
        self.self_loss = CBSTSelfTrainingLoss()
        self.optimizer = self._make_optimizer(self.model.parameters())

    def train_epoch(self) -> dict[str, float]:
        """Runs one epoch of class-balanced self-training.

        Pairs source and target batches, computes the supervised source loss
        and the masked self-training loss under one AMP context, and performs
        a GradScaler-managed optimizer step. Pseudo-labels are recomputed
        every iteration and, when ``n_self_training_rounds > 1``, again for
        each additional self-training round (with an extra optimizer step).

        Returns
        -------
        dict[str, float]
            ``{"loss_total", "loss_ce", "loss_self", "epoch_time"}``; the
            losses are per-iteration averages and ``epoch_time`` is the
            wall-clock duration of the epoch in seconds.
        """
        self.model.train()
        src_iter, tgt_iter, num_batches = self._dual_iterators(
            self.source_loaders["train"], self.target_loaders["train"]
        )
        rounds = int(self.config.n_self_training_rounds)
        total_loss = 0.0
        ce_loss = 0.0
        self_loss = 0.0
        start_time = time.time()

        pbar = tqdm(
            range(num_batches),
            desc=f"Train ({self.config.method})",
            leave=False,
        )
        for step in pbar:
            imgs_s, masks_s = next(src_iter)
            imgs_t, _ = next(tgt_iter)
            imgs_s = imgs_s.to(self.device)
            masks_s = masks_s.to(self.device)
            imgs_t = imgs_t.to(self.device)

            with autocast(
                device_type=self.device.type, enabled=self.use_amp
            ):
                logits_s = self.model(imgs_s, mode="class")
                logits_t = self.model(imgs_t, mode="class")
                loss_ce = self.criterion(logits_s, masks_s)
                with torch.no_grad():
                    pseudo_t, weights_t = self.pseudo_labeler(
                        logits_t.detach()
                    )
                loss_self = self.self_loss(logits_t, pseudo_t, weights_t)
                loss_total = loss_ce + self.config.lambda_self * loss_self

            stepped = super()._backward_step(loss_total, self.optimizer)
            if stepped:
                self.scaler.update()

            total_loss += float(loss_total.detach().cpu().item())
            ce_loss += float(loss_ce.detach().cpu().item())
            self_loss += float(loss_self.detach().cpu().item())

            for _ in range(rounds - 1):
                with autocast(
                    device_type=self.device.type, enabled=self.use_amp
                ):
                    with torch.no_grad():
                        pseudo_t, weights_t = self.pseudo_labeler(
                            logits_t.detach()
                        )
                    loss_self = self.self_loss(
                        logits_t, pseudo_t, weights_t
                    )
                round_step = super()._backward_step(
                    loss_self, self.optimizer
                )
                if round_step:
                    self.scaler.update()

            self.optimizer.zero_grad(set_to_none=True)
            pbar.set_postfix(loss=f"{loss_total.item():.4f}")

        epoch_time = time.time() - start_time
        denom = max(num_batches, 1)
        return {
            "loss_total": float(total_loss / denom),
            "loss_ce": float(ce_loss / denom),
            "loss_self": float(self_loss / denom),
            "epoch_time": float(epoch_time),
        }