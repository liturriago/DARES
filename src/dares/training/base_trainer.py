"""Shared trainer template for all DARES engines.

Every UDA method subclasses :class:`BaseTrainer`, which provides:

* AMP handling (``autocast`` + ``GradScaler``).
* Dual-domain iteration helpers (source / target paired batches).
* Pixel-level metric evaluation via :class:`dares.utils.metrics.MetricTracker`.
* Best-checkpoint tracking on the target validation mIoU.
* The ``fit`` epoch loop (train -> validate -> schedule -> checkpoint).
"""

import copy
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import cycle
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler

from dares.config import TrainConfig
from dares.utils.evaluation import evaluate_segmentation
from dares.utils.formatter import format_time


class BaseTrainer(ABC):
    """Abstract base class shared by all DARES training engines.

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
        model (nn.Module): The model being trained.
        num_classes (int): Number of output classes.
        class_names (list[str]): Human-readable class names.
        use_amp (bool): Whether AMP is enabled.
        scaler (GradScaler): Gradient scaler for AMP.
        best_miou (float): Best target-validation mIoU achieved.
        best_state (dict): State dict of the best model so far.
        history (dict[str, list[float]]): Per-epoch scalar history.
    """

    def __init__(
        self,
        model: nn.Module,
        source_loaders: dict[str, Any],
        target_loaders: dict[str, Any],
        config: TrainConfig,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.source_loaders = source_loaders
        self.target_loaders = target_loaders
        self.config = config
        self.device = device

        self.num_classes = int(model.head.num_classes)
        self.class_names = self._resolve_class_names()

        self.use_amp = bool(config.use_amp and device.type == "cuda")
        self.scaler = GradScaler(enabled=self.use_amp)

        self.best_miou: float = 0.0
        self.best_state = copy.deepcopy(model.state_dict())
        self.history: dict[str, list[float]] = defaultdict(list)

    def _resolve_class_names(self) -> list[str]:
        """Resolves the class names from the dataset when available."""
        for loader in (
            self.source_loaders.get("train"),
            self.target_loaders.get("train"),
        ):
            if loader is not None:
                classes = getattr(loader.dataset, "classes", None)
                if classes:
                    return list(classes)
        return ["non_forest", "forest"]

    def _make_optimizer(
        self, params: Any, lr: float | None = None
    ) -> optim.Adam:
        """Builds an Adam optimizer for the given parameters.

        Args:
            params (Any): Iterable of parameters or parameter groups.
            lr (float | None): Learning rate; defaults to ``config.lr``.

        Returns:
            optim.Adam: The configured Adam optimizer.
        """
        return optim.Adam(
            params,
            lr=self.config.lr if lr is None else lr,
            weight_decay=self.config.weight_decay,
        )

    def _backward_step(
        self,
        loss: torch.Tensor,
        optimizer: optim.Optimizer,
        params: Any | None = None,
    ) -> None:
        """Backpropagates, optionally clips, and steps an optimizer.

        Applies the AMP scaler to ``loss``, backpropagates, and (when
        ``config.grad_clip`` is set) unscales the gradients and clips their
        global norm before stepping. If any gradient is non-finite (NaN / inf)
        the step is skipped to avoid corrupting the weights. Clipping and the
        finiteness guard stabilize adversarial methods whose gradients can
        explode or NaN through the discriminator.

        Args:
            loss (torch.Tensor): Loss to backpropagate.
            optimizer (optim.Optimizer): Optimizer to step.
            params (Any | None): Parameters to check / clip; defaults to all
                optimizer parameters.
        """
        self.scaler.scale(loss).backward()
        if params is None:
            params = [
                p
                for group in optimizer.param_groups
                for p in group["params"]
            ]
        if not all(
            p.grad is None or torch.isfinite(p.grad).all() for p in params
        ):
            self.optimizer.zero_grad(set_to_none=True)
            return
        clip = self.config.grad_clip
        if clip is not None and clip > 0.0:
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, float(clip))
        self.scaler.step(optimizer)

    @staticmethod
    def _dual_iterators(
        source_loader: Any, target_loader: Any
    ) -> tuple[Any, Any, int]:
        """Pairs source and target iterators over the longer of the two.

        Args:
            source_loader (DataLoader): Source training loader.
            target_loader (DataLoader): Target training loader.

        Returns:
            tuple[Any, Any, int]: ``(source_iter, target_iter, num_batches)``;
                the shorter domain is cycled to match the longer one.
        """
        len_s, len_t = len(source_loader), len(target_loader)
        num_batches = max(len_s, len_t)
        src_iter = (
            iter(source_loader) if len_s >= len_t else cycle(source_loader)
        )
        tgt_iter = (
            iter(target_loader) if len_t >= len_s else cycle(target_loader)
        )
        return src_iter, tgt_iter, num_batches

    @abstractmethod
    def train_epoch(self) -> dict[str, float]:
        """Runs a single training iteration over the data.

        Returns:
            dict[str, float]: Scalar metrics for the epoch; MUST include the
                ``"epoch_time"`` key (seconds) and one key per loss component.
        """
        raise NotImplementedError

    @torch.no_grad()
    def evaluate(self, loader: Any, prefix: str = "Val") -> float:
        """Evaluates pixel-level metrics on a labeled loader.

        Args:
            loader (DataLoader): Labeled loader to evaluate.
            prefix (str): Label for the printed report.

        Returns:
            float: The mean IoU (mIoU) used for checkpoint selection.
        """
        metrics = evaluate_segmentation(
            self.model,
            loader,
            self.device,
            self.num_classes,
            self.class_names,
            use_amp=self.use_amp,
            prefix=prefix,
        )
        return float(metrics["mIoU"])

    def fit(
        self, scheduler: optim.lr_scheduler.LRScheduler | None = None
    ) -> nn.Module:
        """Executes the full training loop.

        For each epoch: run ``train_epoch``, evaluate on the source and target
        validation splits, step the scheduler, and keep the checkpoint with the
        best target-validation mIoU. Restores the best weights before returning.

        If ``config.warmup_epochs > 0`` the backbone is frozen (head-only
        warm-start) for the first ``warmup_epochs`` epochs and unfrozen
        afterwards; the DARES engine aligns its alignment-weight warm-up with
        the same period.

        Args:
            scheduler (LRScheduler | None): Optional learning-rate scheduler
                for the main optimizer.

        Returns:
            nn.Module: The model with the best target-validation weights.
        """
        total_start = time.time()
        warmup_epochs = int(self.config.warmup_epochs or 0)
        if warmup_epochs > 0:
            self.model.freeze_backbone()
            print(
                f"Warm-up enabled: backbone frozen for the first "
                f"{warmup_epochs} epoch(s) (head-only training)."
            )

        for epoch in range(self.config.epochs):
            if 0 < warmup_epochs == epoch:
                self.model.unfreeze_backbone()
                print(
                    f"Warm-up complete at epoch {epoch + 1}: backbone unfrozen."
                )

            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            print("-" * 40)

            train_metrics = self.train_epoch()
            epoch_time = train_metrics.pop("epoch_time", 0.0)
            loss_str = " | ".join(
                f"{k}: {v:.4f}"
                for k, v in train_metrics.items()
                if "loss" in k
            )
            print(f"[Train] Time: {format_time(epoch_time)} | {loss_str}")
            for key, value in train_metrics.items():
                self.history[key].append(value)

            src_miou = self.evaluate(
                self.source_loaders["validation"], "Source Val"
            )
            tgt_miou = self.evaluate(
                self.target_loaders["validation"], "Target Val"
            )

            if scheduler is not None:
                scheduler.step()

            if tgt_miou > self.best_miou:
                self.best_miou = tgt_miou
                self.best_state = copy.deepcopy(self.model.state_dict())
                print(f"New best model found! (Target Val mIoU: {self.best_miou:.4f})")

        total_time = time.time() - total_start
        print(f"\n{' TRAINING COMPLETE ':=^50}")
        print(f"Total Duration: {format_time(total_time)}")
        print(f"Best Target Val mIoU: {self.best_miou:.4f}")
        print("=" * 50)

        self.model.load_state_dict(self.best_state)
        return self.model

    def save_checkpoint(self, path: str) -> None:
        """Saves the best model weights and training state.

        Args:
            path (str): Destination checkpoint path.
        """
        torch.save(
            {
                "model": self.best_state,
                "best_miou": self.best_miou,
                "config": self.config.model_dump(),
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Loads a checkpoint saved by :meth:`save_checkpoint`.

        Args:
            path (str): Source checkpoint path.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.best_state = checkpoint["model"]
        self.best_miou = float(checkpoint.get("best_miou", 0.0))
        self.model.load_state_dict(self.best_state)
