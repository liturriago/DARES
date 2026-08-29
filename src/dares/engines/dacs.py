"""DACS training engine: cross-domain mixed sampling for UDA segmentation.

DACS (Tranheden et al., WACV 2021) trains the segmenter on the labeled source
domain and on class-mixed source/target samples with pseudo-labels, using an
adaptive loss weight that tracks the proportion of confident target pixels.

``L = CE(x_s, y_s) + lambda * CE(x_mix, y_mix)``

where ``lambda`` is the mean fraction of confident target pixels (DACS Eq. 1
with the adaptive schedule of Section 3.3).
"""

import time
from typing import Any

import torch
import torch.nn as nn
from torch.amp import autocast
from tqdm import tqdm

from dares.config import TrainConfig
from dares.losses.ce import SegCrossEntropyLoss
from dares.losses.dacs import class_mix, color_jitter, gaussian_blur, pseudo_label
from dares.training.base_trainer import BaseTrainer


class DACSTrainer(BaseTrainer):
    """Cross-domain mixed sampling self-training engine.

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
        Training configuration (DACS hyperparameters under ``dacs_*``).
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

        self.threshold = float(config.dacs_threshold)
        self.mix_ratio = float(config.dacs_mix_ratio)
        self.use_color_jitter = bool(config.dacs_color_jitter)
        self.brightness = float(config.dacs_brightness)
        self.contrast = float(config.dacs_contrast)
        self.saturation = float(config.dacs_saturation)
        self.hue = float(config.dacs_hue)
        self.use_blur = bool(config.dacs_blur)
        self.blur_kernel = int(config.dacs_blur_kernel)
        self.blur_sigma = tuple(config.dacs_blur_sigma)

    def train_epoch(self) -> dict[str, float]:
        """Runs one epoch of DACS cross-domain mixed training.

        Returns
        -------
        dict[str, float]
            ``{"loss_total", "loss_ce", "loss_mix", "lambda_unsup",
            "epoch_time"}``; losses are per-iteration averages and
            ``epoch_time`` is the wall-clock duration in seconds.
        """
        self.model.train()
        src_iter, tgt_iter, num_batches = self._dual_iterators(
            self.source_loaders["train"], self.target_loaders["train"]
        )

        total_loss = 0.0
        ce_loss = 0.0
        mix_loss = 0.0
        lambda_avg = 0.0
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

            # Strong augmentations (fp32, outside autocast).
            if self.use_color_jitter:
                imgs_s = color_jitter(
                    imgs_s, self.brightness, self.contrast, self.saturation, self.hue
                )
                imgs_t = color_jitter(
                    imgs_t, self.brightness, self.contrast, self.saturation, self.hue
                )
            if self.use_blur:
                imgs_t = gaussian_blur(imgs_t, self.blur_kernel, self.blur_sigma)

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                logits_s = self.model(imgs_s, mode="class")
                logits_t = self.model(imgs_t, mode="class")

            # Pseudo-label the (augmented) target and cross-domain mix.
            pseudo_t, confident = pseudo_label(logits_t.detach(), self.threshold)
            mixed_img, mixed_label = class_mix(
                imgs_s, masks_s, imgs_t, pseudo_t, self.mix_ratio
            )

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                logits_mix = self.model(mixed_img, mode="class")
                loss_ce = self.criterion(logits_s, masks_s)
                loss_mix = self.criterion(logits_mix, mixed_label)
                lambda_unsup = confident.float().mean()
                total = loss_ce + lambda_unsup * loss_mix

            stepped = super()._backward_step(total, self.optimizer)
            if stepped:
                self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            pbar.set_postfix(loss=f"{total.item():.4f}")

            total_loss += float(total.detach().cpu().item())
            ce_loss += float(loss_ce.detach().cpu().item())
            mix_loss += float(loss_mix.detach().cpu().item())
            lambda_avg += float(lambda_unsup.detach().cpu().item())

        denom = max(num_batches, 1)
        return {
            "loss_total": total_loss / denom,
            "loss_ce": ce_loss / denom,
            "loss_mix": mix_loss / denom,
            "lambda_unsup": lambda_avg / denom,
            "epoch_time": time.time() - start_time,
        }
