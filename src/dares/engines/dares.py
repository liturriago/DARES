"""DARES training engine: source cross-entropy plus hardened DARES alignment.

The DARES objective combines the supervised segmentation loss on the source
domain with the class-conditional Renyi-2 alignment on unlabeled target
batches. This engine uses the hardened DARESLoss with anti-collapse floors,
inter-class target repulsion and a per-step GradNorm-lite trust region on the
reference parameters.
"""

import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm

from dares.config import TrainConfig
from dares.losses.dares_loss import DARESLoss
from dares.training.base_trainer import BaseTrainer

_METRIC_KEYS = (
    "loss_total",
    "loss_seg",
    "loss_align",
    "loss_anti_collapse",
    "loss_repulsion",
    "loss_em",
    "h2_source_mean",
    "h2_target_mean",
    "delta_align_mean",
    "delta_repulsion_mean",
    "lambda_eff",
    "n_valid_classes",
    "n_rep_pairs",
)


class DARESTrainer(BaseTrainer):
    """Unsupervised domain adaptation via hardened class-conditional alignment."""

    def __init__(
        self,
        model: nn.Module,
        source_loaders: dict[str, Any],
        target_loaders: dict[str, Any],
        config: TrainConfig,
        device: torch.device,
    ) -> None:
        super().__init__(model, source_loaders, target_loaders, config, device)

        # Compatibilidad robusta para nombres de variables entre configs
        gamma_val = getattr(config, "repulsion_gamma", 0.5)

        self.criterion = DARESLoss(
            num_classes=self.num_classes,
            quota=getattr(config, "quota", 256),
            min_samples=getattr(config, "min_samples", 8),
            lambda_max=getattr(config, "lambda_max", 1.0),
            lambda_align=getattr(config, "lambda_align", 1.0),
            beta=getattr(config, "beta", 1.0),
            gamma=gamma_val,
            eta_floor=getattr(config, "eta_floor", 1.0),
            entropy_gap=getattr(config, "entropy_gap", 0.25),
            repulsion_margin=getattr(config, "repulsion_margin", 0.2),
            warmup_steps=getattr(config, "warmup_steps", 1000),
            ramp_steps=getattr(config, "ramp_steps", 4000),
            ramp_delta=getattr(config, "ramp_delta", 10.0),
            grad_ratio=getattr(config, "grad_ratio", 0.8),
            trust_region=getattr(config, "trust_region", True),
            ema_decay=getattr(config, "ema_decay", 0.9),
            use_renyi_em=getattr(config, "use_renyi_em", True),
            lambda_em=getattr(config, "lambda_em", 0.05),
            em_pool=getattr(config, "em_pool", False),
            em_pool_kernel=getattr(config, "em_pool_kernel", 3),
        ).to(device)

        self.ref_params = list(self.model.backbone.reference_params)
        self.optimizer = self._make_optimizer(self.model.parameters())

    def train_epoch(self) -> dict[str, float]:
        """Runs one epoch of DARES training."""
        self.model.train()
        if self.criterion.in_warmup():
            return self._train_warmup_epoch()
        return self._train_dual_epoch()

    def _train_warmup_epoch(self) -> dict[str, float]:
        """Source-only warm-up epoch calibrating EMA gradient statistics."""
        acc: dict[str, float] = {k: 0.0 for k in _METRIC_KEYS}
        start_time = time.time()
        n = 0

        pbar = tqdm(
            self.source_loaders["train"],
            desc=f"Train ({self.config.method}) [warmup]",
            leave=False,
        )
        for batch in pbar:
            imgs, masks = batch[0].to(self.device), batch[1].to(self.device)
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(imgs, mode="class")
                loss_seg = F.cross_entropy(logits, masks, ignore_index=255)

            # Actualiza el EMA de gradiente supervisado desde el paso 1
            self.criterion.update_lambda(loss_seg, None, self.ref_params)

            stepped = super()._backward_step(loss_seg, self.optimizer)
            if stepped:
                self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            pbar.set_postfix(loss=f"{loss_seg.item():.4f}")

            acc["loss_total"] += float(loss_seg.detach().cpu().item())
            acc["loss_seg"] += float(loss_seg.detach().cpu().item())
            n += 1

        epoch_time = time.time() - start_time
        denom = max(n, 1)
        out = {k: v / denom for k, v in acc.items()}
        out["epoch_time"] = epoch_time
        return out

    def _train_dual_epoch(self) -> dict[str, float]:
        """Paired source/target epoch with synchronized DARES alignment."""
        src_iter, tgt_iter, num_batches = self._dual_iterators(
            self.source_loaders["train"], self.target_loaders["train"]
        )

        acc: dict[str, float] = {k: 0.0 for k in _METRIC_KEYS}
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

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                feats_s, logits_s = self.model(imgs_s, mode="deep")
                feats_t, logits_t = self.model(imgs_t, mode="deep")
                _, parts = self.criterion(feats_s, logits_s, masks_s, feats_t, logits_t)

            # 1. Actualiza lambda_eff con base en los gradientes del lote actual
            lam_eff = self.criterion.update_lambda(
                parts["loss_seg"], parts["loss_aux"], self.ref_params
            )

            # 2. Reconstruye el loss total exacto con el lambda_eff recién calculado
            total = (
                parts["loss_seg"]
                + float(lam_eff) * parts["loss_aux"]
                + self.criterion.lambda_em * parts["loss_em"]
            )

            # 3. Paso de optimización
            stepped = super()._backward_step(total, self.optimizer)
            if stepped:
                self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            pbar.set_postfix(loss=f"{total.item():.4f}", lam=f"{lam_eff:.3f}")

            acc["loss_total"] += float(total.detach().cpu().item())
            acc["loss_seg"] += float(parts["loss_seg"].detach().cpu().item())
            acc["loss_align"] += float(parts["loss_align"].detach().cpu().item())
            acc["loss_anti_collapse"] += float(
                parts["loss_anti_collapse"].detach().cpu().item()
            )
            rep = parts["loss_repulsion"].detach().cpu().item()
            acc["loss_repulsion"] += 0.0 if rep != rep else rep
            acc["loss_em"] += float(parts["loss_em"].detach().cpu().item())
            acc["h2_source_mean"] += float(
                parts["h2_source_mean"].detach().cpu().item()
            )
            acc["h2_target_mean"] += float(
                parts["h2_target_mean"].detach().cpu().item()
            )
            acc["delta_align_mean"] += float(
                parts["delta_align_mean"].detach().cpu().item()
            )
            drep = parts["delta_repulsion_mean"].detach().cpu().item()
            acc["delta_repulsion_mean"] += 0.0 if drep != drep else drep
            acc["lambda_eff"] += float(lam_eff)
            acc["n_valid_classes"] += float(
                parts["n_valid_classes"].detach().cpu().item()
            )
            acc["n_rep_pairs"] += float(parts["n_rep_pairs"].detach().cpu().item())

        epoch_time = time.time() - start_time
        denom = max(num_batches, 1)
        out = {k: v / denom for k, v in acc.items()}
        out["epoch_time"] = epoch_time
        return out
