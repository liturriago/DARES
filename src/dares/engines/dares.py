"""DARES training engine: source cross-entropy plus hardened DARES alignment.

The DARES objective combines the supervised segmentation loss on the source
domain with the class-conditional Renyi-2 alignment on unlabeled target
batches. This engine uses the hardened ``DARESLoss`` (ported from the
reference module in ``Docs/``), which adds anti-collapse entropy floors,
inter-class target repulsion and a per-step GradNorm-lite trust region on the
reference (deepest encoder block) parameters.

``L = L_seg + lambda_eff * (L_align + beta * L_ac + gamma * L_rep)``

where ``lambda_eff`` is scheduled per-step by :meth:`DARESLoss.update_lambda`
(warm-up -> sigmoid ramp -> gradient-ratio cap) instead of by epoch.
"""

import copy
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm

from dares.config import TrainConfig
from dares.losses.ce import SegCrossEntropyLoss
from dares.losses.dacs import class_mix
from dares.losses.dares_loss import DARESLoss
from dares.losses.fda import fourier_domain_adaptation
from dares.training.base_trainer import BaseTrainer


class DARESTrainer(BaseTrainer):
    """Unsupervised domain adaptation via hardened class-conditional alignment.

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
    criterion : DARESLoss
        The hardened class-conditional Renyi-2 alignment loss (DARES).
    ref_params : list[torch.Tensor]
        Reference parameters for the trust-region gradient balancing (the
        deepest shared encoder block).
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
        """Initializes the DARES criterion, reference params and optimizer."""
        super().__init__(model, source_loaders, target_loaders, config, device)
        self.criterion = DARESLoss(
            num_classes=self.num_classes,
            quota=config.quota,
            min_samples=config.min_samples,
            lambda_max=config.lambda_max,
            beta=config.beta,
            gamma=config.repulsion_gamma,
            eta_floor=config.eta_floor,
            entropy_gap=config.entropy_gap,
            repulsion_margin=config.repulsion_margin,
            warmup_steps=config.warmup_steps,
            ramp_steps=config.ramp_steps,
            ramp_delta=config.ramp_delta,
            grad_ratio=config.grad_ratio,
            trust_region=config.trust_region,
            ema_decay=config.ema_decay,
            two_sided_gap=config.two_sided_gap,
            pl_threshold=config.pl_threshold,
        ).to(device)
        self.ref_params = list(self.model.backbone.reference_params)
        self.optimizer = self._make_optimizer(self.model.parameters())

        # Dense target supervision (confidence-thresholded pseudo-labels from an
        # EMA teacher, optionally ClassMix-mixed against the source).
        self.use_self_training = bool(config.self_training)
        self.lambda_pl = float(config.lambda_pl)
        self.pl_threshold = float(config.pl_threshold)
        self.teacher_decay = float(config.teacher_ema_decay)
        self.use_classmix = bool(config.use_classmix)
        self.classmix_mix_ratio = float(config.classmix_mix_ratio)
        self.criterion_pl = SegCrossEntropyLoss()

        # FDA-style input-space adaptation (low-frequency amplitude swap).
        self.fda_enable = bool(config.fda_enable)
        self.fda_beta = float(config.fda_beta)

        # EMA teacher producing stable pseudo-labels.
        self.teacher = copy.deepcopy(self.model).to(device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _update_teacher(self) -> None:
        """Moves the EMA teacher towards the current student weights."""
        d = self.teacher_decay
        for tp, sp in zip(self.teacher.parameters(), self.model.parameters()):
            tp.data.mul_(d).add_(sp.data, alpha=1.0 - d)
        for tb, sb in zip(self.teacher.buffers(), self.model.buffers()):
            # Only EMA the floating-point buffers (BN running stats); skip
            # integer bookkeeping buffers such as num_batches_tracked.
            if tb.dtype.is_floating_point:
                tb.data.mul_(d).add_(sb.data, alpha=1.0 - d)

    @torch.no_grad()
    def _pseudo_labels(
        self, imgs_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Thresholded pseudo-labels from the EMA teacher on the target batch.

        Returns ``(pseudo, conf)`` where ``pseudo`` is ``(B, H, W)`` int64 with
        ``255`` at low-confidence pixels and ``conf`` is the per-pixel top-1
        probability ``(B, H, W)``.
        """
        with autocast(device_type=self.device.type, enabled=self.use_amp):
            t_logits = self.teacher(imgs_t, mode="class")
        prob = torch.softmax(t_logits.float(), dim=1)
        conf, pseudo = prob.max(dim=1)
        keep = conf > self.pl_threshold
        pseudo = torch.where(keep, pseudo, torch.full_like(pseudo, 255))
        return pseudo, conf

    def train_epoch(self) -> dict[str, float]:
        """Runs one epoch of DARES training.

        Pairs source and target batches with ``_dual_iterators``. For each pair
        computes the DARES alignment over the deepest encoder features and
        calls :meth:`update_lambda` (after forward, before backward) so the
        alignment weight follows the per-step trust region.

        Returns
        -------
        dict[str, float]
            ``{"loss_total", "loss_seg", "loss_align", "loss_anti_collapse",
            "loss_repulsion", "loss_pl", "loss_mix", "h2_source_mean",
            "h2_target_mean", "delta_align_mean", "delta_repulsion_mean",
            "lambda_eff", "n_valid_classes", "n_rep_pairs", "pseudo_conf",
            "epoch_time"}``; the loss / diagnostic keys are epoch averages and
            ``epoch_time`` is the wall-clock duration in seconds.
        """
        self.model.train()
        src_iter, tgt_iter, num_batches = self._dual_iterators(
            self.source_loaders["train"], self.target_loaders["train"]
        )

        acc: dict[str, float] = {
            "loss_total": 0.0,
            "loss_seg": 0.0,
            "loss_align": 0.0,
            "loss_anti_collapse": 0.0,
            "loss_repulsion": 0.0,
            "loss_pl": 0.0,
            "loss_mix": 0.0,
            "h2_source_mean": 0.0,
            "h2_target_mean": 0.0,
            "delta_align_mean": 0.0,
            "delta_repulsion_mean": 0.0,
            "lambda_eff": 0.0,
            "n_valid_classes": 0.0,
            "n_rep_pairs": 0.0,
            "pseudo_conf": 0.0,
        }
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

            with autocast(
                device_type=self.device.type, enabled=self.use_amp
            ):
                # Deep (bottleneck) features for alignment, full-res logits for
                # the supervised loss and pseudo-label confidence weighting.
                imgs_s_fwd = imgs_s
                if self.fda_enable:
                    imgs_s_fwd = fourier_domain_adaptation(
                        imgs_s, imgs_t, self.fda_beta
                    )
                feats_s, logits_s = self.model(imgs_s_fwd, mode="deep")
                feats_t, logits_t = self.model(imgs_t, mode="deep")
                total, parts = self.criterion(
                    feats_s, logits_s, masks_s, feats_t, logits_t
                )

                # Dense target supervision: thresholded pseudo-labels from the
                # EMA teacher (pixel CE) plus optional ClassMix.
                loss_pl = parts["loss_seg"].new_zeros(())
                loss_mix = parts["loss_seg"].new_zeros(())
                pseudo_conf = parts["loss_seg"].new_zeros(())
                if self.use_self_training:
                    pseudo, conf = self._pseudo_labels(imgs_t)
                    loss_pl = self.criterion_pl(logits_t, pseudo)
                    pseudo_conf = conf.mean()
                    if self.use_classmix:
                        mixed_img, mixed_label = class_mix(
                            imgs_s, masks_s, imgs_t, pseudo,
                            self.classmix_mix_ratio,
                        )
                        with autocast(
                            device_type=self.device.type, enabled=self.use_amp
                        ):
                            logits_mix = self.model(mixed_img, mode="class")
                        loss_mix = self.criterion_pl(logits_mix, mixed_label)
                total = total + self.lambda_pl * (loss_pl + loss_mix)

            # Trust-region lambda update: strictly after forward, before
            # backward (keeps the graph via retain_graph=True).
            self.criterion.update_lambda(
                parts["loss_seg"], parts["loss_aux"], self.ref_params
            )

            # Backprop, AMP guard / clip and optimizer step (base trainer).
            stepped = super()._backward_step(total, self.optimizer)
            # Finalize the AMP scaler cycle only after a real step: update()
            # asserts that an inf check was recorded, which step() does.
            if stepped:
                self.scaler.update()
                self._update_teacher()
            self.optimizer.zero_grad(set_to_none=True)
            pbar.set_postfix(loss=f"{total.item():.4f}")

            acc["loss_total"] += float(total.detach().cpu().item())
            acc["loss_seg"] += float(parts["loss_seg"].detach().cpu().item())
            acc["loss_align"] += float(parts["loss_align"].detach().cpu().item())
            acc["loss_anti_collapse"] += float(parts["loss_anti_collapse"].detach().cpu().item())
            rep = parts["loss_repulsion"].detach().cpu().item()
            acc["loss_repulsion"] += 0.0 if rep != rep else rep  # skip NaN
            acc["loss_pl"] += float(loss_pl.detach().cpu().item())
            acc["loss_mix"] += float(loss_mix.detach().cpu().item())
            acc["h2_source_mean"] += float(parts["h2_source_mean"].detach().cpu().item())
            acc["h2_target_mean"] += float(parts["h2_target_mean"].detach().cpu().item())
            acc["delta_align_mean"] += float(parts["delta_align_mean"].detach().cpu().item())
            drep = parts["delta_repulsion_mean"].detach().cpu().item()
            acc["delta_repulsion_mean"] += 0.0 if drep != drep else drep  # skip NaN
            acc["lambda_eff"] += float(parts["lambda_eff"].detach().cpu().item())
            acc["n_valid_classes"] += float(parts["n_valid_classes"].detach().cpu().item())
            acc["n_rep_pairs"] += float(parts["n_rep_pairs"].detach().cpu().item())
            acc["pseudo_conf"] += float(pseudo_conf.detach().cpu().item())

        epoch_time = time.time() - start_time
        denom = max(num_batches, 1)
        out = {k: v / denom for k, v in acc.items()}
        out["epoch_time"] = epoch_time
        return out
