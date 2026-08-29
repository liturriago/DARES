"""DARESLoss: hardened class-conditional Renyi-2 alignment for UDA segmentation.

This is the DARES adaptation loss, ported from the reference module shipped in
``Docs/`` (theory in ``Docs/KimiReport.txt``) and cleaned to match the DARES
codebase conventions. It extends the base CREDA alignment with three
safeguards:

  1. Intra-class variance collapse is prevented by spectral entropy floors
     ``H2(K_c) >= eta`` (anti-collapse) with asymmetric stop-gradient anchoring:
     the target manifold chases a *frozen* source manifold.
  2. Missing inter-class repulsion is added through a margin-hinged Renyi
     block-matrix divergence between target class-conditional distributions.
  3. Alignment-gradient domination is governed per-step by a GradNorm-lite EMA
     trust region: ``lambda_eff = lambda_max * s(t) * min(1, rho * g_seg/g_aux)``
     computed on the deepest shared encoder block (reference parameters).

All kernel / entropy math runs in float32 with ``autocast`` disabled (AMP-safe).

Objective::
    L = L_seg + lambda_eff * (L_align + beta * L_anti_collapse + gamma * L_repulsion)
"""

import math
from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_LN2 = math.log(2.0)


class DARESLoss(nn.Module):
    """CREDA-style Renyi-2 alignment hardened for dense segmentation.

    Args:
        num_classes (int): Number of semantic classes.
        quota (int): ``M`` — pixels sampled per class per batch.
        min_samples (int): ``tau`` — skip a class if fewer pixels are present.
        lambda_max (float): Peak alignment weight.
        beta (float): Weight of the anti-collapse term inside ``L_aux``.
        gamma (float): Weight of the repulsion term inside ``L_aux``.
        eta_floor (float): Absolute ``H2`` floor (bits) for source classes.
        entropy_gap (float): Require ``H2_target >= H2_source - gap`` (bits).
        repulsion_margin (float): Hinge margin ``m`` (bits) for cross-class
            target divergence.
        warmup_steps (int): Steps with ``lambda_eff = 0`` (source-only warm-up).
        ramp_steps (int): Sigmoid ramp length (in steps) after warm-up.
        ramp_delta (float): Sigmoid steepness.
        grad_ratio (float): ``rho`` — max allowed ``||g_aux|| / ||g_seg||``.
        ema_decay (float): EMA decay for the gradient-norm moving averages.
        weight_cross (bool): Also weight the cross-block ``K_st`` by target
            confidence (symmetric generalization of the paper's weighting).
        normalize_features (bool): L2-normalize the sampled feature vectors.
        sigma_min (float): Lower clamp for the median-heuristic bandwidth.
        sigma_max (float): Upper clamp for the median-heuristic bandwidth.
        eps (float): Numerical epsilon.
    """

    def __init__(
        self,
        num_classes: int,
        quota: int = 256,
        min_samples: int = 8,
        lambda_max: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.5,
        eta_floor: float = 1.0,
        entropy_gap: float = 0.25,
        repulsion_margin: float = 0.2,
        warmup_steps: int = 1000,
        ramp_steps: int = 4000,
        ramp_delta: float = 10.0,
        grad_ratio: float = 0.8,
        ema_decay: float = 0.9,
        weight_cross: bool = False,
        normalize_features: bool = False,
        sigma_min: float = 1e-3,
        sigma_max: float = 1e3,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.quota = int(quota)
        self.min_samples = int(min_samples)
        self.lambda_max = float(lambda_max)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.eta_floor = float(eta_floor)
        self.entropy_gap = float(entropy_gap)
        self.repulsion_margin = float(repulsion_margin)
        self.warmup_steps = int(warmup_steps)
        self.ramp_steps = int(ramp_steps)
        self.ramp_delta = float(ramp_delta)
        self.grad_ratio = float(grad_ratio)
        self.ema_decay = float(ema_decay)
        self.weight_cross = bool(weight_cross)
        self.normalize_features = bool(normalize_features)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.eps = float(eps)

        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        self.register_buffer("lambda_eff", torch.zeros(()))
        self.register_buffer("ema_g_seg", torch.zeros(()))
        self.register_buffer("ema_g_aux", torch.zeros(()))

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _flat_feat(f: torch.Tensor) -> torch.Tensor:
        """(B, d, h, w) -> (B*h*w, d)."""
        return f.flatten(2).transpose(1, 2).reshape(-1, f.shape[1])

    @staticmethod
    def _flat_labels(y: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """(B, H, W) -> (B*h*w,) long at feature resolution."""
        if tuple(y.shape[-2:]) != tuple(size):
            y = F.interpolate(y.unsqueeze(1).float(), size=size, mode="nearest")
            y = y.squeeze(1).long()
        return y.reshape(-1)

    def _flat_probs(self, logits, size):
        """(B, C, H, W) -> (B*h*w, C) probabilities at feature resolution."""
        p = F.softmax(logits.float(), dim=1)
        if tuple(p.shape[-2:]) != tuple(size):
            p = F.interpolate(p, size=size, mode="bilinear", align_corners=False)
            p = p / p.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return p.flatten(2).transpose(1, 2).reshape(-1, p.shape[1])

    def _quota_indices(self, mask: torch.Tensor) -> Optional[torch.Tensor]:
        """Selects up to ``quota`` indices from a class membership mask."""
        idx = mask.nonzero(as_tuple=False).squeeze(1)
        n = idx.numel()
        if n < self.min_samples:
            return None
        if n >= self.quota:
            return idx[torch.randperm(n, device=idx.device)[: self.quota]]
        return idx[torch.randint(0, n, (self.quota,), device=idx.device)]

    def _median_sigma(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Median-heuristic bandwidth over the combined (detached) features."""
        z = torch.cat([x.detach(), y.detach()], dim=0)
        d2 = torch.cdist(z, z, p=2.0) ** 2
        n = z.shape[0]
        iu = torch.triu_indices(n, n, offset=1, device=z.device)
        med = d2[iu[0], iu[1]].median()
        return (med + 1e-12).sqrt().clamp(self.sigma_min, self.sigma_max).detach()

    def _rbf(self, x, y, sigma):
        d2 = torch.cdist(x, y, p=2.0) ** 2
        return torch.exp(-d2 / (2.0 * sigma * sigma + self.eps))

    def _h2_bits(self, s2, tr):
        ptr = s2 / (tr * tr).clamp_min(self.eps)
        return -torch.log(ptr.clamp_min(self.eps)) / _LN2

    def _delta_bits(self, s2_p, tr_p, s2_q, tr_q, s2_pq):
        """Delta(P,Q) in bits from purity sums (no block matrix needed)."""
        ptr_joint = (s2_p + s2_q + 2.0 * s2_pq) / ((tr_p + tr_q) ** 2).clamp_min(self.eps)
        ptr_p = s2_p / (tr_p * tr_p).clamp_min(self.eps)
        ptr_q = s2_q / (tr_q * tr_q).clamp_min(self.eps)
        return (
            -torch.log(ptr_joint.clamp_min(self.eps))
            + 0.5 * torch.log(ptr_p.clamp_min(self.eps))
            + 0.5 * torch.log(ptr_q.clamp_min(self.eps))
        ) / _LN2

    # ---------------------------------------------------------------- forward
    def forward(self, feat_s, logits_s, labels_s, feat_t, logits_t):
        """Computes the DARES objective.

        Args:
            feat_s (torch.Tensor): Source deep features ``(Bs, d, h, w)``.
            logits_s (torch.Tensor): Source logits ``(Bs, C, H, W)``.
            labels_s (torch.Tensor): Source labels ``(Bs, H, W)`` (long).
            feat_t (torch.Tensor): Target deep features ``(Bt, d, h, w)``.
            logits_t (torch.Tensor): Target logits ``(Bt, C, H, W)``.

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]: ``(total, parts)``
                where ``parts`` carries graph tensors ``loss_seg`` /
                ``loss_aux`` (needed by :meth:`update_lambda`) plus diagnostics.
        """
        # 1) Supervised anchor (runs in the caller's autocast context).
        loss_seg = F.cross_entropy(logits_s, labels_s, ignore_index=255)

        # 2) CREDA core: fp32, autocast off (AMP-safe).
        dev = feat_s.device
        with torch.autocast(device_type=dev.type, enabled=False):
            fs = self._flat_feat(feat_s).float()                # (Ns, d)
            ft = self._flat_feat(feat_t).float()                # (Nt, d)
            ys = self._flat_labels(labels_s, feat_s.shape[-2:])  # (Ns,)
            pt = self._flat_probs(logits_t, feat_t.shape[-2:])  # (Nt, C)

            if self.normalize_features:
                fs = F.normalize(fs, dim=-1)
                ft = F.normalize(ft, dim=-1)

            C = self.num_classes
            p2 = (pt * pt).sum(dim=1).clamp_min(self.eps)
            w_t = (1.0 + torch.log2(p2) / math.log2(C)).clamp(0.0, 1.0)
            pl_t = pt.detach().argmax(dim=1)

            zero = fs.new_zeros(())
            align_terms, ac_terms = [], []
            t_feats, t_weights, t_sigmas = [], [], []
            h2s_list, h2t_list, dal_list = [], [], []

            for c in range(C):
                idx_s = self._quota_indices(ys == c)
                idx_t = self._quota_indices(pl_t == c)
                if idx_s is None or idx_t is None:
                    continue

                xs = fs[idx_s]                              # (M, d)
                xt = ft[idx_t]                              # (M, d)
                wc = w_t[idx_t]                             # (M,)
                sig = self._median_sigma(xs, xt)            # detached

                K_s = self._rbf(xs, xs, sig)
                K_t = self._rbf(xt, xt, sig)
                Kt_w = K_t * (wc[:, None] * wc[None, :])
                K_st = self._rbf(xs.detach(), xt, sig)      # anchor: no grad -> source
                if self.weight_cross:
                    K_st = K_st * wc[None, :]

                s2_s = (K_s * K_s).sum()
                tr_s = K_s.trace()
                s2_t = (Kt_w * Kt_w).sum()
                tr_t = (wc * wc).sum().clamp_min(self.eps)
                s2_st = (K_st * K_st).sum()

                d_c = self._delta_bits(s2_s.detach(), tr_s.detach(), s2_t, tr_t, s2_st)

                H_s = self._h2_bits(s2_s, tr_s)
                H_t = self._h2_bits(s2_t, tr_t)
                floor_s = F.relu(H_s.new_tensor(self.eta_floor) - H_s)
                floor_t = F.relu(H_s.detach() - self.entropy_gap - H_t)

                align_terms.append(d_c)
                ac_terms.append(floor_s + floor_t)
                t_feats.append(xt)
                t_weights.append(wc)
                t_sigmas.append(sig)
                h2s_list.append(H_s.detach())
                h2t_list.append(H_t.detach())
                dal_list.append(d_c.detach())

            loss_align = torch.stack(align_terms).mean() if align_terms else zero
            loss_ac = torch.stack(ac_terms).mean() if ac_terms else zero

            # 3) Inter-class target repulsion (batched over class pairs).
            if len(t_feats) >= 2:
                Xt = torch.stack(t_feats)                    # (Cv, M, d)
                Wt = torch.stack(t_weights)                  # (Cv, M)
                sig_r = torch.stack(t_sigmas).mean().detach()

                D2 = torch.cdist(Xt.unsqueeze(1), Xt.unsqueeze(0), p=2.0) ** 2
                K = torch.exp(-D2 / (2.0 * sig_r * sig_r + self.eps))
                Wij = Wt[:, None, :, None] * Wt[None, :, None, :]
                Kw = K * Wij

                s2 = (Kw * Kw).sum(dim=(2, 3))               # (Cv, Cv)
                trv = (Wt * Wt).sum(dim=1).clamp_min(self.eps)  # (Cv,)
                s2d = torch.diagonal(s2)                     # (Cv,)

                ptr_joint = (s2d[:, None] + s2d[None, :] + 2.0 * s2) / (
                    (trv[:, None] + trv[None, :]) ** 2
                ).clamp_min(self.eps)
                ptr_m = s2d / (trv * trv).clamp_min(self.eps)
                Dmat = (
                    -torch.log(ptr_joint.clamp_min(self.eps))
                    + 0.5 * torch.log(ptr_m[:, None].clamp_min(self.eps))
                    + 0.5 * torch.log(ptr_m[None, :].clamp_min(self.eps))
                ) / _LN2

                Cv = Xt.shape[0]
                iu = torch.triu_indices(Cv, Cv, offset=1, device=Dmat.device)
                deltas = Dmat[iu[0], iu[1]]
                loss_rep = F.relu(self.repulsion_margin - deltas).mean()
                rep_mean = deltas.detach().mean()
                n_pairs = int(deltas.numel())
            else:
                loss_rep = zero
                rep_mean = fs.new_tensor(float("nan"))
                n_pairs = 0

            loss_aux = loss_align + self.beta * loss_ac + self.gamma * loss_rep

        # float(): lambda_eff is a scheduled scalar, not a graph node; using the
        # buffer tensor directly would break backward when update_lambda mutates
        # it in place between forward and backward.
        total = loss_seg + float(self.lambda_eff) * loss_aux

        parts = {
            "total": total,
            "loss_seg": loss_seg,
            "loss_aux": loss_aux,
            "loss_align": loss_align.detach(),
            "loss_anti_collapse": loss_ac.detach(),
            "loss_repulsion": loss_rep.detach(),
            "h2_source_mean": torch.stack(h2s_list).mean() if h2s_list else zero,
            "h2_target_mean": torch.stack(h2t_list).mean() if h2t_list else zero,
            "delta_align_mean": torch.stack(dal_list).mean() if dal_list else zero,
            "delta_repulsion_mean": rep_mean,
            "lambda_eff": self.lambda_eff.detach(),
            "n_valid_classes": torch.tensor(len(align_terms), device=dev),
            "n_rep_pairs": torch.tensor(n_pairs, device=dev),
        }
        return total, parts

    # -------------------------------------------------------- trust region
    def update_lambda(
        self,
        loss_seg: torch.Tensor,
        loss_aux: torch.Tensor,
        ref_params: Sequence[torch.Tensor],
    ) -> float:
        """Updates ``lambda_eff`` from a GradNorm-lite EMA trust region.

        Call AFTER ``forward`` and BEFORE ``total.backward()``. Uses
        ``autograd.grad`` on the reference parameters (deepest shared encoder
        block) with ``retain_graph=True``, so the computation graph is kept for
        the main backward pass of the total loss.

        Args:
            loss_seg (torch.Tensor): Supervised segmentation loss (graph node).
            loss_aux (torch.Tensor): Auxiliary alignment loss (graph node).
            ref_params (Sequence[torch.Tensor]): Reference parameters, typically
                ``model.backbone.reference_params``.

        Returns:
            float: The newly scheduled ``lambda_eff``.
        """
        t = int(self.step.item())
        self.step.add_(1)
        if t < self.warmup_steps:
            self.lambda_eff.zero_()
            return 0.0

        ref = [r for r in ref_params if r.requires_grad]

        def safe_grad(loss):
            if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
                return [None] * len(ref)
            if not ref:
                return []
            return torch.autograd.grad(loss, ref, retain_graph=True, allow_unused=True)

        g_s = safe_grad(loss_seg)
        g_a = safe_grad(loss_aux)

        def gnorm(gs):
            acc = torch.zeros((), device=self.lambda_eff.device)
            for g in gs:
                if g is not None:
                    acc = acc + (g.detach().float() ** 2).sum()
            return acc.sqrt()

        ns, na = gnorm(g_s), gnorm(g_a)
        d = self.ema_decay
        self.ema_g_seg.mul_(d).add_((1.0 - d) * ns)
        self.ema_g_aux.mul_(d).add_((1.0 - d) * na)

        ratio = (self.grad_ratio * self.ema_g_seg / (self.ema_g_aux + self.eps)).clamp(max=1.0)
        p = min(1.0, (t - self.warmup_steps) / max(1, self.ramp_steps))
        s = (1.0 - math.exp(-self.ramp_delta * p)) / (1.0 + math.exp(-self.ramp_delta * p))
        lam = float(self.lambda_max * s * ratio.item())
        self.lambda_eff.fill_(lam)
        return lam
