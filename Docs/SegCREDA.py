"""SegCREDALoss: CREDA-style class-conditional Renyi-2 alignment for UDA
semantic segmentation, hardened against:

  1. Intra-class variance collapse (Dirac-delta pathology) via spectral
     entropy floors  H2(K_c) >= eta  <=>  effective-rank(K_c) >= 2^eta,
     plus a source-anchored target dispersion floor and stop-gradient
     asymmetric anchoring (target manifold chases a frozen source manifold).
  2. Missing inter-class repulsion via a margin-hinged Renyi block-matrix
     divergence between target class-conditional distributions.
  3. Alignment-gradient domination via warmup + sigmoid ramp + an EMA
     trust-region ratio computed on reference parameters (GradNorm-lite):
         lambda_eff = lambda_max * s(t) * min(1, rho * ||g_seg|| / ||g_aux||)

Total:  L = L_seg + lambda_eff * (L_align + beta * L_anti_collapse + gamma * L_repulsion)

All kernel/entropy math runs in float32 with autocast disabled (AMP-safe).

Usage:
    crit = SegCREDALoss(num_classes=C)
    ref  = list(model.encoder.layer4.parameters())   # reference params
    for batch in loader:
        total, parts = crit(feat_s, logits_s, labels_s, feat_t, logits_t)
        crit.update_lambda(parts["loss_seg"], parts["loss_aux"], ref)  # before backward
        total.backward()
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
"""
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_LN2 = math.log(2.0)


class SegCREDALoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        quota: int = 128,               # M: pixels sampled per class per batch
        min_samples: int = 8,           # tau: skip class if fewer pixels present
        lambda_max: float = 1.0,        # peak alignment weight
        beta: float = 1.0,              # weight of anti-collapse term in L_aux
        gamma: float = 0.5,             # weight of repulsion term in L_aux
        eta_floor: float = 1.0,         # bits: absolute H2 floor for source classes
        entropy_gap: float = 0.25,      # bits: require H2_target >= H2_source - gap
        repulsion_margin: float = 0.2,  # bits: hinge margin m for cross-class Delta
        warmup_steps: int = 1000,       # lambda = 0 before this many steps
        ramp_steps: int = 9000,         # sigmoid ramp length after warmup
        ramp_delta: float = 10.0,       # sigmoid steepness
        grad_ratio: float = 1.0,        # rho: max allowed ||g_aux|| / ||g_seg||
        ema_decay: float = 0.9,         # EMA for gradient norms
        weight_cross: bool = False,     # also weight K_st by target confidence
        dice_weight: float = 0.0,       # optional soft-Dice inside L_seg
        normalize_features: bool = False,
        sigma_min: float = 1e-3,
        sigma_max: float = 1e3,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.quota = quota
        self.min_samples = min_samples
        self.lambda_max = lambda_max
        self.beta = beta
        self.gamma = gamma
        self.eta_floor = eta_floor
        self.entropy_gap = entropy_gap
        self.repulsion_margin = repulsion_margin
        self.warmup_steps = warmup_steps
        self.ramp_steps = ramp_steps
        self.ramp_delta = ramp_delta
        self.grad_ratio = grad_ratio
        self.ema_decay = ema_decay
        self.weight_cross = weight_cross
        self.dice_weight = dice_weight
        self.normalize_features = normalize_features
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.eps = eps

        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        self.register_buffer("lambda_eff", torch.zeros(()))
        self.register_buffer("ema_g_seg", torch.zeros(()))
        self.register_buffer("ema_g_aux", torch.zeros(()))

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _flat_feat(f: torch.Tensor) -> torch.Tensor:
        # (B, d, h, w) -> (B*h*w, d)
        return f.flatten(2).transpose(1, 2).reshape(-1, f.shape[1])

    def _flat_labels(self, y: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        # (B, H, W) long -> (B*h*w,) at feature resolution
        if tuple(y.shape[-2:]) != tuple(size):
            y = F.interpolate(y.unsqueeze(1).float(), size=size, mode="nearest")
            y = y.squeeze(1).long()
        return y.reshape(-1)

    def _flat_probs(self, logits: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        # (B, C, H, W) -> (B*h*w, C) probabilities at feature resolution
        p = F.softmax(logits.float(), dim=1)
        if tuple(p.shape[-2:]) != tuple(size):
            p = F.interpolate(p, size=size, mode="bilinear", align_corners=False)
            p = p / p.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return p.flatten(2).transpose(1, 2).reshape(-1, p.shape[1])

    def _quota_indices(self, mask: torch.Tensor) -> Optional[torch.Tensor]:
        idx = mask.nonzero(as_tuple=False).squeeze(1)
        n = idx.numel()
        if n < self.min_samples:
            return None
        if n >= self.quota:
            return idx[torch.randperm(n, device=idx.device)[: self.quota]]
        return idx[torch.randint(0, n, (self.quota,), device=idx.device)]

    def _median_sigma(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = torch.cat([x.detach(), y.detach()], dim=0)
        d2 = torch.cdist(z, z, p=2.0) ** 2
        n = z.shape[0]
        iu = torch.triu_indices(n, n, offset=1, device=z.device)
        med = d2[iu[0], iu[1]].median()
        return (med + 1e-12).sqrt().clamp(self.sigma_min, self.sigma_max).detach()

    def _rbf(self, x: torch.Tensor, y: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        d2 = torch.cdist(x, y, p=2.0) ** 2
        return torch.exp(-d2 / (2.0 * sigma * sigma + self.eps))

    def _h2_bits(self, s2: torch.Tensor, tr: torch.Tensor) -> torch.Tensor:
        # H2(A) = -log2 tr(A^2),  tr(A^2) = sum(K^2) / tr(K)^2
        ptr = s2 / (tr * tr).clamp_min(self.eps)
        return -torch.log(ptr.clamp_min(self.eps)) / _LN2

    def _delta_bits(self, s2_p, tr_p, s2_q, tr_q, s2_pq) -> torch.Tensor:
        # Delta(P,Q) = H2(K_mix) - 0.5*(H2(K_P) + H2(K_Q)) in bits, built from
        # purity sums only (no explicit block matrix):
        #   ptr_joint = (S2_PP + S2_QQ + 2*S2_PQ) / (T_P + T_Q)^2
        ptr_joint = (s2_p + s2_q + 2.0 * s2_pq) / ((tr_p + tr_q) ** 2).clamp_min(self.eps)
        ptr_p = s2_p / (tr_p * tr_p).clamp_min(self.eps)
        ptr_q = s2_q / (tr_q * tr_q).clamp_min(self.eps)
        return (
            -torch.log(ptr_joint.clamp_min(self.eps))
            + 0.5 * torch.log(ptr_p.clamp_min(self.eps))
            + 0.5 * torch.log(ptr_q.clamp_min(self.eps))
        ) / _LN2

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        feat_s: torch.Tensor,     # (Bs, d, h, w)  source deep features
        logits_s: torch.Tensor,   # (Bs, C, H, W)  source logits
        labels_s: torch.Tensor,   # (Bs, H, W)     source ground truth (long)
        feat_t: torch.Tensor,     # (Bt, d, h, w)  target deep features
        logits_t: torch.Tensor,   # (Bt, C, H, W)  target logits
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        # 1) Supervised anchor (runs in the caller's autocast context).
        loss_seg = F.cross_entropy(logits_s, labels_s)
        if self.dice_weight > 0.0:
            ps = F.softmax(logits_s.float(), dim=1)
            oh = F.one_hot(labels_s, self.num_classes).permute(0, 3, 1, 2).float()
            inter = (ps * oh).sum(dim=(2, 3))
            denom = ps.sum(dim=(2, 3)) + oh.sum(dim=(2, 3))
            dice = 1.0 - ((2.0 * inter + 1.0) / (denom + 1.0)).mean()
            loss_seg = loss_seg + self.dice_weight * dice

        # 2) CREDA core: fp32, autocast off (AMP-safe).
        dev = feat_s.device
        with torch.autocast(device_type=dev.type, enabled=False):
            fs = self._flat_feat(feat_s).float()                    # (Ns, d)
            ft = self._flat_feat(feat_t).float()                    # (Nt, d)
            ys = self._flat_labels(labels_s, feat_s.shape[-2:])     # (Ns,)
            pt = self._flat_probs(logits_t, feat_t.shape[-2:])      # (Nt, C)

            if self.normalize_features:
                fs = F.normalize(fs, dim=-1)
                ft = F.normalize(ft, dim=-1)

            C = self.num_classes
            # Per-pixel Renyi-2 confidence weight: w = 1 - H2(p)/log2(C)
            p2 = (pt * pt).sum(dim=1).clamp_min(self.eps)
            w_t = (1.0 + torch.log2(p2) / math.log2(C)).clamp(0.0, 1.0)
            pl_t = pt.detach().argmax(dim=1)                        # hard pseudo-labels

            zero = fs.new_zeros(())
            align_terms: List[torch.Tensor] = []
            ac_terms: List[torch.Tensor] = []
            t_feats: List[torch.Tensor] = []
            t_weights: List[torch.Tensor] = []
            t_sigmas: List[torch.Tensor] = []
            h2s_list: List[torch.Tensor] = []
            h2t_list: List[torch.Tensor] = []
            dal_list: List[torch.Tensor] = []

            for c in range(C):
                idx_s = self._quota_indices(ys == c)
                idx_t = self._quota_indices(pl_t == c)
                if idx_s is None or idx_t is None:
                    continue

                xs = fs[idx_s]                    # (M, d), grad (source floor)
                xt = ft[idx_t]                    # (M, d), grad
                wc = w_t[idx_t]                   # (M,)
                sig = self._median_sigma(xs, xt)  # detached

                K_s = self._rbf(xs, xs, sig)                       # grad
                K_t = self._rbf(xt, xt, sig)                       # grad
                Kt_w = K_t * (wc[:, None] * wc[None, :])           # weighted target Gram
                K_st = self._rbf(xs.detach(), xt, sig)             # anchor: no grad -> source
                if self.weight_cross:
                    K_st = K_st * wc[None, :]

                s2_s = (K_s * K_s).sum()
                tr_s = K_s.trace()
                s2_t = (Kt_w * Kt_w).sum()
                tr_t = (wc * wc).sum().clamp_min(self.eps)
                s2_st = (K_st * K_st).sum()

                # Alignment Delta with source side detached (asymmetric anchoring).
                d_c = self._delta_bits(s2_s.detach(), tr_s.detach(), s2_t, tr_t, s2_st)

                H_s = self._h2_bits(s2_s, tr_s)     # grad: pushes source open if collapsed
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

            # 3) Inter-class repulsion in the target domain (batched over pairs).
            if len(t_feats) >= 2:
                Xt = torch.stack(t_feats)                      # (Cv, M, d)
                Wt = torch.stack(t_weights)                    # (Cv, M)
                sig_r = torch.stack(t_sigmas).mean().detach()

                # NB: broadcast dims are required: cdist on (Cv,M,d)x(Cv,M,d)
                # alone would only produce within-class (Cv,M,M) blocks.
                D2 = torch.cdist(Xt.unsqueeze(1), Xt.unsqueeze(0), p=2.0) ** 2  # (Cv,Cv,M,M)
                K = torch.exp(-D2 / (2.0 * sig_r * sig_r + self.eps))
                Wij = Wt[:, None, :, None] * Wt[None, :, None, :]
                Kw = K * Wij

                s2 = (Kw * Kw).sum(dim=(2, 3))                 # (Cv, Cv)
                trv = (Wt * Wt).sum(dim=1).clamp_min(self.eps)  # (Cv,)
                s2d = torch.diagonal(s2)                       # (Cv,)

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
                deltas = Dmat[iu[0], iu[1]]                    # (Cv*(Cv-1)/2,)
                loss_rep = F.relu(self.repulsion_margin - deltas).mean()
                rep_mean = deltas.detach().mean()
                n_pairs = deltas.numel()
            else:
                loss_rep = zero
                rep_mean = fs.new_tensor(float("nan"))
                n_pairs = 0

            loss_aux = loss_align + self.beta * loss_ac + self.gamma * loss_rep

        # float(): lambda_eff is a scheduled scalar, not a graph node; using the
        # buffer tensor directly would break backward when update_lambda mutates
        # it in place between forward and backward.
        total = loss_seg + float(self.lambda_eff) * loss_aux

        parts: Dict[str, torch.Tensor] = {
            "total": total,
            "loss_seg": loss_seg,          # graph tensors (needed by update_lambda)
            "loss_aux": loss_aux,          # graph tensor
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

    # ------------------------------------------------- gradient trust region
    def update_lambda(
        self,
        loss_seg: torch.Tensor,
        loss_aux: torch.Tensor,
        ref_params: Sequence[torch.Tensor],
    ) -> float:
        """Call AFTER forward and BEFORE total.backward(); uses autograd.grad on
        reference parameters (e.g. last shared encoder block), keeps the graph.

            lambda_eff = lambda_max * s(t) * min(1, rho * EMA||g_seg|| / EMA||g_aux||)
        """
        t = int(self.step.item())
        self.step.add_(1)
        if t < self.warmup_steps:
            self.lambda_eff.zero_()
            return 0.0

        def safe_grad(loss: torch.Tensor):
            if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
                return [None] * len(list(ref_params))
            return torch.autograd.grad(loss, ref_params, retain_graph=True, allow_unused=True)

        g_s = safe_grad(loss_seg)
        g_a = safe_grad(loss_aux)

        def gnorm(gs) -> torch.Tensor:
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
