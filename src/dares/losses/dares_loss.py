"""DARESLoss: hardened class-conditional Renyi-2 alignment for UDA segmentation.

Extends matrix-based Renyi-2 alignment with four stabilization safeguards:
  1. Balanced trace-normalized density mixture guaranteeing Delta >= 0 bits.
  2. Spectral entropy floors H2(K_c) >= eta with asymmetric stop-gradient anchoring.
  3. Balanced margin-hinged inter-class repulsion between target classes.
  4. Continuous-EMA GradNorm trust region preventing auxiliary gradient domination.
"""

import math
from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_LN2 = math.log(2.0)


class DARESLoss(nn.Module):
    """Hardened Renyi-2 alignment loss for dense semantic segmentation."""

    def __init__(
        self,
        num_classes: int = 2,
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
        trust_region: bool = True,
        ema_decay: float = 0.9,
        align_form: str = "mi",
        use_renyi_em: bool = True,
        lambda_em: float = 0.05,
        em_pool: bool = False,
        em_pool_kernel: int = 3,
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
        self.trust_region = bool(trust_region)
        self.ema_decay = float(ema_decay)
        self.align_form = align_form
        self.use_renyi_em = bool(use_renyi_em)
        self.lambda_em = float(lambda_em)
        self.em_pool = bool(em_pool)
        self.em_pool_kernel = int(em_pool_kernel)
        self.weight_cross = bool(weight_cross)
        self.normalize_features = bool(normalize_features)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.eps = float(eps)

        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        self.register_buffer("lambda_eff", torch.zeros(()))
        self.register_buffer("ema_g_seg", torch.zeros(()))
        self.register_buffer("ema_g_aux", torch.zeros(()))

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

    def _flat_probs(self, logits: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """(B, C, H, W) -> (B*h*w, C) probabilities at feature resolution."""
        p = F.softmax(logits.float(), dim=1)
        if tuple(p.shape[-2:]) != tuple(size):
            p = F.interpolate(p, size=size, mode="bilinear", align_corners=False)
            p = p / p.sum(dim=1, keepdim=True).clamp_min(self.eps)
        return p.flatten(2).transpose(1, 2).reshape(-1, p.shape[1])

    def _quota_indices(self, mask: torch.Tensor) -> Optional[torch.Tensor]:
        """Draws up to quota indices per class."""
        idx = mask.nonzero(as_tuple=False).squeeze(1)
        n = idx.numel()
        if n < self.min_samples:
            return None
        if n >= self.quota:
            return idx[torch.randperm(n, device=idx.device)[: self.quota]]
        return idx[torch.randint(0, n, (self.quota,), device=idx.device)]

    def _spatial_pool(self, w: torch.Tensor, batch: int, size: tuple[int, int]) -> torch.Tensor:
        """Average-pools confidence weight over a spatial window."""
        wmap = w.view(batch, 1, size[0], size[1])
        pad = self.em_pool_kernel // 2
        wmap = F.pad(wmap, (pad, pad, pad, pad), mode="replicate")
        pooled = F.avg_pool2d(wmap, kernel_size=self.em_pool_kernel, stride=1)
        return pooled.reshape(-1)

    def _median_sigma(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Median-heuristic bandwidth over combined pooled features."""
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
        ptr = s2 / (tr * tr).clamp_min(self.eps)
        return -torch.log2(ptr.clamp_min(self.eps))

    def _delta_bits_normalized(
        self,
        s2_p: torch.Tensor,
        tr_p: torch.Tensor,
        s2_q: torch.Tensor,
        tr_q: torch.Tensor,
        s2_pq: torch.Tensor,
    ) -> torch.Tensor:
        """Trace-normalized Renyi-2 divergence strictly bounded >= 0 bits."""
        ptr_p = s2_p / (tr_p * tr_p).clamp_min(self.eps)
        ptr_q = s2_q / (tr_q * tr_q).clamp_min(self.eps)
        ptr_pq = s2_pq / (tr_p * tr_q).clamp_min(self.eps)

        ptr_joint = 0.25 * (ptr_p + ptr_q + 2.0 * ptr_pq)
        delta = (
            -torch.log2(ptr_joint.clamp_min(self.eps))
            + 0.5 * torch.log2(ptr_p.clamp_min(self.eps))
            + 0.5 * torch.log2(ptr_q.clamp_min(self.eps))
        )
        return F.relu(delta)

    def forward(
        self,
        feat_s: torch.Tensor,
        logits_s: torch.Tensor,
        labels_s: torch.Tensor,
        feat_t: torch.Tensor,
        logits_t: torch.Tensor,
    ):
        """Computes total DARES loss in float32."""
        loss_seg = F.cross_entropy(logits_s, labels_s, ignore_index=255)

        dev = feat_s.device
        with torch.autocast(device_type=dev.type, enabled=False):
            fs = self._flat_feat(feat_s).float()
            ft = self._flat_feat(feat_t).float()
            ys = self._flat_labels(labels_s, feat_s.shape[-2:])
            pt = self._flat_probs(logits_t, feat_t.shape[-2:])

            if self.normalize_features:
                fs = F.normalize(fs, dim=-1)
                ft = F.normalize(ft, dim=-1)

            C = self.num_classes
            p2 = (pt * pt).sum(dim=1).clamp_min(self.eps)
            w_t = (1.0 + torch.log2(p2) / math.log2(C)).clamp(0.0, 1.0)
            pl_t = pt.detach().argmax(dim=1)

            zero = fs.new_zeros(())

            # Dense Renyi-EM on confident predictions
            h2_pred = -torch.log2(p2)
            if self.use_renyi_em:
                w_agg = (
                    self._spatial_pool(w_t, feat_t.shape[0], feat_t.shape[-2:])
                    if self.em_pool
                    else w_t
                )
                loss_em = (w_agg * h2_pred).mean()
            else:
                loss_em = zero

            align_terms, ac_terms = [], []
            t_feats, t_weights, t_sigmas = [], [], []
            h2s_list, h2t_list, dal_list = [], [], []

            for c in range(C):
                idx_s = self._quota_indices(ys == c)
                idx_t = self._quota_indices(pl_t == c)
                if idx_s is None or idx_t is None:
                    continue

                xs = fs[idx_s]
                xt = ft[idx_t]
                wc = w_t[idx_t]
                sig = self._median_sigma(xs, xt)

                K_s = self._rbf(xs, xs, sig)
                K_t = self._rbf(xt, xt, sig)
                Kt_w = K_t * (wc[:, None] * wc[None, :])
                K_st = self._rbf(xs.detach(), xt, sig)
                if self.weight_cross:
                    K_st = K_st * wc[None, :]

                s2_s = (K_s * K_s).sum()
                tr_s = K_s.trace()
                s2_t = (Kt_w * Kt_w).sum()
                tr_t = (wc * wc).sum().clamp_min(self.eps)
                s2_st = (K_st * K_st).sum()

                d_c = self._delta_bits_normalized(
                    s2_s.detach(), tr_s.detach(), s2_t, tr_t, s2_st
                )

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

            # Target inter-class repulsion
            if len(t_feats) >= 2:
                M_max = max(tf.shape[0] for tf in t_feats)
                Xt = torch.stack(
                    [F.pad(tf, (0, 0, 0, M_max - tf.shape[0])) for tf in t_feats]
                )
                Wt = torch.stack(
                    [F.pad(wf, (0, M_max - wf.shape[0])) for wf in t_weights]
                )
                sig_r = torch.stack(t_sigmas).mean().detach()

                D2 = torch.cdist(Xt.unsqueeze(1), Xt.unsqueeze(0), p=2.0) ** 2
                K = torch.exp(-D2 / (2.0 * sig_r * sig_r + self.eps))
                Wij = Wt[:, None, :, None] * Wt[None, :, None, :]
                Kw = K * Wij

                s2 = (Kw * Kw).sum(dim=(2, 3))
                trv = (Wt * Wt).sum(dim=1).clamp_min(self.eps)
                s2d = torch.diagonal(s2)

                ptr_diag = s2d / (trv * trv).clamp_min(self.eps)
                ptr_cross = s2 / (trv[:, None] * trv[None, :]).clamp_min(self.eps)
                ptr_joint = 0.25 * (ptr_diag[:, None] + ptr_diag[None, :] + 2.0 * ptr_cross)

                Dmat = F.relu(
                    -torch.log2(ptr_joint.clamp_min(self.eps))
                    + 0.5 * torch.log2(ptr_diag[:, None].clamp_min(self.eps))
                    + 0.5 * torch.log2(ptr_diag[None, :].clamp_min(self.eps))
                )

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

        total = loss_seg + float(self.lambda_eff) * loss_aux + self.lambda_em * loss_em

        parts = {
            "total": total,
            "loss_seg": loss_seg,
            "loss_aux": loss_aux,
            "loss_align": loss_align.detach(),
            "loss_anti_collapse": loss_ac.detach(),
            "loss_repulsion": loss_rep.detach(),
            "loss_em": loss_em,
            "h2_source_mean": torch.stack(h2s_list).mean() if h2s_list else zero,
            "h2_target_mean": torch.stack(h2t_list).mean() if h2t_list else zero,
            "delta_align_mean": torch.stack(dal_list).mean() if dal_list else zero,
            "delta_repulsion_mean": rep_mean,
            "lambda_eff": self.lambda_eff.detach(),
            "n_valid_classes": torch.tensor(len(align_terms), device=dev),
            "n_rep_pairs": torch.tensor(n_pairs, device=dev),
        }
        return total, parts

    def in_warmup(self) -> bool:
        return int(self.step.item()) < self.warmup_steps

    def update_lambda(
        self,
        loss_seg: torch.Tensor,
        loss_aux: torch.Tensor,
        ref_params: Sequence[torch.Tensor],
    ) -> float:
        """Updates lambda_eff with stabilized gradient-norm moving averages."""
        t = int(self.step.item())
        self.step.add_(1)

        ref = [r for r in ref_params if r.requires_grad]

        def safe_grad(loss: torch.Tensor):
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

        ns = gnorm(g_s)
        na = gnorm(g_a)
        d = self.ema_decay

        # Actualización continua de EMA incluso durante warm-up
        if self.ema_g_seg.item() == 0.0 and ns.item() > 0.0:
            self.ema_g_seg.fill_(ns.item())
        else:
            self.ema_g_seg.mul_(d).add_((1.0 - d) * ns)

        if na.item() > 0.0:
            if self.ema_g_aux.item() == 0.0:
                self.ema_g_aux.fill_(na.item())
            else:
                self.ema_g_aux.mul_(d).add_((1.0 - d) * na)

        if t < self.warmup_steps:
            self.lambda_eff.zero_()
            return 0.0

        p = min(1.0, (t - self.warmup_steps) / max(1, self.ramp_steps))
        s = (1.0 - math.exp(-self.ramp_delta * p)) / (
            1.0 + math.exp(-self.ramp_delta * p)
        )

        if not self.trust_region:
            lam = float(self.lambda_max * s)
            self.lambda_eff.fill_(lam)
            return lam

        ratio = (self.grad_ratio * self.ema_g_seg / (self.ema_g_aux + self.eps)).clamp(
            min=0.15, max=1.0
        )
        lam = float(self.lambda_max * s * ratio.item())
        self.lambda_eff.fill_(lam)
        return lam