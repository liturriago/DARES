"""DARESLossV2: MIL-CREDA hardened alignment for UDA segmentation.

Coexists with the original :class:`dares.losses.dares_loss.DARESLoss` (V1)
and inherits all of its machinery (quota sampling, median bandwidth,
anti-collapse floors, inter-class repulsion, GradNorm-lite trust region).
V2 upgrades the alignment core with three ideas from the MIL-CREDA
reference implementation (``Domain_Adaptation/src/MIL_CREDA``):

  1. Bounded class-global term (Eqs. 27, 32-37): the mixed matrix replaces
     the ReLU-clamped information radius, and conservative entropy bounds
     map each class score onto a common [0, 1] scale. Unlike ``relu(delta)``
     the term keeps a gradient at perfect alignment and rewards within-class
     diversity.
  2. Local correspondence term (Eqs. 28-31, 38): every target pixel pulls
     itself towards a confidence-weighted, softmax-personalized reference
     built from the source pixels of its class, measured as a squared RKHS
     distance computed entirely from kernel evaluations.
  3. Soft class weights (Eq. 29): target pixels are weighted by their full
     softmax mass for the class, not just the hard argmax pseudo-label, so
     ambiguous pixels contribute less to the conditional geometry.

Also fixes the V1 edge cases: an explicit ``num_classes >= 2`` guard and an
optional supervised-term normalization by its exact supremum (Eq. 18).
"""

import math

import torch
import torch.nn.functional as F

from dares.losses.dares_loss import DARESLoss


class DARESLossV2(DARESLoss):
    """MIL-CREDA hardened Renyi-2 alignment loss (bounded global + local)."""

    def __init__(
        self,
        *args,
        lambda_local: float = 0.5,
        tau_local: float = 1.0,
        soft_class_weights: bool = True,
        bounded_align: bool = True,
        normalize_seg: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if self.num_classes < 2:
            raise ValueError(
                "the confidence normalizer log2(C) requires at least two classes"
            )
        self.lambda_local = float(lambda_local)
        self.tau_local = float(tau_local)
        self.soft_class_weights = bool(soft_class_weights)
        self.bounded_align = bool(bounded_align)
        self.normalize_seg = bool(normalize_seg)

    def _source_bound(self) -> float:
        """Exact supremum of a single-pixel CE (Eq. 18): ln(1 + 1/eps)."""
        return math.log(1.0 + 1.0 / self.eps)

    def _class_weights(
        self,
        pt: torch.Tensor,
        w_t: torch.Tensor,
        idx_t: torch.Tensor,
        c: int,
    ) -> torch.Tensor:
        """Confidence weight per sampled target pixel (Eq. 24, soft Eq. 29)."""
        wc = w_t[idx_t]
        if self.soft_class_weights:
            wc = wc * pt[idx_t, c]
        return wc

    def _bounded_global(
        self,
        s2_s: torch.Tensor,
        tr_s: torch.Tensor,
        s2_t: torch.Tensor,
        tr_t: torch.Tensor,
        s2_st: torch.Tensor,
        H_s: torch.Tensor,
        H_t: torch.Tensor,
        n_s: float,
        n_eff_t: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Class-global term: mixed matrix (Eq. 27) + bounds (Eqs. 34-36).

        Returns ``(loss_c, delta_c)`` where ``delta_c`` is the information
        radius under the mixed-matrix normalization (0 at perfect alignment)
        and ``loss_c`` is either the raw ReLU-clamped radius (V1 behaviour,
        ``bounded_align=False``) or the affine [0, 1] map ``(U_c + delta) /
        (U_c - L_c)`` with class-specific conservative bounds.
        """
        if not self.bounded_align:
            d_c = self._delta_bits_normalized(
                s2_s.detach(), tr_s.detach(), s2_t, tr_t, s2_st
            )
            return d_c, d_c

        tr_mix = (tr_s.detach() + tr_t).clamp_min(self.eps)
        s2_mix = s2_s.detach() + s2_t + 2.0 * s2_st
        h_mix = -torch.log2((s2_mix / (tr_mix * tr_mix)).clamp_min(self.eps))
        delta = h_mix - 0.5 * (H_s.detach() + H_t)

        lower = -math.log2(n_s + n_eff_t)
        upper = 0.5 * (math.log2(n_s) + math.log2(n_eff_t))
        loss_c = (upper + delta) / (upper - lower)
        return loss_c, delta

    def _local_term(
        self,
        K_s: torch.Tensor,
        K_st: torch.Tensor,
        wc: torch.Tensor,
    ) -> torch.Tensor:
        """Personalized-reference distance per target pixel (Eqs. 28-31, 38).

        ``K_st`` is (n_s, n_t) with the source side already detached, so the
        gradient reaches target pixels only (asymmetric anchoring). The
        target self-similarity is the RBF diagonal, exactly 1.
        """
        kts = K_st.transpose(0, 1)
        pi = torch.softmax(kts / self.tau_local, dim=1)
        cross = (pi * kts).sum(dim=1)
        proto = (pi * (pi @ K_s.detach())).sum(dim=1)
        d2 = 1.0 - 2.0 * cross + proto
        l = (0.5 * d2).clamp_min(0.0)
        return (wc * l).sum() / (wc.sum() + self.eps)

    def forward(
        self,
        feat_s: torch.Tensor,
        logits_s: torch.Tensor,
        labels_s: torch.Tensor,
        feat_t: torch.Tensor,
        logits_t: torch.Tensor,
    ):
        """Computes the total DARES v2 loss in float32."""
        loss_seg = F.cross_entropy(logits_s, labels_s, ignore_index=255)
        if self.normalize_seg:
            loss_seg = loss_seg / self._source_bound()

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

            align_terms, ac_terms, loc_terms = [], [], []
            t_feats, t_weights, t_sigmas = [], [], []
            h2s_list, h2t_list, dal_list = [], [], []

            for c in range(C):
                idx_s = self._quota_indices(ys == c)
                idx_t = self._quota_indices(pl_t == c)
                if idx_s is None or idx_t is None:
                    continue

                xs = fs[idx_s]
                xt = ft[idx_t]
                wc = self._class_weights(pt, w_t, idx_t, c)
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

                H_s = self._h2_bits(s2_s, tr_s)
                H_t = self._h2_bits(s2_t, tr_t)

                wv = wc.detach()
                n_eff_t = float(
                    (
                        wv.sum() ** 2 / ((wv * wv).sum() + self.eps)
                    ).clamp(1.0, float(wv.numel()))
                )
                d_c, delta_c = self._bounded_global(
                    s2_s, tr_s, s2_t, tr_t, s2_st, H_s, H_t, float(xs.shape[0]), n_eff_t
                )

                floor_s = F.relu(H_s.new_tensor(self.eta_floor) - H_s)
                floor_t = F.relu(H_s.detach() - self.entropy_gap - H_t)

                align_terms.append(d_c)
                ac_terms.append(floor_s + floor_t)
                if self.lambda_local > 0.0:
                    loc_terms.append(self._local_term(K_s, K_st, wc))
                t_feats.append(xt)
                t_weights.append(wc)
                t_sigmas.append(sig)
                h2s_list.append(H_s.detach())
                h2t_list.append(H_t.detach())
                dal_list.append(delta_c.detach())

            loss_align = torch.stack(align_terms).mean() if align_terms else zero
            loss_ac = torch.stack(ac_terms).mean() if ac_terms else zero
            loss_loc = torch.stack(loc_terms).mean() if loc_terms else zero

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
                ptr_joint = 0.25 * (
                    ptr_diag[:, None] + ptr_diag[None, :] + 2.0 * ptr_cross
                )

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

            loss_aux = (
                self.lambda_align * loss_align
                + self.beta * loss_ac
                + self.gamma * loss_rep
                + self.lambda_local * loss_loc
            )

        total = loss_seg + float(self.lambda_eff) * loss_aux + self.lambda_em * loss_em

        parts = {
            "total": total,
            "loss_seg": loss_seg,
            "loss_aux": loss_aux,
            "loss_align": loss_align.detach(),
            "loss_local": loss_loc.detach(),
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
