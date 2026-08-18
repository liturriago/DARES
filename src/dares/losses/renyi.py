"""DARES alpha-Renyi class-conditional alignment loss (Corrected & Optimized).

Implements the matrix-based order-2 Renyi mutual information estimator.
Fixes kernel cross-weighting in RKHS, detaches sigma estimation, balances
sample sizes per class, and optimizes trace computations to O(N^2).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RenyiLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        tau: float = 0.85,
        n_max: int = 512,
        sigma: float | str = "auto",
        alpha: int = 2,
        grid_size: int = 8,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.tau = float(tau)
        self.n_max = int(n_max)
        self.sigma = sigma
        self.alpha = int(alpha)
        self.grid_size = int(grid_size)
        self.eps = 1e-8

    def _squared_euclidean_dist(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        x_norm = (x ** 2).sum(dim=1, keepdim=True)
        y_norm = (y ** 2).sum(dim=1, keepdim=True)
        dist_sq = x_norm + y_norm.t() - 2.0 * torch.mm(x, y.t())
        return torch.clamp(dist_sq, min=0.0)

    def _rbf_kernel(
        self, x: torch.Tensor, y: torch.Tensor, sigma: float | torch.Tensor
    ) -> torch.Tensor:
        dist_sq = self._squared_euclidean_dist(x, y)
        return torch.exp(-dist_sq / (2.0 * (sigma ** 2) + self.eps))

    def _compute_sigma(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            combined = torch.cat([x, y], dim=0)
            dist_sq = self._squared_euclidean_dist(combined, combined)
            n = combined.shape[0]
            non_diag = dist_sq[
                ~torch.eye(n, dtype=torch.bool, device=dist_sq.device)
            ]
            if non_diag.numel() == 0:
                return torch.tensor(1.0, device=x.device)
            med = torch.median(non_diag)
            # Evitar sigma=0 si las características colapsan
            sigma = torch.sqrt(torch.clamp(med, min=1e-4)) + self.eps
            return sigma.detach()

    def _h2(self, A: torch.Tensor) -> torch.Tensor:
        """Calcula H2(A) = -log(tr(A_tilde^2)) en O(N^2) sin GEMM completo."""
        tr_A = torch.trace(A) + self.eps
        A_tilde = A / tr_A
        # tr(A_tilde^T @ A_tilde) es equivalente a la suma de cuadrados de todos los elementos
        trace_prod = (A_tilde ** 2).sum() + self.eps
        return -torch.log(trace_prod)

    def _stratified_target_sampling(
        self,
        features_t: torch.Tensor,
        probs: torch.Tensor,
        pseudo: torch.Tensor,
        conf: torch.Tensor,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        B, D, H, W = features_t.shape
        num_cells = min(self.grid_size, H, W)
        while num_cells > 1 and (H % num_cells != 0 or W % num_cells != 0):
            num_cells -= 1
        if num_cells < 2:
            return {}

        cell_h, cell_w = H // num_cells, W // num_cells
        gg = cell_h * cell_w

        pseudo_c = pseudo.view(B, num_cells, cell_h, num_cells, cell_w)
        conf_c = conf.view(B, num_cells, cell_h, num_cells, cell_w)
        pseudo_p = pseudo_c.permute(0, 1, 3, 2, 4)
        conf_p = conf_c.permute(0, 1, 3, 2, 4)

        feats_c = features_t.view(B, D, num_cells, cell_h, num_cells, cell_w)
        feats_p = feats_c.permute(0, 1, 2, 4, 3, 5)
        feats_flat = feats_p.reshape(B, D, num_cells, num_cells, gg)

        probs_c = probs.view(
            B, self.num_classes, num_cells, cell_h, num_cells, cell_w
        )
        probs_p = probs_c.permute(0, 1, 2, 4, 3, 5)
        probs_flat = probs_p.reshape(
            B, self.num_classes, num_cells, num_cells, gg
        )

        candidates: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for c in range(self.num_classes):
            mask = (pseudo_p == c) & (conf_p > self.tau)
            valid = torch.where(mask, conf_p, torch.tensor(-1.0, device=conf.device))
            valid_flat = valid.reshape(B, num_cells, num_cells, gg)
            best = valid_flat.argmax(dim=-1)
            best_conf = valid_flat.gather(-1, best.unsqueeze(-1)).squeeze(-1)
            cell_ok = best_conf > 0.0

            if not bool(cell_ok.any()):
                continue

            idx = best.unsqueeze(1).unsqueeze(-1)
            best_feats = feats_flat.gather(
                -1, idx.expand(B, D, num_cells, num_cells, 1)
            ).squeeze(-1)
            best_probs = probs_flat.gather(
                -1, idx.expand(B, self.num_classes, num_cells, num_cells, 1)
            ).squeeze(-1)

            feats_f = best_feats.permute(0, 2, 3, 1)[cell_ok]
            probs_f = best_probs.permute(0, 2, 3, 1)[cell_ok]
            confs_f = best_conf[cell_ok]

            if feats_f.shape[0] > self.n_max:
                keep = torch.argsort(confs_f, descending=True)[: self.n_max]
                feats_f = feats_f[keep]
                probs_f = probs_f[keep]

            candidates[c] = (feats_f, probs_f)
        return candidates

    def forward(
        self,
        features_s: torch.Tensor,
        labels_s: torch.Tensor,
        features_t: torch.Tensor,
        logits_t: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        device = features_s.device
        B, D, H, W = features_s.shape

        features_s = features_s.float()
        features_t = features_t.float()
        labels_s = labels_s.long()
        logits_t = logits_t.float()

        feats_s_flat = features_s.permute(0, 2, 3, 1).reshape(-1, D)
        labels_flat = labels_s.reshape(-1)

        source_by_class: dict[int, torch.Tensor] = {}
        for c in range(self.num_classes):
            mask_c = labels_flat == c
            if not mask_c.any():
                continue
            F_s_c = feats_s_flat[mask_c]
            source_by_class[c] = F_s_c

        probs = F.softmax(logits_t, dim=1)
        pseudo = torch.argmax(probs, dim=1)
        conf = torch.amax(probs, dim=1)
        candidates = self._stratified_target_sampling(
            features_t, probs, pseudo, conf
        )

        losses: list[torch.Tensor] = []
        valid_classes = 0
        n_source_sampled = 0
        n_target_sampled = 0

        for c in range(self.num_classes):
            F_s_c = source_by_class.get(c)
            if F_s_c is None or c not in candidates:
                continue
            F_t_c, probs_c = candidates[c]

            # 1. Balanceo estricto de muestras para eliminar sesgo de soporte
            n_pair = min(F_s_c.shape[0], F_t_c.shape[0], self.n_max)
            if n_pair < 4:
                continue

            if F_s_c.shape[0] > n_pair:
                idx_s = torch.randperm(F_s_c.shape[0], device=device)[:n_pair]
                F_s_c = F_s_c[idx_s]
            if F_t_c.shape[0] > n_pair:
                F_t_c = F_t_c[:n_pair]
                probs_c = probs_c[:n_pair]

            # 2. Estimación de ancho de banda sin gradiente
            if self.sigma == "auto":
                sigma = self._compute_sigma(F_s_c, F_t_c)
            else:
                sigma = float(self.sigma)

            # 3. Ponderación por entropía de predicción en Target
            sq_sum = torch.clamp((probs_c ** 2).sum(dim=1), min=self.eps)
            entropy = -torch.log(sq_sum)
            max_ent = math.log(self.num_classes)
            w = torch.clamp(1.0 - (entropy / (max_ent + self.eps)), min=0.0, max=1.0)
            
            # Matrices de Gram
            K_s = self._rbf_kernel(F_s_c, F_s_c, sigma)
            K_t = self._rbf_kernel(F_t_c, F_t_c, sigma)
            K_st = self._rbf_kernel(F_s_c, F_t_c, sigma)

            # 4. Modulación consistente en RKHS para Target y Cross-Kernel
            W = torch.outer(w, w)
            Kt_tilde = K_t * W
            Kst_tilde = K_st * w.unsqueeze(0)  # Ponderación correcta de columnas

            Kmix = torch.cat(
                [
                    torch.cat([K_s, Kst_tilde], dim=1),
                    torch.cat([Kst_tilde.t(), Kt_tilde], dim=1),
                ],
                dim=0,
            )

            h_s = self._h2(K_s)
            h_t = self._h2(Kt_tilde)
            h_mix = self._h2(Kmix)
            
            # Información mutua de Rényi
            i2 = 0.5 * (h_s + h_t) - h_mix

            losses.append(i2)
            valid_classes += 1
            n_source_sampled += int(n_pair)
            n_target_sampled += int(n_pair)

        if valid_classes == 0:
            metrics = {
                "loss_renyi": 0.0,
                "valid_classes": 0,
                "n_source_sampled": 0,
                "n_target_sampled": 0,
            }
            return torch.tensor(0.0, device=device), metrics

        alignment = torch.stack(losses).mean()
        metrics = {
            "loss_renyi": float(alignment.detach().item()),
            "valid_classes": int(valid_classes),
            "n_source_sampled": int(n_source_sampled),
            "n_target_sampled": int(n_target_sampled),
        }
        return alignment, metrics