"""DARES alpha-Renyi class-conditional alignment loss.

Implements the matrix-based order-2 Renyi mutual information estimator used by
the DARES methodology (Section 3.3 of the paper). Source features are sampled
per class (capped at ``n_max``), target features are selected by the
Spatially-Stratified Confidence-Guided Sampling operator ``Phi_c``, and the
alignment is computed from Gaussian Gram matrices built with a median-heuristic
kernel bandwidth.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RenyiLoss(nn.Module):
    """Class-conditional alpha-Renyi mutual-information alignment loss.

    For every class ``c`` present in both domains the loss builds the Gram
    matrices ``K_s``, ``Ktilde_t`` and ``K_st`` from sampled feature vectors
    and estimates the mutual information

    ``I2 = 0.5 * (H2(K_s) + H2(Ktilde_t)) - H2(Kmix)``

    with ``H2(A) = -log(tr((A/tr(A))^T (A/tr(A))))`` and ``Kmix`` the block
    matrix of source / target kernels. The returned scalar is the mean of
    ``I2`` over valid classes and is meant to be *maximized*.

    Args:
        num_classes (int): Number of semantic classes.
        tau (float): Confidence threshold for target pseudo-label selection.
        n_max (int): Maximum number of samples per class per mini-batch.
        sigma (float | str): Gaussian kernel bandwidth; ``"auto"`` selects it
            via the median heuristic over the combined source + target
            features.
        alpha (int): Renyi entropy order (the estimator uses ``alpha = 2``).
        grid_size (int): Number of spatial cells per side used by the
            stratified sampling operator ``Phi_c``.
    """

    def __init__(
        self,
        num_classes: int,
        tau: float = 0.85,
        n_max: int = 1024,
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
        """Pairwise squared Euclidean distances between two feature sets.

        Args:
            x (torch.Tensor): Features of shape ``(N, D)``.
            y (torch.Tensor): Features of shape ``(M, D)``.

        Returns:
            torch.Tensor: Distance matrix of shape ``(N, M)``.
        """
        x_norm = (x ** 2).sum(dim=1, keepdim=True)
        y_norm = (y ** 2).sum(dim=1, keepdim=True)
        dist_sq = x_norm + y_norm.t() - 2.0 * torch.mm(x, y.t())
        return torch.clamp(dist_sq, min=0.0)

    def _rbf_kernel(
        self, x: torch.Tensor, y: torch.Tensor, sigma: float | torch.Tensor
    ) -> torch.Tensor:
        """Evaluates the Gaussian kernel ``k(x, y) = exp(-||x-y||^2 / 2 sigma^2)``.

        Args:
            x (torch.Tensor): Features of shape ``(N, D)``.
            y (torch.Tensor): Features of shape ``(M, D)``.
            sigma (float | torch.Tensor): Kernel bandwidth.

        Returns:
            torch.Tensor: Gram matrix of shape ``(N, M)``.
        """
        dist_sq = self._squared_euclidean_dist(x, y)
        return torch.exp(-dist_sq / (2.0 * sigma ** 2))

    def _compute_sigma(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Median-heuristic bandwidth over the combined feature set.

        The median is computed over the pairwise squared Euclidean distances
        (excluding the zero diagonal) of ``cat([x, y])``, so the bandwidth
        adapts to the local scale of both domains.

        Args:
            x (torch.Tensor): Features of shape ``(N, D)``.
            y (torch.Tensor): Features of shape ``(M, D)``.

        Returns:
            torch.Tensor: Scalar bandwidth ``sqrt(median) + eps``.
        """
        combined = torch.cat([x, y], dim=0)
        dist_sq = self._squared_euclidean_dist(combined, combined)
        n = combined.shape[0]
        non_diag = dist_sq[
            ~torch.eye(n, dtype=torch.bool, device=dist_sq.device)
        ]
        return torch.sqrt(torch.median(non_diag)) + self.eps

    def _h2(self, A: torch.Tensor) -> torch.Tensor:
        """Matrix-based order-2 Renyi entropy of a Gram matrix.

        ``H2(A) = -log(tr((A / tr(A))^T (A / tr(A))) + eps)`` with an epsilon
        guard on the trace normalization.

        Args:
            A (torch.Tensor): Positive Gram matrix of shape ``(N, N)``.

        Returns:
            torch.Tensor: Scalar entropy.
        """
        tr_A = torch.trace(A) + self.eps
        A_tilde = A / tr_A
        trace_prod = torch.trace(A_tilde.t() @ A_tilde) + self.eps
        return -torch.log(trace_prod)

    def _stratified_target_sampling(
        self,
        features_t: torch.Tensor,
        probs: torch.Tensor,
        pseudo: torch.Tensor,
        conf: torch.Tensor,
    ) -> dict[int, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
        """Spatially-Stratified Confidence-Guided sampling operator ``Phi_c``.

        Grid-stratifies each ``(H, W)`` prediction map and, for every class and
        cell, keeps the single most-confident pixel whose pseudo-label matches
        the class and whose confidence exceeds ``tau``. This guarantees spatial
        diversity while bounding the candidate count per class.

        Args:
            features_t (torch.Tensor): Target features of shape ``(B, D, H, W)``.
            probs (torch.Tensor): Target softmax probabilities ``(B, C, H, W)``.
            pseudo (torch.Tensor): Target pseudo-labels ``(B, H, W)``.
            conf (torch.Tensor): Target per-pixel confidence ``(B, H, W)``.

        Returns:
            dict[int, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]:
                Per-class lists of ``(feature, confidence, probability)``
                candidate triples.
        """
        B, D, H, W = features_t.shape
        feats_t_flat = features_t.permute(0, 2, 3, 1).reshape(-1, D)
        probs_t_flat = probs.permute(0, 2, 3, 1).reshape(-1, self.num_classes)

        rows = torch.linspace(0, H, self.grid_size + 1, device=features_t.device)
        cols = torch.linspace(0, W, self.grid_size + 1, device=features_t.device)

        candidates: dict[int, list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {
            c: [] for c in range(self.num_classes)
        }

        for b in range(B):
            for i in range(self.grid_size):
                r0 = int(round(rows[i].item()))
                r1 = int(round(rows[i + 1].item()))
                for j in range(self.grid_size):
                    c0 = int(round(cols[j].item()))
                    c1 = int(round(cols[j + 1].item()))
                    cell_pseudo = pseudo[b, r0:r1, c0:c1]
                    cell_conf = conf[b, r0:r1, c0:c1]
                    for c in range(self.num_classes):
                        mask = (cell_pseudo == c) & (cell_conf > self.tau)
                        if not bool(mask.any()):
                            continue
                        idxs = torch.nonzero(mask, as_tuple=False)
                        confs = cell_conf[idxs[:, 0], idxs[:, 1]]
                        best = int(torch.argmax(confs).item())
                        rr = int(idxs[best, 0].item()) + r0
                        cc = int(idxs[best, 1].item()) + c0
                        p_global = b * H * W + rr * W + cc
                        candidates[c].append(
                            (
                                feats_t_flat[p_global],
                                confs[best],
                                probs_t_flat[p_global],
                            )
                        )
        return candidates

    def forward(
        self,
        features_s: torch.Tensor,
        labels_s: torch.Tensor,
        features_t: torch.Tensor,
        logits_t: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Computes the class-conditional Renyi alignment between the domains.

        Args:
            features_s (torch.Tensor): Source features ``(B, D, H, W)``.
            labels_s (torch.Tensor): Source masks ``(B, H, W)`` (int64).
            features_t (torch.Tensor): Target features ``(B, D, H, W)``.
            logits_t (torch.Tensor): Target logits ``(B, C, H, W)``.

        Returns:
            tuple[torch.Tensor, dict[str, float]]: The mean alignment
                ``I2tilde`` over valid classes (a differentiable scalar) and a
                metrics dict with ``loss_renyi``, ``valid_classes``,
                ``n_source_sampled`` and ``n_target_sampled``.
        """
        device = features_s.device
        B, D, H, W = features_s.shape

        feats_s_flat = features_s.permute(0, 2, 3, 1).reshape(-1, D)
        labels_flat = labels_s.reshape(-1)

        source_by_class: dict[int, torch.Tensor] = {}
        for c in range(self.num_classes):
            F_s_c = feats_s_flat[labels_flat == c]
            if F_s_c.shape[0] == 0:
                continue
            if F_s_c.shape[0] > self.n_max:
                idx = torch.randperm(F_s_c.shape[0], device=device)[: self.n_max]
                F_s_c = F_s_c[idx]
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
            if F_s_c is None or not candidates[c]:
                continue
            F_t_c = torch.stack([t[0] for t in candidates[c]])
            confs_c = torch.stack([t[1] for t in candidates[c]])
            probs_c = torch.stack([t[2] for t in candidates[c]])

            if F_t_c.shape[0] > self.n_max:
                keep = torch.argsort(confs_c, descending=True)[: self.n_max]
                F_t_c = F_t_c[keep]
                probs_c = probs_c[keep]

            n_s, n_t = F_s_c.shape[0], F_t_c.shape[0]
            if n_s < 2 or n_t < 2:
                continue

            if self.sigma == "auto":
                sigma = self._compute_sigma(F_s_c, F_t_c)
            else:
                sigma = float(self.sigma)

            K_s = self._rbf_kernel(F_s_c, F_s_c, sigma)
            K_t = self._rbf_kernel(F_t_c, F_t_c, sigma)
            K_st = self._rbf_kernel(F_s_c, F_t_c, sigma)

            sq_sum = torch.clamp((probs_c ** 2).sum(dim=1), min=self.eps)
            entropy = -torch.log(sq_sum)
            w = 1.0 - entropy / math.log(self.num_classes)
            W = torch.outer(w, w)
            Kt_tilde = K_t * W

            Kmix = torch.cat(
                [
                    torch.cat([K_s, K_st], dim=1),
                    torch.cat([K_st.t(), Kt_tilde], dim=1),
                ],
                dim=0,
            )

            h_s = self._h2(K_s)
            h_t = self._h2(Kt_tilde)
            h_mix = self._h2(Kmix)
            i2 = 0.5 * (h_s + h_t) - h_mix

            losses.append(i2)
            valid_classes += 1
            n_source_sampled += int(n_s)
            n_target_sampled += int(n_t)

        if valid_classes == 0:
            metrics = {
                "loss_renyi": 0.0,
                "valid_classes": 0,
                "n_source_sampled": int(n_source_sampled),
                "n_target_sampled": int(n_target_sampled),
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
