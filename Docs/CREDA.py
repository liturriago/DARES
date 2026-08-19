import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Union
from banana_creda.config import TrainConfig

class CREDALoss(nn.Module):
    """Advanced Implementation of CREDA (Class-Regularized Entropy Domain Adaptation).

    This loss aligns source and target feature distributions using Rényi 
    mutual information computed over class-conditional kernel matrices.

    Attributes:
        config (TrainConfig): Training configuration.
        num_classes (int): Number of target classes.
        eps (float): Small constant for numerical stability.
        lambda_creda (float): Weight for the CREDA alignment loss.
        log2 (torch.Tensor): Precomputed log(2) for entropy normalization.
    """
    def __init__(self, config: TrainConfig, num_classes: int):
        """Initializes the CREDALoss module.

        Args:
            config (TrainConfig): Configuration object containing hyperparameters.
            num_classes (int): Number of output classes for pseudo-labeling.
        """
        super(CREDALoss, self).__init__()
        self.config = config
        self.num_classes = num_classes
        self.eps = 1e-8
        self.lambda_creda = config.lambda_creda if config.lambda_creda is not None else 0.0
        self.register_buffer("log2", torch.log(torch.tensor(2.0)))

    def _compute_sigma(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the dynamic bandwidth (sigma) using the median heuristic.

        Args:
            x (torch.Tensor): Feature matrix from domain A.
            y (torch.Tensor): Feature matrix from domain B.

        Returns:
            torch.Tensor: Scalar sigma value.
        """
        combined = torch.cat([x, y], dim=0)
        dist_sq = torch.cdist(combined, combined, p=2) ** 2
        triu_indices = torch.triu_indices(dist_sq.size(0), dist_sq.size(0), offset=1)
        non_diag = dist_sq[triu_indices[0], triu_indices[1]]
        return torch.sqrt(torch.median(non_diag) + 1e-6)

    def _rbf_kernel(self, x: torch.Tensor, y: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Computes the Gaussian RBF kernel matrix.

        Args:
            x (torch.Tensor): First feature matrix.
            y (torch.Tensor): Second feature matrix.
            sigma (torch.Tensor): Kernel bandwidth.

        Returns:
            torch.Tensor: Kernel matrix of size [len(x), len(y)].
        """
        dist_sq = torch.cdist(x, y, p=2) ** 2
        return torch.exp(-dist_sq / (2 * sigma ** 2 + self.eps))

    def _renyi_entropy_order_2(self, K: torch.Tensor) -> torch.Tensor:
        """Calculates the Rényi entropy of order 2 from a kernel matrix.

        Formula: H2(A) = -log2(tr(A^2)) where A is the trace-normalized kernel matrix.

        Args:
            K (torch.Tensor): Kernel matrix.

        Returns:
            torch.Tensor: Scalar entropy value in bits.
        """
        if K.size(0) == 0:
            return torch.tensor(0.0, device=K.device)
        # Normalization by trace to obtain density matrix
        A = K / (torch.trace(K) + self.eps)
        # Information potential (tr(A @ A))
        info_potential = torch.trace(A @ A)
        return -torch.log(info_potential + self.eps) / self.log2

    def forward(
        self,
        features_s: torch.Tensor,
        logits_s: torch.Tensor,
        labels_s: torch.Tensor,
        features_t: torch.Tensor,
        logits_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Performs the forward pass to compute supervised and CREDA alignment losses.

        Args:
            features_s (torch.Tensor): Deep features of source samples.
            logits_s (torch.Tensor): Logits of source samples.
            labels_s (torch.Tensor): Ground truth labels for source samples.
            features_t (torch.Tensor): Deep features of target samples.
            logits_t (torch.Tensor): Logits of target samples.

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: 
                Total loss and a dictionary with individual loss components.
        """
        # 1. Supervised Classification Loss (Source)
        loss_cls = F.cross_entropy(logits_s, labels_s)

        # Probabilities needed for pseudo-labels and uncertainty weighting
        probs_t = F.softmax(logits_t, dim=1)

        # 2. CREDA Alignment
        loss_creda = torch.tensor(0.0, device=features_s.device)
        pseudo_labels_t = torch.argmax(probs_t.detach(), dim=1)
        
        # --- UNCERTAINTY WEIGHTING (RENYI) ---
        uncertainty_weights = None
        if self.config.use_uncertainty:
            # H2 of probability vector: -log2(sum(p^2))
            prob_sq_sum = torch.sum(probs_t ** 2, dim=1)
            h2_probs = -torch.log(prob_sq_sum + self.eps) / self.log2
            
            # Normalization: H2_max = log2(C)
            h2_max = torch.log(torch.tensor(float(self.num_classes))) / self.log2
            # Confidence weight: 1 - (Entropy normalized)
            uncertainty_weights = 1.0 - (h2_probs / (h2_max + self.eps))

        creda_accum = 0.0
        valid_classes = 0

        for c in range(self.num_classes):
            mask_s = (labels_s == c)
            mask_t = (pseudo_labels_t == c)

            if mask_s.sum() < 2 or mask_t.sum() < 2:
                continue

            f_s_c, f_t_c = features_s[mask_s], features_t[mask_t]
            
            # Adaptive sigma per class
            if self.config.sigma == 'auto':
                sigma_val = self._compute_sigma(f_s_c, f_t_c)
            else:
                sigma_val = torch.tensor(float(self.config.sigma), device=features_s.device)

            K_s = self._rbf_kernel(f_s_c, f_s_c, sigma_val)
            K_t = self._rbf_kernel(f_t_c, f_t_c, sigma_val)
            K_st = self._rbf_kernel(f_s_c, f_t_c, sigma_val)

            # Application of Rényi confidence weights
            if self.config.use_uncertainty:
                w_c = uncertainty_weights[mask_t]
                K_t = K_t * torch.outer(w_c, w_c)

            # Construction of the Joint Matrix (Blocks)
            row1 = torch.cat([K_s, K_st], dim=1)
            row2 = torch.cat([K_st.t(), K_t], dim=1)
            K_mix = torch.cat([row1, row2], dim=0)

            # Calculation of Rényi Mutual Information
            h_s = self._renyi_entropy_order_2(K_s)
            h_t = self._renyi_entropy_order_2(K_t)
            h_mix = self._renyi_entropy_order_2(K_mix)

            creda_accum += (h_mix - 0.5 * (h_s + h_t))
            valid_classes += 1

        if valid_classes > 0:
            loss_creda = creda_accum / valid_classes

        # Final combination with configuration weights
        total_loss = loss_cls + (self.lambda_creda * loss_creda)
        
        return total_loss, {
            "total_loss": total_loss.item(),
            "loss_cls": loss_cls.item(),
            "loss_creda": loss_creda.item(),
        }