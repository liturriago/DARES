class CREDALoss(nn.Module):
    def __init__(self, sigma='auto', lambda_creda=1.0, use_entropy_weighting=True):
        super(CREDALoss, self).__init__()
        self.lambda_creda = lambda_creda
        self.sigma = sigma
        self.use_entropy_weighting = use_entropy_weighting
    def _squared_euclidean_dist(self, x, y):
        # x: (N, D), y: (M, D)
        x_norm = (x ** 2).sum(dim=1).view(-1, 1)  # (N, 1)
        y_norm = (y ** 2).sum(dim=1).view(1, -1)  # (1, M)
        dist_sq = x_norm + y_norm - 2.0 * (x @ y.T)
        return torch.clamp(dist_sq, min=0.0)

    def _compute_sigma(self, x, y):
        combined = torch.cat([x, y], dim=0)
        dist_sq = self._squared_euclidean_dist(combined, combined)
        # Elimina diagonal (distancia 0) para evitar sesgo
        non_diag = dist_sq[~torch.eye(dist_sq.shape[0], dtype=bool, device=dist_sq.device)]
        return torch.sqrt(torch.median(non_diag) + 1e-6)

    def _gaussian_kernel(self, x, y, sigma_val):
        dist_sq = self._squared_euclidean_dist(x, y)
        return torch.exp(-dist_sq / (2 * sigma_val ** 2))


    def _renyi_entropy_order_2(self, K):
        tr_K = torch.trace(K) + 1e-6
        K_norm = K / tr_K
        return -torch.log(torch.trace(K_norm.T @ K_norm) + 1e-6)

    def _mix_kernel_concat(self, K_s, K_t, K_st):
        return torch.cat([
            torch.cat([K_s, K_st], dim=1),
            torch.cat([K_st.T, K_t], dim=1)
        ], dim=0)

    def forward(self, f_s, f_t, y_s, g_t, reduction='mean'):
        y_t_pseudo = torch.argmax(g_t, dim=1)
        unique_classes = torch.unique(y_s, sorted=True)
        losses_per_class = []
        valid_class_count = 0

        if self.use_entropy_weighting:
            squared_sum = torch.sum(g_t ** 2, dim=1)
            entropy = -torch.log(squared_sum + 1e-6)
            target_weights = 1.0 - entropy / torch.log(torch.tensor(float(g_t.shape[1]), device=g_t.device))
        else:
            target_weights = None

        for c in unique_classes:
            f_s_c = f_s[y_s == c]
            f_t_c = f_t[y_t_pseudo == c]

            if f_s_c.shape[0] == 0 or f_t_c.shape[0] == 0:
                continue

            sigma_val = self._compute_sigma(f_s_c, f_t_c) if self.sigma == 'auto' else self.sigma

            K_s_c = self._gaussian_kernel(f_s_c, f_s_c, sigma_val)
            K_t_c = self._gaussian_kernel(f_t_c, f_t_c, sigma_val)
            K_st_c = self._gaussian_kernel(f_s_c, f_t_c, sigma_val)

            if self.use_entropy_weighting:
                weights_c = target_weights[y_t_pseudo == c]
                W_c = torch.outer(weights_c, weights_c)
                K_t_c = K_t_c * W_c

            K_mix_c = self._mix_kernel_concat(K_s_c, K_t_c, K_st_c)

            h_s = self._renyi_entropy_order_2(K_s_c)
            h_t = self._renyi_entropy_order_2(K_t_c)
            h_mix = self._renyi_entropy_order_2(K_mix_c)
            creda_c = h_mix - 0.5 * (h_s + h_t)
            losses_per_class.append(creda_c)
            valid_class_count += 1

        if valid_class_count == 0:
            return torch.tensor(0.0, device=f_s.device)

        losses = torch.stack(losses_per_class)

        if reduction == 'none':
            return self.lambda_creda * losses
        else:
            return self.lambda_creda * losses.mean()