"""FDA (Fourier Domain Adaptation) augmentation and entropy regularization.

FDA (Yang & Soatto, CVPR 2020) aligns domains without any adversarial
training: the low-frequency amplitude of the FFT of a source image is swapped
with that of a randomly sampled target image, keeping the source phase. The
resulting "source in target style" images are trained on with cross-entropy,
complemented by a Charbonnier-weighted entropy minimization term on the
target domain.
"""

import torch
import torch.nn.functional as F

_EPS = 1e-8


def fourier_domain_adaptation(
    source_img: torch.Tensor,
    target_img: torch.Tensor,
    beta: float = 0.09,
) -> torch.Tensor:
    """Swaps the low-frequency amplitude of the source with the target's.

    Args:
        source_img (torch.Tensor): Source images ``(B, C, H, W)`` float.
        target_img (torch.Tensor): Target images ``(B, C, H, W)`` float.
        beta (float): Fraction ``(0, 1]`` of the (centered) spectrum whose
            amplitude is swapped; larger ``beta`` transfers more target style
            but can introduce artifacts (paper recommends ``beta <= 0.15``).

    Returns:
        torch.Tensor: Adapted source images ``(B, C, H, W)`` with the target's
            low-frequency amplitude but the source's phase.
    """
    _, _, height, width = source_img.shape
    fft_src = torch.fft.fftn(source_img, dim=(-2, -1))
    fft_tgt = torch.fft.fftn(target_img, dim=(-2, -1))

    amp_src = torch.fft.fftshift(torch.abs(fft_src), dim=(-2, -1))
    amp_tgt = torch.fft.fftshift(torch.abs(fft_tgt), dim=(-2, -1))
    phase_src = torch.angle(fft_src)

    half_h = int(beta * height) // 2
    half_w = int(beta * width) // 2
    cy, cx = height // 2, width // 2
    mask = torch.zeros((height, width), device=source_img.device)
    mask[cy - half_h : cy + half_h, cx - half_w : cx + half_w] = 1.0

    amp_new = (1.0 - mask) * amp_src + mask * amp_tgt
    amp_new = torch.fft.ifftshift(amp_new, dim=(-2, -1))
    fft_new = amp_new * torch.exp(1j * phase_src)
    return torch.fft.ifftn(fft_new, dim=(-2, -1)).real


def charbonnier_entropy(
    logits: torch.Tensor, eta: float = 2.0, eps: float = 0.001
) -> torch.Tensor:
    """Charbonnier-penalized mean prediction entropy over a batch.

    ``rho(x) = (x^2 + eps^2)^eta`` penalizes high-entropy predictions more
    strongly than low-entropy ones for ``eta > 0.5`` (FDA Eq. 5).

    Args:
        logits (torch.Tensor): Logits of shape ``(B, C, H, W)``.
        eta (float): Charbonnier exponent (paper uses ``2.0``).
        eps (float): Charbonnier shift (paper uses ``0.001``).

    Returns:
        torch.Tensor: Scalar entropy penalty (mean over batch and pixels).
    """
    prob = F.softmax(logits.float(), dim=1)
    entropy = -(prob * (prob + _EPS).log()).sum(dim=1)  # (B, H, W), nats
    return (entropy ** 2 + eps ** 2).pow(eta).mean()
