"""DACS augmentations and helpers: cross-domain mixed sampling.

DACS (Tranheden et al., WACV 2021) is a self-training UDA method for semantic
segmentation. Each iteration:

1. Color-jitters both the source and target batches, and applies a Gaussian
   blur to the target batch (smoothing away pseudo-label noise).
2. Pseudo-labels the target batch with the model's own predictions, keeping
   only pixels whose top-1 confidence exceeds a threshold.
3. Class-mixes the source and target images: half of the classes present in
   the *source* ground-truth are selected, their pixels are pasted onto the
   target image, and the labels are mixed the same way (source GT over the
   pasted region, target pseudo-label elsewhere).
4. Trains on ``CE(source) + lambda * CE(mixed)`` where ``lambda`` adaptively
   tracks the proportion of confident target pixels.

All helpers operate on ``(B, C, H, W)`` float32 image tensors and ``(B, H, W)``
int64 label tensors.
"""

import torch
import torch.nn.functional as F
from torchvision.transforms import functional as tv_f


def color_jitter(
    image: torch.Tensor,
    brightness: float = 0.5,
    contrast: float = 0.5,
    saturation: float = 0.5,
    hue: float = 0.25,
) -> torch.Tensor:
    """Applies a random (batch-shared) color jitter to an image tensor.

    Brightness, contrast and saturation are applied channel-agnostically, so
    they work for arbitrary input channels (the DARES 4-band B2/B3/B4/B8
    imagery included). Hue rotation is only defined for 3-channel RGB and is
    silently skipped otherwise.

    Args:
        image (torch.Tensor): Input ``(B, C, H, W)`` float tensor.
        brightness (float): Max brightness factor delta (``[1-b, 1+b]``).
        contrast (float): Max contrast factor delta.
        saturation (float): Max saturation factor delta.
        hue (float): Max hue rotation (radians); only for ``C == 3``.

    Returns:
        torch.Tensor: Jittered tensor of the same shape / dtype / device.
    """
    if brightness > 0.0:
        b = 1.0 + (torch.rand(1, device=image.device) * 2.0 - 1.0) * brightness
        image = image * b

    if contrast > 0.0:
        c = 1.0 + (torch.rand(1, device=image.device) * 2.0 - 1.0) * contrast
        mean = image.mean(dim=(2, 3), keepdim=True)
        image = (image - mean) * c + mean

    if saturation > 0.0:
        s = 1.0 + (torch.rand(1, device=image.device) * 2.0 - 1.0) * saturation
        gray = image.mean(dim=1, keepdim=True)
        image = (image - gray) * s + gray

    if hue > 0.0 and image.shape[1] == 3:
        h = (torch.rand(1, device=image.device) * 2.0 - 1.0) * hue
        image = tv_f.adjust_hue(image, h.item())

    return image


def gaussian_blur(
    image: torch.Tensor,
    kernel_size: int = 5,
    sigma: tuple[float, float] = (0.1, 2.0),
) -> torch.Tensor:
    """Applies a Gaussian blur with a randomly sampled bandwidth.

    Args:
        image (torch.Tensor): Input ``(B, C, H, W)`` float tensor.
        kernel_size (int): Gaussian kernel size (odd).
        sigma (tuple[float, float]): ``(min, max)`` bandwidth range.

    Returns:
        torch.Tensor: Blurred tensor of the same shape / dtype / device.
    """
    sigma_val = float(
        torch.empty(1, device=image.device).uniform_(*sigma).item()
    )
    return tv_f.gaussian_blur(
        image, kernel_size=[kernel_size, kernel_size], sigma=[sigma_val, sigma_val]
    )


@torch.no_grad()
def pseudo_label(
    logits: torch.Tensor, threshold: float = 0.968, ignore_index: int = 255
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes thresholded pseudo-labels from class logits.

    Args:
        logits (torch.Tensor): Logits of shape ``(B, C, H, W)``.
        threshold (float): Confidence below which pixels are marked ``ignore``.
        ignore_index (int): Label value assigned to low-confidence pixels.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(pseudo, confident)`` where
            ``pseudo`` is ``(B, H, W)`` int64 with ``ignore_index`` at
            low-confidence pixels, and ``confident`` is a ``(B, H, W)`` boolean
            mask of the kept pixels.
    """
    prob = F.softmax(logits.float(), dim=1)
    confidence, pseudo = prob.max(dim=1)
    confident = confidence > threshold
    pseudo = torch.where(
        confident,
        pseudo,
        torch.full_like(pseudo, ignore_index),
    )
    return pseudo, confident


def class_mix(
    source_img: torch.Tensor,
    source_label: torch.Tensor,
    target_img: torch.Tensor,
    target_pseudo: torch.Tensor,
    mix_ratio: float = 0.5,
    ignore_index: int = 255,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Class-mixes source and target images across domains.

    For each sample, a random ``mix_ratio`` fraction of the classes present in
    the source ground-truth is selected; those source pixels are pasted onto
    the target image, and the labels are mixed identically.

    Args:
        source_img (torch.Tensor): Source images ``(B, C, H, W)``.
        source_label (torch.Tensor): Source ground-truth ``(B, H, W)`` int64.
        target_img (torch.Tensor): Target images ``(B, C, H, W)``.
        target_pseudo (torch.Tensor): Target pseudo-labels ``(B, H, W)`` int64.
        mix_ratio (float): Fraction of source classes pasted (default ``0.5``).
        ignore_index (int): Ignore label excluded from class selection.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(mixed_img, mixed_label)``.
    """
    batch, _, height, width = source_img.shape
    device = source_img.device
    mixed_img = torch.empty_like(source_img)
    mixed_label = torch.empty_like(source_label)

    for i in range(batch):
        classes = torch.unique(source_label[i])
        classes = classes[classes != ignore_index]
        n_classes = classes.numel()
        if n_classes == 0:
            mixed_img[i] = target_img[i]
            mixed_label[i] = target_pseudo[i]
            continue

        keep = max(1, int(round(mix_ratio * n_classes)))
        selected = classes[torch.randperm(n_classes, device=device)[:keep]]
        mask = torch.isin(source_label[i], selected)  # (H, W) bool
        mixed_img[i] = torch.where(mask, source_img[i], target_img[i])
        mixed_label[i] = torch.where(mask, source_label[i], target_pseudo[i])

    return mixed_img, mixed_label
