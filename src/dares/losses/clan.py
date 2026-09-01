"""CLAN (CAA-Net) losses: category-level output-space adversarial alignment.

Implements the core module of Ruan et al. (2019), "Category-Level Adversaries
for Semantic Domain Adaptation" (CAA-Net / CLAN): a multi-category
discriminator applied in the *output space* of the segmentation network.

Rather than discriminating whole prediction maps, the discriminator takes a
slice of a prediction -- a single class channel -- element-wise multiplied by
the class mask ``M_n = C_n(E(x)) * mask_n`` -- and classifies whether that
slice belongs to the source or target domain.

For the source domain the mask comes from the real labels, so the
discriminator learns the prior knowledge of every category. For the target
domain the mask comes from the model's own pseudo-labels. Source slices are
labeled with their own category ``n`` while target slices are labeled with an
extra ``num_classes`` "target" category. The backbone is then pushed to make
target slices be classified as their pseudo-label category (fooling the
discriminator), closing the source/target output-space gap.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLANDiscriminator(nn.Module):
    """Small CNN classifying a single masked class map of a prediction.

    Input is a ``(B, 1, H, W)`` masked class channel
    (``C_n(E(x)) * mask_n``). The output is a ``(B, C + 1)`` multi-category
    logit vector: the first ``C`` logits correspond to the source classes and
    the last one encodes "target fake" --- the label assigned to every target
    masked slice.

    Args:
        num_classes (int): Number of semantic classes ``C``.
        base (int): Base number of channels of the convolutional stack.
        num_layers (int): Number of stride-2 convolution layers (each halves
            the spatial resolution).
    """

    def __init__(
        self, num_classes: int, base: int = 32, num_layers: int = 3
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        layers: list[nn.Module] = []
        channels = 1
        for _ in range(num_layers):
            layers.append(
                nn.Conv2d(channels, base, kernel_size=4, stride=2, padding=1)
            )
            layers.append(nn.BatchNorm2d(base))
            layers.append(nn.ReLU(inplace=True))
            channels = base
        layers.append(nn.Conv2d(base, base, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        self.convs = nn.Sequential(*layers)
        self.fc = nn.Linear(base, num_classes + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes per-sample multi-category domain logits.

        Args:
            x (torch.Tensor): Masked class map of shape ``(B, 1, H, W)``.

        Returns:
            torch.Tensor: Logits of shape ``(B, C + 1)``.
        """
        x = x.float()
        features = self.convs(x)
        pooled = features.mean(dim=(2, 3))
        return self.fc(pooled)


def masked_class_slices(
    logits: torch.Tensor,
    masks: torch.Tensor,
    threshold: float = 0.5,
    ignore_index: int = 255,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extracts the masked per-class slices of a prediction.

    For every class, the class channel of the softmax prediction is
    element-wise multiplied by a binary mask marking the pixels belonging to
    that class. When ``masks`` carry ground-truth labels the binary mask is
    taken from them directly; when they carry pseudo-labels it is built from
    ``masks`` as well (the caller passes pseudo-labels in). Classes with no
    pixel assigned are skipped.

    Args:
        logits (torch.Tensor): Raw class logits ``(B, C, H, W)``.
        masks (torch.Tensor): Per-pixel class labels ``(B, H, W)`` (int64);
            may contain ``ignore_index`` pixels, which are excluded.
        threshold (float): Not used for the mask selection itself; kept for
            call-site symmetry with the target pseudo-labeling path.
        ignore_index (int): Label excluded from class-mask construction.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(slices, labels)`` where
            ``slices`` is a ``(N, 1, H, W)`` tensor of masked class channels
            with the same class order as ``labels``, a ``(N,)`` tensor of the
            class label of each slice. ``N`` is the total number of masked
            slices across the batch (one per present class per sample).
    """
    probs = F.softmax(logits.float(), dim=1)
    batch, num_classes, height, width = probs.shape
    device = probs.device

    slices: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for i in range(batch):
        mask = masks[i]
        valid = mask != ignore_index
        classes = torch.unique(mask[valid])
        for c in classes:
            c = int(c)
            class_mask = (mask == c) & valid  # (H, W) bool
            if not torch.any(class_mask):
                continue
            slice_map = probs[i, c] * class_mask
            slices.append(slice_map.unsqueeze(0))  # (1, H, W)
            labels.append(torch.tensor(c, device=device, dtype=torch.long))

    if not slices:
        empty = torch.empty(
            0, 1, height, width, device=device, dtype=probs.dtype
        )
        empty_labels = torch.empty(0, device=device, dtype=torch.long)
        return empty, empty_labels

    return torch.stack(slices, dim=0), torch.stack(labels, dim=0)


def target_pseudo_slices(
    logits: torch.Tensor,
    threshold: float = 0.5,
    ignore_index: int = 255,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extracts masked class slices of a prediction from its pseudo-labels.

    The binary mask for each class is derived from the model's own
    softmax-sorted predictions (the target has no real labels), optionally
    gating low-confidence pixels out via ``threshold``.

    Args:
        logits (torch.Tensor): Raw class logits ``(B, C, H, W)``.
        threshold (float): Confidence below which pixels are marked ignore.
        ignore_index (int): Label value assigned to low-confidence pixels.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ``(slices, labels,
        pseudo)`` where ``slices`` is ``(N, 1, H, W)``, ``labels`` is the
        ``(N,)`` pseudo-label class of each slice and ``pseudo`` is the
        ``(B, H, W)`` int64 pseudo-label map (with ``ignore_index`` on
        low-confidence pixels).
    """
    probs = F.softmax(logits.float(), dim=1)
    confidence, pseudo = probs.max(dim=1)
    confident = confidence > threshold
    pseudo = torch.where(
        confident,
        pseudo,
        torch.full_like(pseudo, ignore_index),
    )
    slices, labels = masked_class_slices(
        logits, pseudo, threshold=threshold, ignore_index=ignore_index
    )
    return slices, labels, pseudo


def clan_discriminator_loss(
    d_src: torch.Tensor,
    labels_src: torch.Tensor,
    d_tgt: torch.Tensor,
    labels_tgt: torch.Tensor,
    num_classes: int,
    lambda_out: float = 1.0,
) -> torch.Tensor:
    """Multi-category cross-entropy for the CLAN discriminator (Eq. 11).

    Source masked slices are labeled with their own category ``labels_src``;
    target masked slices are labeled with the extra ``num_classes`` "target"
    category (their ``labels_tgt`` is discarded in this term).

    Args:
        d_src (torch.Tensor): Source slice logits ``(N_s, C + 1)``.
        labels_src (torch.Tensor): Source slice class labels ``(N_s,)``.
        d_tgt (torch.Tensor): Target slice logits ``(N_t, C + 1)``.
        labels_tgt (torch.Tensor): Target slice pseudo-labels ``(N_t,)``.
        num_classes (int): Number of semantic classes ``C``.
        lambda_out (float): Weight of the target term.

    Returns:
        torch.Tensor: The scalar discriminator loss.
    """
    target_label = torch.full_like(labels_tgt, num_classes)
    loss = F.cross_entropy(d_src, labels_src)
    if d_tgt.numel() > 0:
        loss = loss + lambda_out * F.cross_entropy(d_tgt, target_label)
    return loss


def clan_adversarial_loss(
    d_tgt: torch.Tensor, labels_tgt: torch.Tensor
) -> torch.Tensor:
    """Backbone adversarial feedback on target masked slices.

    Pushes the target masked slices toward their own pseudo-label category so
    they are classified as if they came from the source domain, shrinking the
    source/target output-space gap.

    Args:
        d_tgt (torch.Tensor): Target slice logits ``(N_t, C + 1)``.
        labels_tgt (torch.Tensor): Target slice pseudo-labels ``(N_t,)``.

    Returns:
        torch.Tensor: The scalar adversarial loss; ``0`` when ``N_t == 0``.
    """
    if d_tgt.numel() == 0:
        return d_tgt.sum() * 0.0
    return F.cross_entropy(d_tgt, labels_tgt)
