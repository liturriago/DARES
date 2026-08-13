"""Regression tests for AMP (mixed-precision) dtype boundaries.

Under ``use_amp=True`` model features / logits arrive as float16. Any module
that runs convolutions outside an ``autocast`` context would crash on a
half-vs-float weight mismatch. These tests assert every such boundary is
robust by feeding float16 inputs directly.
"""

import torch

from dares.losses.advent import entropy_map
from dares.losses.cbst import CBSTPseudoLabeling
from dares.losses.cycada import PatchDiscriminator
from dares.losses.domain import DomainDiscriminator
from dares.losses.renyi import RenyiLoss


def test_domain_discriminator_accepts_half_input():
    discriminator = DomainDiscriminator(in_channels=1)
    x = torch.randn(2, 1, 16, 16, dtype=torch.float16)
    out = discriminator(x)
    assert out.shape == (2,)
    assert out.dtype == torch.float32


def test_patch_discriminator_accepts_half_input():
    discriminator = PatchDiscriminator(in_channels=4)
    x = torch.randn(2, 4, 16, 16, dtype=torch.float16)
    out = discriminator(x)
    assert out.shape == (2, 1, 1, 1)
    assert out.dtype == torch.float32


def test_entropy_map_half_logits_returns_float():
    logits = torch.randn(2, 4, 16, 16, dtype=torch.float16)
    ent = entropy_map(logits)
    assert ent.shape == (2, 1, 16, 16)
    assert ent.dtype == torch.float32


def test_renyi_loss_half_features_is_finite():
    loss = RenyiLoss(num_classes=2, tau=0.0, n_max=512, grid_size=4)
    features_s = torch.randn(1, 8, 16, 16, dtype=torch.float16)
    features_t = torch.randn(1, 8, 16, 16, dtype=torch.float16)
    labels_s = torch.zeros(1, 16, 16, dtype=torch.long)
    logits_t = torch.randn(1, 2, 16, 16, dtype=torch.float16)

    alignment, metrics = loss(features_s, labels_s, features_t, logits_t)
    assert alignment.dtype == torch.float32
    assert torch.isfinite(alignment)
    assert "valid_classes" in metrics


def test_cbst_pseudo_labeling_accepts_half_logits():
    labeler = CBSTPseudoLabeling(num_classes=2, topk_ratio=0.5, threshold=0.0)
    logits = torch.randn(2, 2, 16, 16, dtype=torch.float16)
    pseudo, weights = labeler(logits)
    assert pseudo.shape == (2, 16, 16)
    assert weights.shape == (2, 16, 16)
    assert weights.dtype == torch.float32
