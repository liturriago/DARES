"""Regression tests for AMP (mixed-precision) dtype boundaries.

Under ``use_amp=True`` model features / logits arrive as float16. Any module
that runs convolutions outside an ``autocast`` context would crash on a
half-vs-float weight mismatch. These tests assert every such boundary is
robust by feeding float16 inputs directly.
"""

import torch

from dares.losses.advent import entropy_map
from dares.losses.domain import DomainDiscriminator


def test_domain_discriminator_accepts_half_input():
    discriminator = DomainDiscriminator(in_channels=1)
    x = torch.randn(2, 1, 16, 16, dtype=torch.float16)
    out = discriminator(x)
    assert out.shape == (2,)
    assert out.dtype == torch.float32


def test_entropy_map_half_logits_returns_float():
    logits = torch.randn(2, 4, 16, 16, dtype=torch.float16)
    ent = entropy_map(logits)
    assert ent.shape == (2, 1, 16, 16)
    assert ent.dtype == torch.float32
