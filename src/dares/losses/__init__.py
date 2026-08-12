"""
Loss functions for the DARES framework.

Shared modules:

* ``ce``  ->  ``SegCrossEntropyLoss`` (pixel-wise cross-entropy).
* ``domain``  ->  ``DomainDiscriminator`` + ``adversarial_loss`` (shared by the
  ADVENT and CyCADA engines).

Method-specific losses live in sibling modules:

* ``advent``  ->  entropy minimization loss (ADVENT engine).
* ``cycada``  ->  pixel generators, PatchGAN discriminator and cycle losses.
* ``cbst``  ->  class-balanced pseudo-label selection / self-training loss.
* ``renyi``  ->  the DARES alpha-Renyi alignment loss (sampling operator
  ``Phi_c`` and matrix-based mutual information).
"""
from dares.losses.ce import SegCrossEntropyLoss

__all__ = ["SegCrossEntropyLoss"]
