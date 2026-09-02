"""
Loss functions for the DARES framework.

Shared modules:

* ``ce``  ->  ``SegCrossEntropyLoss`` (pixel-wise cross-entropy).
* ``domain``  ->  ``DomainDiscriminator`` + ``adversarial_loss`` (used by the
  ADVENT engine).

Method-specific losses live in sibling modules:

* ``advent``  ->  entropy minimization loss (ADVENT engine).
* ``cbst``  ->  class-balanced pseudo-labeling and masked self-training loss
  (CBST engine).
* ``clan``  ->  category-level multi-class discriminator and masked class-slice
  alignment losses (CLAN engine).
* ``dacs``  ->  color jitter, Gaussian blur, ClassMix and pseudo-labeling
  helpers (DACS engine).
* ``fda``  ->  Fourier-domain amplitude swap and Charbonnier entropy loss (FDA
  engine).
* ``dares_loss``  ->  the hardened DARES alignment loss (anti-collapse entropy
  floors, inter-class target repulsion and a GradNorm-lite trust region).
* ``dares_loss_v2``  ->  DARES v2 (MIL-CREDA hardening: bounded class-global
  term via the mixed matrix and conservative entropy bounds, local
  correspondence to a personalized source reference, soft class weights).
"""
from dares.losses.ce import SegCrossEntropyLoss

__all__ = ["SegCrossEntropyLoss"]
