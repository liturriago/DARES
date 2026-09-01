"""
Training engines for the DARES framework.

One engine per unsupervised domain adaptation method. Every engine subclasses
``dares.training.base_trainer.BaseTrainer`` so they share the same design
patterns (AMP, dual-domain batching, pixel metrics, checkpointing, ``fit``
loop). Available methods:

* ``source_only``  ->  supervised cross-entropy baseline (no adaptation).
* ``advent``       ->  adversarial entropy minimization.
* ``cbst``         ->  class-balanced self-training with pseudo-labels.
* ``clan``         ->  category-level adversarial output-space adaptation.
* ``dacs``         ->  cross-domain mixed sampling self-training.
* ``fda``          ->  Fourier Domain Adaptation (spectral amplitude swap).
* ``dares``        ->  Domain Adaptation via alpha-Renyi Entropy.

Use ``build_engine(name, model, source_loaders, target_loaders, config,
device)`` to instantiate an engine.
"""
from dares.engines.registry import build_engine

__all__ = ["build_engine"]
