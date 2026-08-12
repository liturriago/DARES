"""
Training engines for the DARES framework.

One engine per unsupervised domain adaptation method. Every engine subclasses
``dares.training.base_trainer.BaseTrainer`` so they share the same design
patterns (AMP, dual-domain batching, pixel metrics, checkpointing, ``fit``
loop). Available methods:

* ``source_only``  ->  supervised cross-entropy baseline (no adaptation).
* ``advent``       ->  adversarial entropy minimization.
* ``cycada``       ->  cycle-consistent adversarial domain adaptation
  (pixel + feature alignment).
* ``cbst``         ->  class-balanced self-training.
* ``dares``        ->  Domain Adaptation via alpha-Renyi Entropy.

Use ``build_engine(name, model, source_loaders, target_loaders, config,
device)`` to instantiate an engine.
"""
from dares.engines.registry import build_engine

__all__ = ["build_engine"]

