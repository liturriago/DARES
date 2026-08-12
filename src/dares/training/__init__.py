"""
Training package for the DARES framework.

``base_trainer`` provides the shared ``BaseTrainer`` template: AMP handling,
dual-domain batching, pixel metric evaluation, best-checkpoint tracking and
the epoch loop. Each UDA method implements its own engine under
``dares.engines`` subclassing ``BaseTrainer``.
"""
