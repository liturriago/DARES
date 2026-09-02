"""DARES v2 training engine: source CE plus MIL-CREDA hardened alignment.

Same loop as the original DARES engine (warm-up, dual-domain batches,
GradNorm-lite trust region on the reference parameters) but wired to
:class:`dares.losses.dares_loss_v2.DARESLossV2`, which adds the bounded
class-global term, the local correspondence term and soft class weights.
"""

from dares.engines.dares import DARESTrainer
from dares.losses.dares_loss_v2 import DARESLossV2


class DARESV2Trainer(DARESTrainer):
    """Unsupervised domain adaptation via MIL-CREDA hardened alignment."""

    _metric_keys = DARESTrainer._metric_keys + ("loss_local",)

    def _build_criterion(self) -> DARESLossV2:
        config = self.config
        return DARESLossV2(
            **self._criterion_kwargs(),
            lambda_local=getattr(config, "lambda_local", 0.5),
            tau_local=getattr(config, "tau_local", 1.0),
            soft_class_weights=getattr(config, "soft_class_weights", True),
            bounded_align=getattr(config, "bounded_align", True),
            normalize_seg=getattr(config, "normalize_seg", False),
        )
