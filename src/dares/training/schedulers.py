"""Learning-rate schedulers for the DARES training engines."""

import math
from typing import Any

import torch
from torch.optim.lr_scheduler import LRScheduler

from dares.config import TrainConfig


class DARESScheduler(LRScheduler):
    """DARES dynamic learning-rate schedule (Eq. 29 of the paper).

    ``eta(p) = eta0 * (1 + alpha * p)^(-beta)`` with ``p`` the relative
    training progress ``epoch / total_epochs``. Because the schedule is
    parameterized by relative progress it is invariant to the total epoch
    budget, unlike a per-epoch exponential decay.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer whose base learning rate
            is scaled.
        total_epochs (int): Total number of training epochs.
        alpha (float): Schedule hyperparameter (default 20.0).
        beta (float): Schedule hyperparameter (default 0.75).
        last_epoch (int): Last epoch index (default -1).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_epochs: int,
        alpha: float = 20.0,
        beta: float = 0.75,
        last_epoch: int = -1,
    ) -> None:
        self.total_epochs = max(int(total_epochs), 1)
        self.alpha = float(alpha)
        self.beta = float(beta)
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list[float]:
        """Computes the scaled learning rate for each parameter group."""
        progress = min(max(self.last_epoch / self.total_epochs, 0.0), 1.0)
        factor = (1.0 + self.alpha * progress) ** (-self.beta)
        return [base_lr * factor for base_lr in self.base_lrs]


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: TrainConfig
) -> LRScheduler:
    """Builds the learning-rate scheduler selected by the configuration.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer to schedule.
        config (TrainConfig): Training configuration; ``lr_schedule`` enables
            the DARES dynamic schedule, otherwise an exponential decay with
            ``gamma`` is used.

    Returns:
        LRScheduler: The configured scheduler.
    """
    if config.lr_schedule:
        return DARESScheduler(
            optimizer,
            total_epochs=config.epochs,
            alpha=config.schedule_alpha,
            beta=config.schedule_beta,
        )
    return torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=config.gamma
    )
