"""TPO-Torch — Target Policy Optimization for RLHF."""

from .loss import tpo_loss, tpo_loss_from_logits
from .models import TPOModel
from .trainer import TPODataCollator, TPOTrainer

__version__ = "0.1.0"

__all__ = [
    "TPODataCollator",
    "TPOModel",
    "TPOTrainer",
    "tpo_loss",
    "tpo_loss_from_logits",
]
