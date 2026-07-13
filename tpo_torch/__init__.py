"""TPO-Torch — Target Policy Optimization for RLHF."""

from .loss import tpo_loss, tpo_loss_from_logits

__version__ = "0.1.0"

__all__ = [
    "TPODataCollator",
    "TPOModel",
    "TPOTrainer",
    "tpo_loss",
    "tpo_loss_from_logits",
]


def __getattr__(name):
    if name == "TPOTrainer":
        from .trainer import TPOTrainer
        return TPOTrainer
    if name == "TPODataCollator":
        from .trainer import TPODataCollator
        return TPODataCollator
    if name == "TPOModel":
        from .models import TPOModel
        return TPOModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
