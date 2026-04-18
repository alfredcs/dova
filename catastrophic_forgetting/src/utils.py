"""Shared utilities for catastrophic forgetting mitigation."""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def verify_frozen_weights(model: nn.Module) -> dict[str, int]:
    """Verify which parameters are frozen vs trainable.

    Returns:
        Dictionary with 'frozen', 'trainable', 'total' parameter counts.
    """
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = frozen + trainable

    logger.info(
        "Parameters: %d trainable, %d frozen, %d total (%.2f%% trainable)",
        trainable, frozen, total, 100.0 * trainable / total if total > 0 else 0,
    )
    return {"frozen": frozen, "trainable": trainable, "total": total}


def enable_gradient_checkpointing(model: nn.Module) -> None:
    """Enable gradient checkpointing for memory-efficient training of large models.

    Trades compute for memory by recomputing activations during backward pass
    instead of storing them. Essential for 70B+ models.
    """
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    else:
        logger.warning("Model does not support gradient_checkpointing_enable()")


def get_optimizer(
    model: nn.Module,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    """Create AdamW optimizer with weight decay, excluding bias and LayerNorm.

    Args:
        model: Model to optimize.
        learning_rate: Learning rate.
        weight_decay: Weight decay coefficient.

    Returns:
        Configured AdamW optimizer.
    """
    no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    params = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return torch.optim.AdamW(params, lr=learning_rate)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
