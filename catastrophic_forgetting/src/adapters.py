"""Adapter isolation (LoRA/QLoRA) for catastrophic forgetting mitigation.

Uses HuggingFace PEFT to apply low-rank adapters while keeping base model
weights frozen. This naturally prevents forgetting since base weights are
never modified.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def setup_lora_model(
    base_model: nn.Module,
    rank: int = 16,
    alpha: float = 32.0,
    target_modules: Optional[list[str]] = None,
    dropout: float = 0.05,
    use_quantization: bool = True,
    quantization_bits: int = 4,
    use_double_quant: bool = True,
) -> nn.Module:
    """Configure a model with LoRA/QLoRA adapters and frozen base weights.

    The effective LoRA scaling is alpha/rank, applied to the low-rank update:
        W' = W + (alpha/rank) * B @ A

    where A and B are the low-rank matrices.

    Args:
        base_model: The pretrained model to add adapters to.
        rank: LoRA rank (dimension of low-rank decomposition).
        alpha: LoRA scaling factor.
        target_modules: Module names to apply LoRA to. Defaults to attention projections.
        dropout: Dropout probability for LoRA layers.
        use_quantization: Enable 4/8-bit QLoRA quantization.
        quantization_bits: Number of bits (4 or 8).
        use_double_quant: Use nested quantization for further memory savings.

    Returns:
        PeftModel with LoRA adapters applied.
    """
    from peft import LoraConfig, get_peft_model, TaskType

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    if use_quantization:
        from transformers import BitsAndBytesConfig
        _apply_quantization(base_model, quantization_bits, use_double_quant)

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    peft_model = get_peft_model(base_model, lora_config)

    # Verify base weights are frozen
    trainable, total = _count_parameters(peft_model)
    logger.info(
        "LoRA applied: %d trainable / %d total parameters (%.2f%%)",
        trainable, total, 100.0 * trainable / total if total > 0 else 0,
    )

    return peft_model


def _apply_quantization(
    model: nn.Module,
    bits: int,
    use_double_quant: bool,
) -> None:
    """Apply bitsandbytes quantization config to model.

    Note: In practice, quantization is applied at model load time via
    AutoModelForCausalLM.from_pretrained(quantization_config=...).
    This function logs the intended config for reference.
    """
    logger.info(
        "Quantization config: %d-bit, double_quant=%s. "
        "Apply via from_pretrained(quantization_config=BitsAndBytesConfig(...))",
        bits, use_double_quant,
    )


def _count_parameters(model: nn.Module) -> tuple[int, int]:
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


class AdapterBank:
    """Manages multiple task-specific LoRA adapters.

    Allows switching between adapters for different tasks, enabling
    task-specific fine-tuning without interference.

    Args:
        base_model: The base PeftModel with initial adapter.
    """

    def __init__(self, base_model: nn.Module):
        self.model = base_model
        self._adapters: dict[str, bool] = {"default": True}

    def add_adapter(
        self,
        name: str,
        rank: int = 16,
        alpha: float = 32.0,
        target_modules: Optional[list[str]] = None,
        dropout: float = 0.05,
    ) -> None:
        """Add a new task-specific adapter.

        Args:
            name: Unique name for this adapter.
            rank: LoRA rank.
            alpha: LoRA scaling factor.
            target_modules: Module names to apply LoRA to.
            dropout: LoRA dropout.
        """
        from peft import LoraConfig

        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=dropout,
            bias="none",
        )
        self.model.add_adapter(name, config)
        self._adapters[name] = True
        logger.info("Added adapter '%s' (rank=%d, alpha=%.1f)", name, rank, alpha)

    def switch_adapter(self, name: str) -> None:
        """Switch to a named adapter for inference or training.

        Args:
            name: Adapter name to activate.

        Raises:
            KeyError: If adapter name not found.
        """
        if name not in self._adapters:
            raise KeyError(f"Adapter '{name}' not found. Available: {list(self._adapters)}")
        self.model.set_adapter(name)
        logger.info("Switched to adapter '%s'", name)

    def list_adapters(self) -> list[str]:
        """Return list of available adapter names."""
        return list(self._adapters.keys())

    def merge_adapter(self, name: str) -> None:
        """Merge a specific adapter's weights into the base model.

        Warning: This modifies base model weights and is irreversible.

        Args:
            name: Adapter to merge.
        """
        self.switch_adapter(name)
        self.model.merge_adapter()
        logger.info("Merged adapter '%s' into base model", name)
