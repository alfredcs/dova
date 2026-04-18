"""Configuration dataclasses for catastrophic forgetting mitigation."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EWCConfig:
    """Configuration for Elastic Weight Consolidation.

    Args:
        lambda_ewc: Regularization strength. Higher values preserve old knowledge more.
        fisher_samples: Number of samples to estimate Fisher information matrix.
        online: If True, use online EWC with running average of Fisher matrices.
        online_gamma: Decay factor for online EWC Fisher accumulation (0-1).
            Only used when online=True.
        gradient_checkpointing: Use gradient checkpointing to reduce memory for 70B+ models.
    """
    lambda_ewc: float = 1000.0
    fisher_samples: int = 1000
    online: bool = True
    online_gamma: float = 0.95
    gradient_checkpointing: bool = True


@dataclass
class AdapterConfig:
    """Configuration for LoRA/QLoRA adapter isolation.

    Args:
        rank: LoRA rank (dimension of low-rank matrices).
        alpha: LoRA scaling factor. Effective scaling is alpha/rank.
        target_modules: Module name patterns to apply LoRA to.
        dropout: Dropout probability for LoRA layers.
        use_quantization: If True, use 4-bit QLoRA quantization.
        quantization_bits: Number of bits for quantization (4 or 8).
        use_double_quant: Use double quantization for further memory savings.
    """
    rank: int = 16
    alpha: float = 32.0
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    dropout: float = 0.05
    use_quantization: bool = True
    quantization_bits: int = 4
    use_double_quant: bool = True


@dataclass
class ReplayConfig:
    """Configuration for experience replay buffer.

    Args:
        buffer_size: Maximum number of examples to store.
        sampling_strategy: 'reservoir' for uniform or 'importance' for priority-based.
        replay_ratio: Fraction of each batch that comes from replay buffer (0-1).
        on_disk: If True, store buffer on disk instead of memory.
        disk_path: Path for on-disk storage. Required if on_disk=True.
    """
    buffer_size: int = 10000
    sampling_strategy: str = "reservoir"
    replay_ratio: float = 0.2
    on_disk: bool = False
    disk_path: Optional[str] = None


@dataclass
class TrainingConfig:
    """Combined training configuration.

    Args:
        use_ewc: Enable EWC regularization.
        use_adapters: Enable LoRA/QLoRA adapter isolation.
        use_replay: Enable experience replay.
        ewc: EWC configuration.
        adapter: Adapter configuration.
        replay: Replay configuration.
        learning_rate: Optimizer learning rate.
        num_epochs: Number of training epochs.
        batch_size: Training batch size.
        gradient_accumulation_steps: Steps to accumulate gradients before update.
        mixed_precision: 'bf16', 'fp16', or None for fp32.
        max_grad_norm: Maximum gradient norm for clipping.
        logging_steps: Log metrics every N steps.
        eval_steps: Evaluate every N steps. None to evaluate only at epoch end.
        output_dir: Directory for checkpoints and logs.
    """
    use_ewc: bool = True
    use_adapters: bool = True
    use_replay: bool = True
    ewc: EWCConfig = field(default_factory=EWCConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    learning_rate: float = 2e-5
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    mixed_precision: Optional[str] = "bf16"
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    eval_steps: Optional[int] = None
    output_dir: str = "./output"
