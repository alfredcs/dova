"""Catastrophic forgetting mitigation framework for 70B+ model fine-tuning."""

from .config import EWCConfig, AdapterConfig, ReplayConfig, TrainingConfig
from .ewc import EWCRegularizer, compute_fisher_information
from .adapters import setup_lora_model, AdapterBank
from .replay_buffer import ReservoirBuffer, ImportanceWeightedBuffer, ReplayDataLoader
from .combined_trainer import ContinualFineTuner
from .metrics import compute_forgetting_rate, compute_forward_transfer, compute_backward_transfer

__all__ = [
    "EWCConfig",
    "AdapterConfig",
    "ReplayConfig",
    "TrainingConfig",
    "EWCRegularizer",
    "compute_fisher_information",
    "setup_lora_model",
    "AdapterBank",
    "ReservoirBuffer",
    "ImportanceWeightedBuffer",
    "ReplayDataLoader",
    "ContinualFineTuner",
    "compute_forgetting_rate",
    "compute_forward_transfer",
    "compute_backward_transfer",
]
