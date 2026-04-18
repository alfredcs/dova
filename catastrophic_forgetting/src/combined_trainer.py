"""Combined training loop integrating EWC, adapter isolation, and experience replay.

Orchestrates all three catastrophic forgetting mitigation methods into a single
training pipeline. Supports any combination: EWC-only, Adapter-only, Replay-only,
or any mix.
"""

import logging
import os
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .ewc import EWCRegularizer, compute_fisher_information
from .adapters import setup_lora_model
from .replay_buffer import ReservoirBuffer, ImportanceWeightedBuffer, ReplayDataLoader
from .metrics import compute_forgetting_rate
from .utils import enable_gradient_checkpointing, get_optimizer, verify_frozen_weights

logger = logging.getLogger(__name__)


class ContinualFineTuner:
    """Orchestrates continual fine-tuning with forgetting mitigation.

    Combines three strategies:
    1. EWC regularization (penalizes changes to important parameters)
    2. LoRA/QLoRA adapter isolation (freezes base weights)
    3. Experience replay (mixes old data into training)

    The recommended recipe for 70B+ models (from research):
    - QLoRA (NF4, double quant) + LoRA r=32, alpha=64
    - 10% replay mix ratio
    - Gradient checkpointing enabled
    - bf16 mixed precision

    Args:
        model: Base model to fine-tune.
        config: Training configuration.
        device: Target device. Auto-detected if None.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # Setup adapter isolation if enabled
        if config.use_adapters:
            self.model = setup_lora_model(
                model,
                rank=config.adapter.rank,
                alpha=config.adapter.alpha,
                target_modules=config.adapter.target_modules,
                dropout=config.adapter.dropout,
                use_quantization=config.adapter.use_quantization,
                quantization_bits=config.adapter.quantization_bits,
                use_double_quant=config.adapter.use_double_quant,
            )
        else:
            self.model = model

        # Enable gradient checkpointing for memory efficiency
        if config.ewc.gradient_checkpointing:
            enable_gradient_checkpointing(self.model)

        verify_frozen_weights(self.model)

        # Setup EWC if enabled
        self.ewc: Optional[EWCRegularizer] = None
        if config.use_ewc:
            self.ewc = EWCRegularizer(
                self.model,
                lambda_ewc=config.ewc.lambda_ewc,
                online=config.ewc.online,
                online_gamma=config.ewc.online_gamma,
            )

        # Setup replay buffer if enabled
        self.replay_buffer: Optional[ReservoirBuffer | ImportanceWeightedBuffer] = None
        if config.use_replay:
            if config.replay.sampling_strategy == "importance":
                self.replay_buffer = ImportanceWeightedBuffer(
                    buffer_size=config.replay.buffer_size,
                )
            else:
                self.replay_buffer = ReservoirBuffer(
                    buffer_size=config.replay.buffer_size,
                    on_disk=config.replay.on_disk,
                    disk_path=config.replay.disk_path,
                )

        self._task_count = 0

    def populate_replay_buffer(self, dataloader: DataLoader) -> None:
        """Fill replay buffer with examples from the current task's data.

        Should be called before training on a new task to preserve
        representative examples from the current distribution.

        Args:
            dataloader: DataLoader providing examples to store.
        """
        if self.replay_buffer is None:
            return

        count = 0
        for batch in dataloader:
            batch_size = next(iter(batch.values())).shape[0]
            for i in range(batch_size):
                example = {k: v[i].cpu() for k, v in batch.items()}

                if isinstance(self.replay_buffer, ImportanceWeightedBuffer):
                    # Compute loss as priority
                    with torch.no_grad():
                        single = {k: v[i:i+1].to(self.device) for k, v in batch.items()}
                        loss = self.model(**single).loss.item()
                    self.replay_buffer.add(example, priority=loss)
                else:
                    self.replay_buffer.add(example)

                count += 1

        logger.info("Populated replay buffer with %d examples (buffer size: %d)",
                    count, len(self.replay_buffer))

    def train_task(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        task_name: Optional[str] = None,
    ) -> dict[str, list[float]]:
        """Train on a single task with all enabled mitigation strategies.

        Training loop:
        1. Wraps dataloader with replay mixing (if enabled)
        2. For each batch: compute task loss + EWC penalty (if enabled)
        3. After training: compute Fisher and register with EWC (if enabled)

        Args:
            train_dataloader: Training data for the new task.
            eval_dataloader: Optional eval data for logging metrics.
            task_name: Optional name for logging.

        Returns:
            Dictionary with training history ('train_loss', 'ewc_penalty', etc.)
        """
        task_name = task_name or f"task_{self._task_count}"
        logger.info("Starting training on '%s' (task %d)", task_name, self._task_count)

        # Wrap with replay if enabled and buffer has data
        if self.replay_buffer is not None and len(self.replay_buffer) > 0:
            effective_dataloader = ReplayDataLoader(
                train_dataloader,
                self.replay_buffer,
                replay_ratio=self.config.replay.replay_ratio,
                device=self.device,
            )
        else:
            effective_dataloader = train_dataloader

        optimizer = get_optimizer(self.model, self.config.learning_rate)

        # Setup mixed precision
        scaler = None
        autocast_dtype = None
        if self.config.mixed_precision == "bf16":
            autocast_dtype = torch.bfloat16
        elif self.config.mixed_precision == "fp16":
            autocast_dtype = torch.float16
            scaler = torch.amp.GradScaler("cuda")

        history = {"train_loss": [], "ewc_penalty": [], "total_loss": []}

        self.model.train()
        global_step = 0

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for step, batch in enumerate(effective_dataloader):
                if self.device is not None:
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass with optional autocast
                if autocast_dtype is not None:
                    with torch.amp.autocast("cuda", dtype=autocast_dtype):
                        outputs = self.model(**batch)
                        task_loss = outputs.loss
                else:
                    outputs = self.model(**batch)
                    task_loss = outputs.loss

                # Add EWC penalty
                ewc_penalty = torch.tensor(0.0, device=self.device)
                if self.ewc is not None:
                    ewc_penalty = self.ewc.penalty()

                total_loss = task_loss + ewc_penalty

                # Scale for gradient accumulation
                scaled_loss = total_loss / self.config.gradient_accumulation_steps

                # Backward pass
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                # Optimizer step every gradient_accumulation_steps
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.max_grad_norm > 0:
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.max_grad_norm
                        )

                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    optimizer.zero_grad()
                    global_step += 1

                    # Logging
                    if global_step % self.config.logging_steps == 0:
                        logger.info(
                            "[%s] epoch=%d step=%d task_loss=%.4f ewc=%.4f total=%.4f",
                            task_name, epoch, global_step,
                            task_loss.item(), ewc_penalty.item(), total_loss.item(),
                        )

                history["train_loss"].append(task_loss.item())
                history["ewc_penalty"].append(ewc_penalty.item())
                history["total_loss"].append(total_loss.item())

                epoch_loss += total_loss.item()
                num_batches += 1

                # Periodic evaluation
                if (
                    eval_dataloader is not None
                    and self.config.eval_steps is not None
                    and global_step > 0
                    and global_step % self.config.eval_steps == 0
                ):
                    self._evaluate(eval_dataloader, task_name, global_step)
                    self.model.train()

            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            logger.info("[%s] Epoch %d complete. Avg loss: %.4f",
                        task_name, epoch, avg_epoch_loss)

        # Post-training: compute Fisher and register task for EWC
        if self.ewc is not None:
            logger.info("Computing Fisher information for EWC registration...")
            fisher = compute_fisher_information(
                self.model, train_dataloader,
                num_samples=self.config.ewc.fisher_samples,
                device=self.device,
            )
            self.ewc.register_task(fisher)

        self._task_count += 1
        logger.info("Completed training on '%s'", task_name)
        return history

    def _evaluate(
        self,
        eval_dataloader: DataLoader,
        task_name: str,
        step: int,
    ) -> float:
        """Run evaluation and log metrics."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                if self.device is not None:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                batch_size = next(iter(batch.values())).shape[0]
                total_loss += outputs.loss.item() * batch_size
                total_samples += batch_size

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        logger.info("[%s] Eval at step %d: loss=%.4f", task_name, step, avg_loss)
        return avg_loss

    def save_checkpoint(self, path: str) -> None:
        """Save model and EWC state to disk.

        Args:
            path: Directory to save checkpoint.
        """
        os.makedirs(path, exist_ok=True)

        # Save model (handles both PEFT and regular models)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(os.path.join(path, "model"))
        else:
            torch.save(
                self.model.state_dict(),
                os.path.join(path, "model.pt"),
            )

        # Save EWC state
        if self.ewc is not None and self.ewc._initialized:
            torch.save(
                {
                    "fisher": self.ewc._fisher,
                    "saved_params": self.ewc._saved_params,
                },
                os.path.join(path, "ewc_state.pt"),
            )

        logger.info("Saved checkpoint to %s", path)

    def load_ewc_state(self, path: str) -> None:
        """Load EWC state from a checkpoint.

        Args:
            path: Directory containing the checkpoint.
        """
        if self.ewc is None:
            logger.warning("EWC not enabled, skipping load")
            return

        ewc_path = os.path.join(path, "ewc_state.pt")
        if not os.path.exists(ewc_path):
            logger.warning("No EWC state found at %s", ewc_path)
            return

        state = torch.load(ewc_path, weights_only=True)
        self.ewc._fisher = state["fisher"]
        self.ewc._saved_params = state["saved_params"]
        self.ewc._initialized = True
        logger.info("Loaded EWC state from %s", path)
