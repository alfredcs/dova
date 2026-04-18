"""Tests for ContinualFineTuner combined training integration."""

import os
import tempfile

import pytest
import torch

from src.config import TrainingConfig, EWCConfig, AdapterConfig, ReplayConfig
from src.combined_trainer import ContinualFineTuner


def _make_config(**overrides):
    """Create a minimal training config for testing."""
    defaults = dict(
        use_ewc=True,
        use_adapters=False,  # Skip LoRA for speed in most tests
        use_replay=True,
        ewc=EWCConfig(lambda_ewc=10.0, fisher_samples=8, gradient_checkpointing=False),
        adapter=AdapterConfig(rank=4, alpha=8.0, use_quantization=False),
        replay=ReplayConfig(buffer_size=20, replay_ratio=0.1),
        learning_rate=1e-3,
        num_epochs=1,
        batch_size=4,
        gradient_accumulation_steps=1,
        mixed_precision=None,  # CPU tests, no amp
        max_grad_norm=1.0,
        logging_steps=100,
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)


class TestContinualFineTuner:
    """Tests for the combined training orchestrator."""

    def test_init_ewc_only(self, small_model):
        """Should initialize with EWC only."""
        config = _make_config(use_ewc=True, use_adapters=False, use_replay=False)
        tuner = ContinualFineTuner(small_model, config, device=torch.device("cpu"))
        assert tuner.ewc is not None
        assert tuner.replay_buffer is None

    def test_init_replay_only(self, small_model):
        """Should initialize with replay only."""
        config = _make_config(use_ewc=False, use_adapters=False, use_replay=True)
        tuner = ContinualFineTuner(small_model, config, device=torch.device("cpu"))
        assert tuner.ewc is None
        assert tuner.replay_buffer is not None

    def test_init_all_disabled(self, small_model):
        """Should initialize with all strategies disabled."""
        config = _make_config(use_ewc=False, use_adapters=False, use_replay=False)
        tuner = ContinualFineTuner(small_model, config, device=torch.device("cpu"))
        assert tuner.ewc is None
        assert tuner.replay_buffer is None

    def test_train_task_returns_history(self, small_model, sample_dataloader):
        """train_task should return a history dict with loss traces."""
        config = _make_config(use_ewc=True, use_adapters=False, use_replay=False)
        tuner = ContinualFineTuner(small_model, config, device=torch.device("cpu"))
        history = tuner.train_task(sample_dataloader, task_name="test_task")
        assert "train_loss" in history
        assert "ewc_penalty" in history
        assert "total_loss" in history
        assert len(history["train_loss"]) > 0

    def test_ewc_penalty_increases_after_second_task(self, small_model, sample_dataloader):
        """EWC penalty should be non-zero when training on a second task."""
        config = _make_config(use_ewc=True, use_adapters=False, use_replay=False)
        tuner = ContinualFineTuner(small_model, config, device=torch.device("cpu"))

        # Train first task (registers Fisher)
        tuner.train_task(sample_dataloader, task_name="task_0")

        # Train second task -- EWC penalty should appear
        history = tuner.train_task(sample_dataloader, task_name="task_1")
        ewc_penalties = history["ewc_penalty"]
        assert any(p > 0 for p in ewc_penalties), "EWC penalty should be non-zero on second task"

    def test_replay_buffer_populated(self, small_model, sample_dataloader):
        """populate_replay_buffer should add examples."""
        config = _make_config(use_ewc=False, use_adapters=False, use_replay=True)
        tuner = ContinualFineTuner(small_model, config, device=torch.device("cpu"))
        assert len(tuner.replay_buffer) == 0
        tuner.populate_replay_buffer(sample_dataloader)
        assert len(tuner.replay_buffer) > 0

    def test_task_count_increments(self, small_model, sample_dataloader):
        """Task count should increment after each train_task call."""
        config = _make_config(use_ewc=False, use_adapters=False, use_replay=False)
        tuner = ContinualFineTuner(small_model, config, device=torch.device("cpu"))
        assert tuner._task_count == 0
        tuner.train_task(sample_dataloader)
        assert tuner._task_count == 1
        tuner.train_task(sample_dataloader)
        assert tuner._task_count == 2

    def test_save_and_load_checkpoint(self, small_model, sample_dataloader):
        """Should save and load EWC state."""
        config = _make_config(use_ewc=True, use_adapters=False, use_replay=False)
        tuner = ContinualFineTuner(small_model, config, device=torch.device("cpu"))
        tuner.train_task(sample_dataloader, task_name="task_0")

        with tempfile.TemporaryDirectory() as tmpdir:
            tuner.save_checkpoint(tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "ewc_state.pt"))

            # Load into a fresh tuner
            from catastrophic_forgetting.tests.conftest import SmallCausalLM
            new_model = SmallCausalLM()
            new_tuner = ContinualFineTuner(new_model, config, device=torch.device("cpu"))
            new_tuner.load_ewc_state(tmpdir)
            assert new_tuner.ewc._initialized

    def test_gradient_accumulation(self, small_model, sample_dataloader):
        """Training with gradient accumulation > 1 should not crash."""
        config = _make_config(
            use_ewc=False, use_adapters=False, use_replay=False,
            gradient_accumulation_steps=4,
        )
        tuner = ContinualFineTuner(small_model, config, device=torch.device("cpu"))
        history = tuner.train_task(sample_dataloader)
        assert len(history["train_loss"]) > 0

    def test_importance_weighted_replay(self, small_model, sample_dataloader):
        """Should work with importance-weighted replay strategy."""
        config = _make_config(
            use_ewc=False, use_adapters=False, use_replay=True,
            replay=ReplayConfig(buffer_size=20, sampling_strategy="importance", replay_ratio=0.1),
        )
        tuner = ContinualFineTuner(small_model, config, device=torch.device("cpu"))
        tuner.populate_replay_buffer(sample_dataloader)
        assert len(tuner.replay_buffer) > 0
