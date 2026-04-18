"""Tests for LoRA adapter isolation and AdapterBank."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch
from peft import get_peft_model, LoraConfig, TaskType

from adapters import setup_lora_model, AdapterBank, _count_parameters


class TestSetupLoraModel:
    """Tests for LoRA model setup."""

    def test_base_weights_frozen(self, small_model):
        """Base model weights should be frozen after LoRA setup."""
        peft_model = setup_lora_model(
            small_model, rank=4, alpha=8.0, use_quantization=False,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        for name, param in peft_model.named_parameters():
            if "lora_" not in name:
                assert not param.requires_grad, f"Base param {name} should be frozen"

    def test_lora_weights_trainable(self, small_model):
        """LoRA adapter weights should be trainable."""
        peft_model = setup_lora_model(
            small_model, rank=4, alpha=8.0, use_quantization=False,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        lora_params = [
            (n, p) for n, p in peft_model.named_parameters() if "lora_" in n
        ]
        assert len(lora_params) > 0, "No LoRA parameters found"
        for name, param in lora_params:
            assert param.requires_grad, f"LoRA param {name} should be trainable"

    def test_trainable_params_much_smaller(self, small_model):
        """Trainable params should be a small fraction of total."""
        peft_model = setup_lora_model(
            small_model, rank=4, alpha=8.0, use_quantization=False,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        trainable, total = _count_parameters(peft_model)
        assert trainable < total
        assert trainable / total < 0.5

    def test_rank_1_lora(self, small_model):
        """LoRA should work with minimum rank=1."""
        peft_model = setup_lora_model(
            small_model, rank=1, alpha=1.0, use_quantization=False,
            target_modules=["q_proj"],
        )
        trainable, total = _count_parameters(peft_model)
        assert trainable > 0

    def test_forward_pass_works(self, small_model):
        """Model should still produce outputs after LoRA setup."""
        peft_model = setup_lora_model(
            small_model, rank=4, alpha=8.0, use_quantization=False,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        input_ids = torch.randint(0, 100, (2, 8))
        labels = input_ids.clone()
        output = peft_model(input_ids=input_ids, labels=labels)
        assert output.loss is not None
        assert output.logits.shape == (2, 8, 100)

    def test_base_weights_unchanged_after_training_step(self, small_model):
        """Base weights should not change after a gradient step on LoRA params."""
        peft_model = setup_lora_model(
            small_model, rank=4, alpha=8.0, use_quantization=False,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

        # Save base weights
        base_weights = {}
        for name, param in peft_model.named_parameters():
            if "lora_" not in name:
                base_weights[name] = param.data.clone()

        # Do a training step
        optimizer = torch.optim.Adam(
            [p for p in peft_model.parameters() if p.requires_grad], lr=1e-3
        )
        input_ids = torch.randint(0, 100, (4, 16))
        output = peft_model(input_ids=input_ids, labels=input_ids.clone())
        output.loss.backward()
        optimizer.step()

        # Verify base weights unchanged
        for name, param in peft_model.named_parameters():
            if "lora_" not in name and name in base_weights:
                assert torch.equal(param.data, base_weights[name]), \
                    f"Base weight {name} changed after training step"


class TestAdapterBank:
    """Tests for multi-adapter management."""

    def _make_peft_model(self, small_model):
        return setup_lora_model(
            small_model, rank=4, alpha=8.0, use_quantization=False,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

    def test_initial_adapter_listed(self, small_model):
        """AdapterBank should list the default adapter."""
        peft_model = self._make_peft_model(small_model)
        bank = AdapterBank(peft_model)
        assert "default" in bank.list_adapters()

    def test_add_adapter(self, small_model):
        """Should be able to add a named adapter."""
        peft_model = self._make_peft_model(small_model)
        bank = AdapterBank(peft_model)
        bank.add_adapter("task_a", rank=4, alpha=8.0,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        assert "task_a" in bank.list_adapters()

    def test_switch_adapter(self, small_model):
        """Should switch between adapters without error."""
        peft_model = self._make_peft_model(small_model)
        bank = AdapterBank(peft_model)
        bank.add_adapter("task_a", rank=4, alpha=8.0,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        bank.switch_adapter("task_a")
        bank.switch_adapter("default")

    def test_switch_nonexistent_adapter_raises(self, small_model):
        """Switching to a non-existent adapter should raise KeyError."""
        peft_model = self._make_peft_model(small_model)
        bank = AdapterBank(peft_model)
        with pytest.raises(KeyError, match="not_real"):
            bank.switch_adapter("not_real")
