"""Tests for EWC regularizer and Fisher information computation."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ewc import EWCRegularizer, compute_fisher_information


class TestComputeFisherInformation:
    """Tests for diagonal Fisher information computation."""

    def test_fisher_keys_match_trainable_params(self, small_model, sample_dataloader):
        """Fisher should have entries for all trainable parameters."""
        fisher = compute_fisher_information(small_model, sample_dataloader, num_samples=8)
        trainable_names = {n for n, p in small_model.named_parameters() if p.requires_grad}
        assert set(fisher.keys()) == trainable_names

    def test_fisher_shapes_match_params(self, small_model, sample_dataloader):
        """Each Fisher diagonal should match the shape of its parameter."""
        fisher = compute_fisher_information(small_model, sample_dataloader, num_samples=8)
        for name, param in small_model.named_parameters():
            if param.requires_grad:
                assert fisher[name].shape == param.shape

    def test_fisher_values_non_negative(self, small_model, sample_dataloader):
        """Fisher diagonal values are squared gradients, so must be >= 0."""
        fisher = compute_fisher_information(small_model, sample_dataloader, num_samples=8)
        for name, f in fisher.items():
            assert (f >= 0).all(), f"Negative Fisher values for {name}"

    def test_fisher_single_sample(self, small_model):
        """Fisher computation should work with a single sample."""
        input_ids = torch.randint(0, 100, (1, 8))
        labels = input_ids.clone()
        dataset = TensorDataset(input_ids, torch.ones_like(input_ids), labels)

        def collate_fn(batch):
            ids, mask, lab = zip(*batch)
            return {"input_ids": torch.stack(ids), "attention_mask": torch.stack(mask), "labels": torch.stack(lab)}

        dl = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
        fisher = compute_fisher_information(small_model, dl, num_samples=1)
        assert len(fisher) > 0

    def test_fisher_respects_num_samples(self, small_model, sample_dataloader):
        """Fisher with more samples should generally differ from fewer samples."""
        fisher_few = compute_fisher_information(small_model, sample_dataloader, num_samples=4)
        fisher_many = compute_fisher_information(small_model, sample_dataloader, num_samples=16)
        # They should both exist and have the same keys
        assert set(fisher_few.keys()) == set(fisher_many.keys())

    def test_fisher_excludes_frozen_params(self, small_model, sample_dataloader):
        """Fisher should not include parameters with requires_grad=False."""
        # Freeze embedding
        small_model.embedding.weight.requires_grad = False
        fisher = compute_fisher_information(small_model, sample_dataloader, num_samples=8)
        assert "embedding.weight" not in fisher


class TestEWCRegularizer:
    """Tests for the EWC regularizer class."""

    def test_penalty_zero_before_registration(self, small_model):
        """Penalty should be 0.0 before any task is registered."""
        ewc = EWCRegularizer(small_model, lambda_ewc=1000.0)
        penalty = ewc.penalty()
        assert penalty.item() == 0.0

    def test_penalty_zero_at_reference_point(self, small_model, sample_dataloader):
        """Penalty should be 0 when params haven't changed from reference."""
        ewc = EWCRegularizer(small_model, lambda_ewc=1000.0, online=False)
        fisher = compute_fisher_information(small_model, sample_dataloader, num_samples=8)
        ewc.register_task(fisher)
        penalty = ewc.penalty()
        assert penalty.item() == pytest.approx(0.0, abs=1e-6)

    def test_penalty_increases_with_param_change(self, small_model, sample_dataloader):
        """Penalty should increase when parameters move away from reference."""
        ewc = EWCRegularizer(small_model, lambda_ewc=1000.0, online=False)
        fisher = compute_fisher_information(small_model, sample_dataloader, num_samples=8)
        ewc.register_task(fisher)

        # Perturb parameters
        with torch.no_grad():
            for param in small_model.parameters():
                param.add_(torch.randn_like(param) * 0.1)

        penalty = ewc.penalty()
        assert penalty.item() > 0.0

    def test_penalty_scales_with_lambda(self, small_model, sample_dataloader):
        """Larger lambda should produce larger penalty."""
        fisher = compute_fisher_information(small_model, sample_dataloader, num_samples=8)

        ewc_small = EWCRegularizer(small_model, lambda_ewc=1.0, online=False)
        ewc_small.register_task(fisher)

        ewc_large = EWCRegularizer(small_model, lambda_ewc=100.0, online=False)
        ewc_large.register_task(fisher)

        # Perturb parameters
        with torch.no_grad():
            for param in small_model.parameters():
                param.add_(torch.randn_like(param) * 0.1)

        penalty_small = ewc_small.penalty()
        penalty_large = ewc_large.penalty()
        assert penalty_large.item() > penalty_small.item()
        # Should scale linearly: penalty_large / penalty_small ~= 100
        ratio = penalty_large.item() / penalty_small.item()
        assert ratio == pytest.approx(100.0, rel=0.01)

    def test_ewc_formula_numerical(self, small_model, sample_dataloader):
        """Verify EWC penalty matches the formula: (lambda/2) * sum(F * (theta - theta*)^2)."""
        ewc = EWCRegularizer(small_model, lambda_ewc=50.0, online=False)
        fisher = compute_fisher_information(small_model, sample_dataloader, num_samples=8)
        ewc.register_task(fisher)

        # Save reference and perturb
        saved = {n: p.data.clone() for n, p in small_model.named_parameters()}
        with torch.no_grad():
            for param in small_model.parameters():
                param.add_(torch.randn_like(param) * 0.05)

        # Compute penalty via the class
        penalty = ewc.penalty().item()

        # Compute manually
        manual = 0.0
        for name, param in small_model.named_parameters():
            if name in fisher:
                diff = param.data - saved[name]
                manual += (fisher[name] * diff.pow(2)).sum().item()
        manual *= 50.0 / 2.0

        assert penalty == pytest.approx(manual, rel=1e-4)

    def test_online_ewc_accumulates_fisher(self, small_model, sample_dataloader):
        """Online EWC should blend Fisher matrices across tasks."""
        gamma = 0.9
        ewc = EWCRegularizer(small_model, lambda_ewc=1.0, online=True, online_gamma=gamma)

        fisher1 = compute_fisher_information(small_model, sample_dataloader, num_samples=8)
        ewc.register_task(fisher1)

        fisher2 = compute_fisher_information(small_model, sample_dataloader, num_samples=8)
        ewc.register_task(fisher2)

        # Verify the blended Fisher = gamma * fisher1 + fisher2
        for name in fisher1:
            expected = gamma * fisher1[name] + fisher2[name]
            actual = ewc._fisher[name]
            assert torch.allclose(actual, expected, atol=1e-5), f"Fisher mismatch for {name}"

    def test_standard_ewc_replaces_fisher(self, small_model, sample_dataloader):
        """Standard (non-online) EWC should replace Fisher, not accumulate."""
        ewc = EWCRegularizer(small_model, lambda_ewc=1.0, online=False)

        fisher1 = compute_fisher_information(small_model, sample_dataloader, num_samples=8)
        ewc.register_task(fisher1)

        fisher2 = compute_fisher_information(small_model, sample_dataloader, num_samples=8)
        ewc.register_task(fisher2)

        # Should be exactly fisher2
        for name in fisher2:
            assert torch.allclose(ewc._fisher[name], fisher2[name], atol=1e-6)

    def test_penalty_is_differentiable(self, small_model, sample_dataloader):
        """EWC penalty should be differentiable for backprop."""
        ewc = EWCRegularizer(small_model, lambda_ewc=1000.0, online=False)
        fisher = compute_fisher_information(small_model, sample_dataloader, num_samples=8)
        ewc.register_task(fisher)

        # Perturb and compute penalty
        with torch.no_grad():
            for param in small_model.parameters():
                param.add_(torch.randn_like(param) * 0.1)

        penalty = ewc.penalty()
        penalty.backward()

        # Check that gradients exist
        has_grad = any(p.grad is not None for p in small_model.parameters())
        assert has_grad
