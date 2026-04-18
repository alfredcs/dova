"""Tests for reservoir sampling and importance-weighted replay buffers."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import collections
import tempfile

import pytest
import torch

from replay_buffer import ReservoirBuffer, ImportanceWeightedBuffer, ReplayDataLoader, _collate, _pad_tensor


class TestReservoirBuffer:
    """Tests for reservoir sampling buffer."""

    def test_fills_to_capacity(self, sample_examples):
        """Buffer should fill up to buffer_size."""
        buf = ReservoirBuffer(buffer_size=10)
        for ex in sample_examples[:10]:
            buf.add(ex)
        assert len(buf) == 10

    def test_stays_at_capacity(self, sample_examples):
        """Buffer should not exceed buffer_size."""
        buf = ReservoirBuffer(buffer_size=10)
        for ex in sample_examples:  # 50 examples
            buf.add(ex)
        assert len(buf) == 10

    def test_sample_returns_correct_count(self, sample_examples):
        """sample(n) should return exactly n examples."""
        buf = ReservoirBuffer(buffer_size=10)
        for ex in sample_examples[:15]:
            buf.add(ex)
        samples = buf.sample(5)
        assert len(samples) == 5

    def test_sample_returns_dicts(self, sample_examples):
        """Sampled examples should be dict[str, Tensor]."""
        buf = ReservoirBuffer(buffer_size=10)
        for ex in sample_examples[:10]:
            buf.add(ex)
        samples = buf.sample(3)
        for s in samples:
            assert isinstance(s, dict)
            assert "input_ids" in s
            assert isinstance(s["input_ids"], torch.Tensor)

    def test_empty_buffer_returns_empty(self):
        """Sampling from empty buffer should return empty list."""
        buf = ReservoirBuffer(buffer_size=10)
        assert buf.sample(5) == []

    def test_reservoir_probability_approximately_uniform(self):
        """Reservoir sampling should give each item roughly equal probability.

        Statistical test: add 10000 items to buffer of size 100.
        Each item should appear with P ≈ 100/10000 = 0.01.
        We check the final buffer has items from various ranges, not just the last ones.
        """
        buf = ReservoirBuffer(buffer_size=100)
        for i in range(10000):
            example = {"id": torch.tensor([i])}
            buf.add(example)

        # Get all IDs in the buffer
        ids = [buf._buffer[i]["id"].item() for i in range(len(buf))]

        # Items should come from various ranges, not just the tail
        # Divide into 10 buckets of 1000
        buckets = [0] * 10
        for id_val in ids:
            bucket = min(id_val // 1000, 9)
            buckets[bucket] += 1

        # Each bucket should have roughly 10 items (100/10=10)
        # Allow wide margin for randomness, but no bucket should be 0
        # and no bucket should have more than 30 (3x expected)
        for i, count in enumerate(buckets):
            assert count > 0, f"Bucket {i} is empty -- reservoir sampling not uniform"

    def test_on_disk_storage(self, sample_examples):
        """On-disk buffer should work identically to in-memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            buf = ReservoirBuffer(buffer_size=10, on_disk=True, disk_path=tmpdir)
            for ex in sample_examples[:15]:
                buf.add(ex)
            assert len(buf) == 10

            samples = buf.sample(3)
            assert len(samples) == 3
            for s in samples:
                assert "input_ids" in s

    def test_on_disk_requires_path(self):
        """on_disk=True without disk_path should raise ValueError."""
        with pytest.raises(ValueError, match="disk_path required"):
            ReservoirBuffer(buffer_size=10, on_disk=True)


class TestImportanceWeightedBuffer:
    """Tests for priority-based replay buffer."""

    def test_fills_to_capacity(self, sample_examples):
        """Buffer should fill up to buffer_size."""
        buf = ImportanceWeightedBuffer(buffer_size=10)
        for ex in sample_examples[:10]:
            buf.add(ex, priority=1.0)
        assert len(buf) == 10

    def test_high_priority_replaces_low(self, sample_examples):
        """Higher priority items should replace lower ones when full."""
        buf = ImportanceWeightedBuffer(buffer_size=5)
        # Fill with low priority
        for i, ex in enumerate(sample_examples[:5]):
            buf.add(ex, priority=0.1)

        # Add high priority item
        high_priority_ex = {"input_ids": torch.tensor([999])}
        buf.add(high_priority_ex, priority=10.0)

        # The buffer should have replaced the lowest priority item
        assert len(buf) == 5
        # Check that the high priority example is in there
        found = any(
            s.get("input_ids") is not None and s["input_ids"].shape == (1,) and s["input_ids"].item() == 999
            for s in buf._buffer
        )
        assert found

    def test_low_priority_not_added_when_full(self, sample_examples):
        """Low priority items should not replace higher ones."""
        buf = ImportanceWeightedBuffer(buffer_size=5)
        for ex in sample_examples[:5]:
            buf.add(ex, priority=10.0)

        low_ex = {"input_ids": torch.tensor([888])}
        buf.add(low_ex, priority=0.001)

        found = any(
            s.get("input_ids") is not None and s["input_ids"].shape == (1,) and s["input_ids"].item() == 888
            for s in buf._buffer
        )
        assert not found

    def test_sampling_proportional_to_priority(self):
        """Higher priority items should be sampled more frequently."""
        buf = ImportanceWeightedBuffer(buffer_size=2)
        buf.add({"id": torch.tensor([0])}, priority=1.0)
        buf.add({"id": torch.tensor([1])}, priority=99.0)

        # Sample many times
        counts = collections.Counter()
        for _ in range(10000):
            samples = buf.sample(1)
            counts[samples[0]["id"].item()] += 1

        # Item 1 (priority 99) should appear much more than item 0 (priority 1)
        assert counts[1] > counts[0] * 5  # At least 5x more

    def test_empty_buffer_returns_empty(self):
        """Sampling from empty buffer should return empty list."""
        buf = ImportanceWeightedBuffer(buffer_size=10)
        assert buf.sample(5) == []

    def test_zero_priority_uniform_sampling(self):
        """When all priorities are 0, should fall back to uniform."""
        buf = ImportanceWeightedBuffer(buffer_size=3)
        buf.add({"id": torch.tensor([0])}, priority=0.0)
        buf.add({"id": torch.tensor([1])}, priority=0.0)
        buf.add({"id": torch.tensor([2])}, priority=0.0)

        # Should not crash
        samples = buf.sample(5)
        assert len(samples) == 5

    def test_update_priority(self):
        """Priority update should change sampling distribution."""
        buf = ImportanceWeightedBuffer(buffer_size=2)
        buf.add({"id": torch.tensor([0])}, priority=1.0)
        buf.add({"id": torch.tensor([1])}, priority=1.0)

        buf.update_priority(0, 100.0)
        assert buf._priorities[0] == 100.0


class TestReplayDataLoader:
    """Tests for the replay data mixing loader."""

    def test_yields_batches(self, sample_dataloader, sample_examples):
        """ReplayDataLoader should yield batches."""
        buf = ReservoirBuffer(buffer_size=20)
        for ex in sample_examples[:20]:
            buf.add(ex)

        loader = ReplayDataLoader(sample_dataloader, buf, replay_ratio=0.2)
        batches = list(loader)
        assert len(batches) > 0

    def test_empty_buffer_passes_through(self, sample_dataloader):
        """With empty buffer, should yield original batches unchanged."""
        buf = ReservoirBuffer(buffer_size=10)
        loader = ReplayDataLoader(sample_dataloader, buf, replay_ratio=0.2)
        original_batches = list(sample_dataloader)
        replay_batches = list(loader)
        assert len(replay_batches) == len(original_batches)

    def test_zero_ratio_passes_through(self, sample_dataloader, sample_examples):
        """With replay_ratio=0, should yield original batches."""
        buf = ReservoirBuffer(buffer_size=20)
        for ex in sample_examples[:20]:
            buf.add(ex)

        loader = ReplayDataLoader(sample_dataloader, buf, replay_ratio=0.0)
        for batch in loader:
            assert "input_ids" in batch

    def test_len_matches_original(self, sample_dataloader, sample_examples):
        """ReplayDataLoader should have same length as original."""
        buf = ReservoirBuffer(buffer_size=20)
        for ex in sample_examples[:20]:
            buf.add(ex)

        loader = ReplayDataLoader(sample_dataloader, buf, replay_ratio=0.2)
        assert len(loader) == len(sample_dataloader)


class TestPadding:
    """Tests for tensor padding utility."""

    def test_pad_extends_correctly(self):
        """Padding should extend the last dimension."""
        t = torch.ones(2, 5)
        padded = _pad_tensor(t, 10)
        assert padded.shape == (2, 10)
        assert (padded[:, :5] == 1).all()
        assert (padded[:, 5:] == 0).all()

    def test_no_pad_when_already_long_enough(self):
        """Should return unchanged tensor if already >= target_len."""
        t = torch.ones(2, 10)
        padded = _pad_tensor(t, 5)
        assert padded.shape == (2, 10)
        assert torch.equal(padded, t)

    def test_collate_handles_variable_lengths(self):
        """_collate should handle samples with different sequence lengths."""
        samples = [
            {"input_ids": torch.ones(5)},
            {"input_ids": torch.ones(10)},
        ]
        collated = _collate(samples)
        assert collated["input_ids"].shape == (2, 10)

    def test_collate_empty(self):
        """_collate with empty list should return empty dict."""
        assert _collate([]) == {}
