"""Experience replay buffers for continual learning.

Implements reservoir sampling (Vitter, 1985) for uniform random sampling
and importance-weighted buffering for priority-based replay.
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class ReservoirBuffer:
    """Reservoir sampling buffer for experience replay.

    Maintains a fixed-size buffer using Algorithm R (Vitter, 1985):
    - First buffer_size items fill the buffer directly
    - Item n (n > buffer_size) replaces a random item with probability buffer_size/n

    This guarantees each item has equal probability of being in the buffer.

    Args:
        buffer_size: Maximum number of examples to store.
        on_disk: If True, store examples on disk to save memory.
        disk_path: Directory for on-disk storage. Required if on_disk=True.
    """

    def __init__(
        self,
        buffer_size: int = 10000,
        on_disk: bool = False,
        disk_path: Optional[str] = None,
    ):
        self.buffer_size = buffer_size
        self.on_disk = on_disk
        self.disk_path = disk_path
        self._buffer: list[dict[str, torch.Tensor]] = []
        self._count = 0  # Total items seen (for reservoir probability)

        if on_disk:
            if disk_path is None:
                raise ValueError("disk_path required when on_disk=True")
            os.makedirs(disk_path, exist_ok=True)

    def add(self, example: dict[str, torch.Tensor]) -> None:
        """Add an example to the buffer using reservoir sampling.

        Args:
            example: Dictionary with tokenized fields (input_ids, attention_mask, labels).
        """
        self._count += 1

        if len(self._buffer) < self.buffer_size:
            self._store(len(self._buffer), example)
            self._buffer.append(example if not self.on_disk else None)
        else:
            # Reservoir sampling: replace with probability buffer_size / count
            j = random.randint(0, self._count - 1)
            if j < self.buffer_size:
                self._store(j, example)
                if not self.on_disk:
                    self._buffer[j] = example

    def sample(self, n: int) -> list[dict[str, torch.Tensor]]:
        """Sample n random examples from the buffer.

        Args:
            n: Number of examples to sample (with replacement if n > buffer size).

        Returns:
            List of example dictionaries.
        """
        current_size = min(len(self._buffer), self.buffer_size)
        if current_size == 0:
            return []

        indices = [random.randint(0, current_size - 1) for _ in range(n)]
        return [self._load(i) for i in indices]

    def _store(self, index: int, example: dict[str, torch.Tensor]) -> None:
        """Store example at index, either in memory or on disk."""
        if self.on_disk:
            path = Path(self.disk_path) / f"{index}.pt"
            torch.save({k: v.cpu() for k, v in example.items()}, path)
        # In-memory storage is handled directly in add()

    def _load(self, index: int) -> dict[str, torch.Tensor]:
        """Load example from index."""
        if self.on_disk:
            path = Path(self.disk_path) / f"{index}.pt"
            return torch.load(path, weights_only=True)
        return self._buffer[index]

    def __len__(self) -> int:
        return min(len(self._buffer), self.buffer_size)


class ImportanceWeightedBuffer:
    """Priority-based experience replay buffer.

    Each example is assigned an importance score (e.g., loss value).
    Sampling probability is proportional to importance score.
    Higher-loss examples are replayed more frequently.

    Args:
        buffer_size: Maximum number of examples to store.
    """

    def __init__(self, buffer_size: int = 10000):
        self.buffer_size = buffer_size
        self._buffer: list[dict[str, torch.Tensor]] = []
        self._priorities: list[float] = []

    def add(self, example: dict[str, torch.Tensor], priority: float = 1.0) -> None:
        """Add an example with an importance priority.

        Args:
            example: Tokenized example dictionary.
            priority: Importance score (higher = more likely to be sampled).
        """
        if len(self._buffer) < self.buffer_size:
            self._buffer.append(example)
            self._priorities.append(priority)
        else:
            # Replace the lowest priority item if new item has higher priority
            min_idx = self._priorities.index(min(self._priorities))
            if priority > self._priorities[min_idx]:
                self._buffer[min_idx] = example
                self._priorities[min_idx] = priority

    def update_priority(self, index: int, priority: float) -> None:
        """Update the priority of a buffered example.

        Args:
            index: Buffer index.
            priority: New importance score.
        """
        if 0 <= index < len(self._priorities):
            self._priorities[index] = priority

    def sample(self, n: int) -> list[dict[str, torch.Tensor]]:
        """Sample n examples with probability proportional to priority.

        Args:
            n: Number of examples to sample.

        Returns:
            List of example dictionaries.
        """
        if not self._buffer:
            return []

        total = sum(self._priorities)
        if total == 0:
            weights = [1.0 / len(self._priorities)] * len(self._priorities)
        else:
            weights = [p / total for p in self._priorities]

        indices = random.choices(range(len(self._buffer)), weights=weights, k=n)
        return [self._buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self._buffer)


class ReplayDataLoader:
    """DataLoader that mixes new training data with replay buffer samples.

    Each batch contains (1 - replay_ratio) new examples and replay_ratio
    replay examples.

    Args:
        train_dataloader: DataLoader for new training data.
        buffer: Replay buffer (ReservoirBuffer or ImportanceWeightedBuffer).
        replay_ratio: Fraction of batch from replay buffer (0-1).
        device: Device to move replay examples to.
    """

    def __init__(
        self,
        train_dataloader: DataLoader,
        buffer: ReservoirBuffer | ImportanceWeightedBuffer,
        replay_ratio: float = 0.2,
        device: Optional[torch.device] = None,
    ):
        self.train_dataloader = train_dataloader
        self.buffer = buffer
        self.replay_ratio = replay_ratio
        self.device = device

    def __iter__(self):
        for batch in self.train_dataloader:
            if len(self.buffer) == 0 or self.replay_ratio <= 0:
                yield batch
                continue

            batch_size = next(iter(batch.values())).shape[0]
            n_replay = max(1, int(batch_size * self.replay_ratio))
            n_new = batch_size - n_replay

            # Get replay samples
            replay_samples = self.buffer.sample(n_replay)
            if not replay_samples:
                yield batch
                continue

            # Collate replay samples
            replay_batch = _collate(replay_samples)

            # Truncate new data and concatenate with replay
            merged = {}
            for key in batch:
                new_part = batch[key][:n_new]
                replay_part = replay_batch.get(key)
                if replay_part is not None:
                    # Pad to same sequence length if needed
                    max_len = max(new_part.shape[-1], replay_part.shape[-1])
                    if new_part.shape[-1] < max_len:
                        new_part = _pad_tensor(new_part, max_len)
                    if replay_part.shape[-1] < max_len:
                        replay_part = _pad_tensor(replay_part, max_len)
                    merged[key] = torch.cat([new_part, replay_part], dim=0)
                else:
                    merged[key] = new_part

                if self.device is not None:
                    merged[key] = merged[key].to(self.device)

            yield merged

    def __len__(self) -> int:
        return len(self.train_dataloader)


def _collate(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate a list of example dicts into a batched dict."""
    if not samples:
        return {}

    keys = samples[0].keys()
    collated = {}
    for key in keys:
        tensors = [s[key] for s in samples if key in s]
        if not tensors:
            continue
        # Pad to max length in this mini-batch
        max_len = max(t.shape[-1] for t in tensors)
        padded = [_pad_tensor(t.unsqueeze(0) if t.dim() == 1 else t, max_len) for t in tensors]
        collated[key] = torch.cat(padded, dim=0)

    return collated


def _pad_tensor(tensor: torch.Tensor, target_len: int, pad_value: int = 0) -> torch.Tensor:
    """Pad the last dimension of a tensor to target_len."""
    if tensor.shape[-1] >= target_len:
        return tensor
    pad_size = target_len - tensor.shape[-1]
    padding = torch.full(
        (*tensor.shape[:-1], pad_size),
        pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat([tensor, padding], dim=-1)
