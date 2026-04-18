"""Shared fixtures for catastrophic forgetting mitigation tests."""

import sys
import os

# Add the parent of src/ to sys.path so 'src' can be imported as a package
_cf_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _cf_root)
# Also add src/ directly for modules that don't use relative imports
sys.path.insert(0, os.path.join(_cf_root, "src"))

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class SmallCausalLM(nn.Module):
    """Tiny causal LM for testing. Mimics HuggingFace model interface."""

    def __init__(self, vocab_size: int = 100, hidden_size: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()
        # Minimal config object for PEFT compatibility
        self.config = type("Config", (), {"model_type": "custom"})()
        self.config.to_dict = lambda: {"model_type": "custom"}

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        x = self.embedding(input_ids)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn = torch.softmax(q @ k.transpose(-1, -2) / (q.shape[-1] ** 0.5), dim=-1)
        x = self.o_proj(attn @ v)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        return _ModelOutput(loss=loss, logits=logits)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}


class _ModelOutput:
    """Minimal output object matching HuggingFace interface."""
    def __init__(self, loss, logits):
        self.loss = loss
        self.logits = logits


@pytest.fixture
def small_model():
    """A small model for testing."""
    return SmallCausalLM(vocab_size=100, hidden_size=32)


@pytest.fixture
def sample_dataloader():
    """DataLoader with small batches of tokenized data."""
    batch_size = 4
    seq_len = 16
    num_samples = 20

    input_ids = torch.randint(0, 100, (num_samples, seq_len))
    attention_mask = torch.ones(num_samples, seq_len, dtype=torch.long)
    labels = input_ids.clone()

    dataset = TensorDataset(input_ids, attention_mask, labels)

    def collate_fn(batch):
        ids, mask, lab = zip(*batch)
        return {
            "input_ids": torch.stack(ids),
            "attention_mask": torch.stack(mask),
            "labels": torch.stack(lab),
        }

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)


@pytest.fixture
def sample_examples():
    """List of individual example dicts for replay buffer testing."""
    examples = []
    for _ in range(50):
        seq_len = torch.randint(8, 20, (1,)).item()
        examples.append({
            "input_ids": torch.randint(0, 100, (seq_len,)),
            "attention_mask": torch.ones(seq_len, dtype=torch.long),
            "labels": torch.randint(0, 100, (seq_len,)),
        })
    return examples
