"""Elastic Weight Consolidation (EWC) for catastrophic forgetting mitigation.

Implements the EWC penalty from Kirkpatrick et al. (2017):
    L_EWC = (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2

where F_i is the diagonal Fisher information for parameter i,
theta_i is the current parameter, and theta*_i is the parameter after
training on the previous task.

Online EWC (Schwarz et al., 2018) maintains a running average:
    F_online = gamma * F_old + F_new
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def compute_fisher_information(
    model: nn.Module,
    dataloader: DataLoader,
    num_samples: int,
    device: Optional[torch.device] = None,
) -> dict[str, torch.Tensor]:
    """Compute diagonal Fisher information matrix via empirical Fisher approximation.

    Uses the squared gradients of the log-likelihood as a diagonal approximation:
        F_i = E[ (d log p(y|x, theta) / d theta_i)^2 ]

    Processes parameters one at a time to keep memory usage bounded for 70B+ models.

    Args:
        model: The model to compute Fisher information for.
        dataloader: DataLoader providing (input_ids, attention_mask, labels) batches.
        num_samples: Number of samples to use for estimation.
        device: Device to compute on. Defaults to model's device.

    Returns:
        Dictionary mapping parameter names to their diagonal Fisher values.
    """
    model.eval()
    fisher = {}

    # Initialize Fisher accumulators
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param.data)

    samples_seen = 0
    for batch in dataloader:
        if samples_seen >= num_samples:
            break

        if device is not None:
            batch = {k: v.to(device) for k, v in batch.items()}

        model.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        batch_size = next(iter(batch.values())).shape[0]
        samples_seen += batch_size

        # Accumulate squared gradients (diagonal Fisher)
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[name] += param.grad.data.pow(2) * batch_size

    # Normalize by total samples
    if samples_seen > 0:
        for name in fisher:
            fisher[name] /= samples_seen

    logger.info("Computed Fisher information over %d samples for %d parameters",
                samples_seen, len(fisher))
    return fisher


class EWCRegularizer:
    """Elastic Weight Consolidation regularizer.

    Computes the EWC penalty:
        L_EWC = (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2

    For online EWC, the Fisher matrix is a running average:
        F_online = gamma * F_old + F_new

    Args:
        model: The model being trained.
        lambda_ewc: Regularization strength.
        online: Whether to use online EWC.
        online_gamma: Decay factor for online Fisher accumulation.
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_ewc: float = 1000.0,
        online: bool = True,
        online_gamma: float = 0.95,
    ):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.online = online
        self.online_gamma = online_gamma

        # Stored reference parameters (theta*) and Fisher diagonals
        self._saved_params: dict[str, torch.Tensor] = {}
        self._fisher: dict[str, torch.Tensor] = {}
        self._initialized = False

    def register_task(
        self,
        fisher: dict[str, torch.Tensor],
    ) -> None:
        """Register a completed task by storing Fisher and current parameters.

        For online EWC, accumulates Fisher with exponential decay:
            F = gamma * F_old + F_new

        For standard EWC, replaces the stored Fisher.

        Args:
            fisher: Diagonal Fisher information from compute_fisher_information().
        """
        for name, param in self.model.named_parameters():
            if name not in fisher:
                continue

            # Store reference parameters (detached copy)
            self._saved_params[name] = param.data.clone()

            if self.online and self._initialized:
                # Online EWC: running average
                self._fisher[name] = (
                    self.online_gamma * self._fisher[name] + fisher[name]
                )
            else:
                self._fisher[name] = fisher[name].clone()

        self._initialized = True
        logger.info("Registered task with %d Fisher entries (online=%s)",
                    len(self._fisher), self.online)

    def penalty(self) -> torch.Tensor:
        """Compute the EWC penalty term.

        Returns:
            L_EWC = (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2

            Returns 0.0 if no task has been registered yet.
        """
        if not self._initialized:
            return torch.tensor(0.0)

        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)

        for name, param in self.model.named_parameters():
            if name not in self._fisher:
                continue
            # EWC quadratic penalty weighted by Fisher diagonal
            diff = param - self._saved_params[name]
            loss += (self._fisher[name] * diff.pow(2)).sum()

        return (self.lambda_ewc / 2.0) * loss
