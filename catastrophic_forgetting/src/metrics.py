"""Metrics for measuring catastrophic forgetting and knowledge transfer.

Implements standard continual learning metrics:
- Forgetting rate: degradation on previously learned tasks
- Forward transfer: performance on new tasks
- Backward transfer: effect of new learning on old tasks
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@torch.no_grad()
def compute_forgetting_rate(
    model: nn.Module,
    eval_dataloader: DataLoader,
    baseline_loss: float,
    device: Optional[torch.device] = None,
) -> dict[str, float]:
    """Measure capability degradation on a previously learned task.

    Forgetting rate = (current_loss - baseline_loss) / baseline_loss

    A positive value indicates forgetting; negative indicates improvement.

    Args:
        model: The model to evaluate.
        eval_dataloader: DataLoader for the old task evaluation set.
        baseline_loss: Loss on this task immediately after training on it.
        device: Device to run evaluation on.

    Returns:
        Dictionary with 'current_loss', 'baseline_loss', 'forgetting_rate'.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for batch in eval_dataloader:
        if device is not None:
            batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        batch_size = next(iter(batch.values())).shape[0]
        total_loss += outputs.loss.item() * batch_size
        total_samples += batch_size

    current_loss = total_loss / total_samples if total_samples > 0 else 0.0

    forgetting_rate = (
        (current_loss - baseline_loss) / baseline_loss
        if baseline_loss > 0 else 0.0
    )

    result = {
        "current_loss": current_loss,
        "baseline_loss": baseline_loss,
        "forgetting_rate": forgetting_rate,
    }
    logger.info("Forgetting rate: %.4f (loss: %.4f -> %.4f)",
                forgetting_rate, baseline_loss, current_loss)
    return result


@torch.no_grad()
def compute_forward_transfer(
    model: nn.Module,
    new_task_dataloader: DataLoader,
    random_baseline_loss: float,
    device: Optional[torch.device] = None,
) -> dict[str, float]:
    """Measure how well prior knowledge helps on a new task (before training on it).

    Forward transfer = (random_baseline_loss - model_loss) / random_baseline_loss

    Positive means prior learning helps; negative means it hurts.

    Args:
        model: The model to evaluate (trained on prior tasks, not yet on new task).
        new_task_dataloader: DataLoader for the new task.
        random_baseline_loss: Loss of a randomly initialized model on this task.
        device: Device for evaluation.

    Returns:
        Dictionary with 'model_loss', 'random_baseline_loss', 'forward_transfer'.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for batch in new_task_dataloader:
        if device is not None:
            batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        batch_size = next(iter(batch.values())).shape[0]
        total_loss += outputs.loss.item() * batch_size
        total_samples += batch_size

    model_loss = total_loss / total_samples if total_samples > 0 else 0.0

    forward_transfer = (
        (random_baseline_loss - model_loss) / random_baseline_loss
        if random_baseline_loss > 0 else 0.0
    )

    result = {
        "model_loss": model_loss,
        "random_baseline_loss": random_baseline_loss,
        "forward_transfer": forward_transfer,
    }
    logger.info("Forward transfer: %.4f (loss: %.4f vs random: %.4f)",
                forward_transfer, model_loss, random_baseline_loss)
    return result


@torch.no_grad()
def compute_backward_transfer(
    model: nn.Module,
    old_task_dataloaders: dict[str, DataLoader],
    baseline_losses: dict[str, float],
    device: Optional[torch.device] = None,
) -> dict[str, dict[str, float]]:
    """Measure retention across multiple old tasks after new learning.

    For each old task, computes forgetting rate and aggregates.

    Backward transfer = average forgetting rate across all old tasks.
    Negative backward transfer means the model has forgotten.

    Args:
        model: The model to evaluate after training on new task(s).
        old_task_dataloaders: Mapping of task_name -> DataLoader for each old task.
        baseline_losses: Mapping of task_name -> loss after training on that task.
        device: Device for evaluation.

    Returns:
        Dictionary with per-task forgetting rates and 'average_backward_transfer'.
    """
    results = {}
    total_bt = 0.0

    for task_name, dataloader in old_task_dataloaders.items():
        baseline = baseline_losses.get(task_name, 0.0)
        task_result = compute_forgetting_rate(model, dataloader, baseline, device)
        results[task_name] = task_result
        # Backward transfer is negative of forgetting rate
        total_bt += -task_result["forgetting_rate"]

    n_tasks = len(old_task_dataloaders)
    avg_bt = total_bt / n_tasks if n_tasks > 0 else 0.0
    results["average_backward_transfer"] = {"value": avg_bt}

    logger.info("Average backward transfer: %.4f across %d tasks", avg_bt, n_tasks)
    return results
