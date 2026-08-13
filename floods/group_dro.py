from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch.utils.data import Dataset


def event_id_from_path(path: str | Path) -> str:
    """Extract an EMSR event identifier from a processed tile path."""
    match = re.search(r"(EMSR\d+)", Path(path).name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not extract an EMSR event ID from: {path}")
    return match.group(1).upper()


class EventIndexedDataset(Dataset):
    """Attach a stable integer event group to every training sample.

    The wrapped dataset remains index-compatible with any sampler built from the
    original dataset. All public attributes (label_files, categories, mean, etc.)
    are forwarded so existing training code continues to work.
    """

    def __init__(self, dataset: Dataset, label_files: Sequence[str | Path] | None = None, *, require_multiple: bool = True) -> None:
        self.dataset = dataset
        paths = list(label_files if label_files is not None else getattr(dataset, "label_files"))
        if len(paths) != len(dataset):
            raise ValueError(
                "GroupDRO event indexing requires one label path per dataset item: "
                f"labels={len(paths)} dataset={len(dataset)}"
            )
        event_ids = [event_id_from_path(path) for path in paths]
        self.event_names = sorted(set(event_ids))
        self.event_to_index = {name: index for index, name in enumerate(self.event_names)}
        self.event_indices = [self.event_to_index[name] for name in event_ids]
        if require_multiple and len(self.event_names) < 2:
            raise ValueError("Event indexing requires at least two distinct events for this use case")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        item = self.dataset[index]
        if not isinstance(item, (tuple, list)):
            raise TypeError(f"Wrapped training sample must be tuple/list, got {type(item)!r}")
        group_index = torch.tensor(self.event_indices[index], dtype=torch.long)
        return tuple(item) + (group_index,)

    def __getattr__(self, name: str) -> Any:
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)


def group_mean_losses(
    per_sample_losses: torch.Tensor,
    group_indices: torch.Tensor,
    num_groups: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return differentiable group means, counts, and a present-group mask."""
    losses = per_sample_losses.reshape(-1)
    groups = group_indices.reshape(-1).long().to(losses.device)
    if losses.numel() != groups.numel():
        raise ValueError(
            "Per-sample losses and group indices must have the same length: "
            f"losses={losses.numel()} groups={groups.numel()}"
        )
    if num_groups <= 0:
        raise ValueError("num_groups must be positive")
    if groups.numel() and (int(groups.min()) < 0 or int(groups.max()) >= num_groups):
        raise ValueError("group_indices contain values outside [0, num_groups)")

    sums = torch.zeros(num_groups, dtype=losses.dtype, device=losses.device)
    counts = torch.zeros(num_groups, dtype=losses.dtype, device=losses.device)
    if losses.numel():
        sums.scatter_add_(0, groups, losses)
        counts.scatter_add_(0, groups, torch.ones_like(losses))
    present = counts > 0
    means = torch.zeros_like(sums)
    means[present] = sums[present] / counts[present]
    return means, counts, present


def update_group_weights(
    weights: torch.Tensor,
    observed_group_losses: torch.Tensor,
    present: torch.Tensor,
    *,
    eta: float,
    min_weight: float = 0.0,
) -> torch.Tensor:
    """Exponentiated-gradient GroupDRO update with an optional probability floor."""
    if eta < 0:
        raise ValueError("GroupDRO eta must be non-negative")
    q = weights.detach().float().clone()
    losses = observed_group_losses.detach().float().to(q.device)
    present = present.to(device=q.device, dtype=torch.bool)
    if q.ndim != 1 or losses.shape != q.shape or present.shape != q.shape:
        raise ValueError("weights, observed_group_losses, and present must be equal 1D shapes")
    if not torch.isfinite(q).all() or float(q.sum()) <= 0:
        q.fill_(1.0 / max(q.numel(), 1))
    if present.any() and eta > 0:
        safe_losses = torch.nan_to_num(losses, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
        q[present] = q[present] * torch.exp(float(eta) * safe_losses[present])
    q = q.clamp_min(0.0)
    total = q.sum()
    if not torch.isfinite(total) or float(total) <= 0:
        q.fill_(1.0 / max(q.numel(), 1))
    else:
        q /= total

    floor = float(min_weight or 0.0)
    if floor < 0:
        raise ValueError("GroupDRO min_weight must be non-negative")
    if floor > 0:
        if floor * q.numel() >= 1.0:
            raise ValueError("GroupDRO min_weight is too large for the number of groups")
        q = q * (1.0 - floor * q.numel()) + floor
        q /= q.sum()
    return q


def robust_present_group_loss(
    group_losses: torch.Tensor,
    present: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted robust loss, normalised over groups represented in this minibatch."""
    present = present.to(device=group_losses.device, dtype=torch.bool)
    if not present.any():
        return group_losses.sum() * 0.0
    q = weights.to(device=group_losses.device, dtype=group_losses.dtype)
    local_q = q[present]
    denominator = local_q.sum()
    if not torch.isfinite(denominator) or float(denominator.detach()) <= 0:
        local_q = torch.full_like(local_q, 1.0 / local_q.numel())
    else:
        local_q = local_q / denominator
    return torch.sum(local_q * group_losses[present])


def effective_group_count(weights: torch.Tensor) -> float:
    q = weights.detach().float().clamp_min(torch.finfo(torch.float32).tiny)
    q = q / q.sum()
    entropy = -torch.sum(q * torch.log(q))
    return float(torch.exp(entropy).cpu())


def append_event_weight_rows(
    path: str | Path,
    *,
    epoch: int,
    event_names: Sequence[str],
    weights: torch.Tensor,
    mean_losses: torch.Tensor,
    observation_counts: torch.Tensor,
) -> None:
    """Append one epoch of event weights and observed losses to CSV."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_header = not destination.exists()
    q = weights.detach().float().cpu()
    losses = mean_losses.detach().float().cpu()
    counts = observation_counts.detach().float().cpu()
    with destination.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "event_id", "group_weight", "mean_observed_loss", "observed_samples"],
        )
        if write_header:
            writer.writeheader()
        for index, event_name in enumerate(event_names):
            count = float(counts[index])
            writer.writerow(
                {
                    "epoch": int(epoch),
                    "event_id": str(event_name),
                    "group_weight": float(q[index]),
                    "mean_observed_loss": float(losses[index]) if count > 0 else math.nan,
                    "observed_samples": int(round(count)),
                }
            )
