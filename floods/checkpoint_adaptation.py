from __future__ import annotations

import torch


def adapt_input_state_dict(model: torch.nn.Module, state: dict, mode: str = "strict") -> tuple[dict, list[str]]:
    """Adapt only wider convolutional input tensors during a fresh warm start.

    ``zero_extra`` copies every checkpoint input channel exactly and initialises
    newly added channels to zero. The initial forward pass therefore preserves
    the old model contribution while allowing gradients to learn the new inputs.
    All other shape mismatches remain fatal.
    """
    mode = str(mode or "strict").strip().lower().replace("-", "_")
    if mode == "strict":
        return state, []
    if mode != "zero_extra":
        raise ValueError(f"Unsupported init channel adaptation mode: {mode}")
    target_state = model.state_dict()
    adapted_state = dict(state)
    adapted_keys: list[str] = []
    incompatible: list[str] = []
    for key, source in state.items():
        target = target_state.get(key)
        if target is None or not torch.is_tensor(source) or not torch.is_tensor(target):
            continue
        if tuple(source.shape) == tuple(target.shape):
            continue
        can_expand_input = (
            source.ndim == 4
            and target.ndim == 4
            and source.shape[0] == target.shape[0]
            and source.shape[2:] == target.shape[2:]
            and source.shape[1] < target.shape[1]
        )
        if can_expand_input:
            expanded = torch.zeros_like(target, device=source.device, dtype=source.dtype)
            expanded[:, : source.shape[1], ...] = source
            adapted_state[key] = expanded
            adapted_keys.append(key)
        else:
            incompatible.append(f"{key}: checkpoint={tuple(source.shape)} model={tuple(target.shape)}")
    if incompatible:
        raise RuntimeError(
            "Warm-start checkpoint has unsupported shape mismatches. "
            "zero_extra only expands convolution input channels.\n  - "
            + "\n  - ".join(incompatible[:20])
        )
    if not adapted_keys:
        raise RuntimeError(
            "init_channel_adaptation=zero_extra was requested, but no wider convolutional input tensor required adaptation."
        )
    return adapted_state, adapted_keys

