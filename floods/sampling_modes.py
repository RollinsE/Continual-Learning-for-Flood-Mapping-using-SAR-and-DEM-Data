SAMPLING_FLAGS = {
    "weighted_sampling": "entropy-weighted sampling",
    "foreground_balanced_sampling": "foreground-balanced sampling",
    "stratified_sampling": "foreground-ratio stratified sampling",
    "event_balanced_sampling": "event-balanced foreground-ratio sampling",
    "hard_example_sampling": "audit-based hard-example sampling",
    "hard_negative_region_sampling": "audit-guided hard-negative region sampling",
    "hard_positive_region_sampling": "audit-guided hard-positive region sampling",
}


def active_sampling_modes(data_config) -> list[str]:
    """Return the enabled replacement-sampling modes in a dataset config."""
    return [flag for flag in SAMPLING_FLAGS if bool(getattr(data_config, flag, False))]


def validate_sampling_modes(data_config) -> None:
    """Reject ambiguous sampler settings before a long training run starts."""
    active = active_sampling_modes(data_config)
    if len(active) > 1:
        readable = ", ".join(f"{flag} ({SAMPLING_FLAGS[flag]})" for flag in active)
        raise ValueError(
            "Only one training sampler can be enabled at a time. "
            f"Enabled samplers: {readable}. Use the matching --no-... flags or a clean config."
        )
