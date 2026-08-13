from __future__ import annotations

"""Central pretrained-weight and model-candidate registry.

The registry is intentionally shared by ordinary training, event-level cross-validation,
OOF threshold calibration, and checkpoint evaluation.  A provider is therefore a model
initialisation choice, not a separate training path.
"""

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

from floods.modalities import canonicalize_modalities


@dataclass(frozen=True)
class WeightSourceSpec:
    name: str
    provider: str
    default_encoder: str | None
    default_decoder: str | None
    adapter: str
    normalization_mode: str
    allowed_modality_sets: tuple[tuple[str, ...], ...]
    foundation_input_size: int | None = None
    description: str = ""

    def supports(self, modalities: Sequence[str]) -> bool:
        value = tuple(canonicalize_modalities(modalities))
        return not self.allowed_modality_sets or value in self.allowed_modality_sets

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["allowed_modality_sets"] = [list(value) for value in self.allowed_modality_sets]
        return payload


@dataclass(frozen=True)
class ResolvedModelSpec:
    weights_source: str
    provider: str
    encoder: str
    decoder: str
    pretrained: bool
    modalities: tuple[str, ...]
    normalization_mode: str
    adapter: str
    foundation_input_size: int | None
    label: str

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["modalities"] = list(self.modalities)
        return payload


@dataclass(frozen=True)
class CandidateSpec:
    weights_source: str
    decoder: str | None = None
    encoder: str | None = None
    label: str | None = None


SOURCES: dict[str, WeightSourceSpec] = {
    "random": WeightSourceSpec(
        name="random", provider="timm", default_encoder=None, default_decoder=None,
        adapter="timm", normalization_mode="robust_percentile", allowed_modality_sets=(),
        description="Randomly initialised timm encoder.",
    ),
    "imagenet": WeightSourceSpec(
        name="imagenet", provider="timm", default_encoder=None, default_decoder=None,
        adapter="timm", normalization_mode="robust_percentile", allowed_modality_sets=(),
        description="ImageNet-pretrained timm encoder.",
    ),
    "ssl4eo_s1_moco": WeightSourceSpec(
        name="ssl4eo_s1_moco", provider="torchgeo", default_encoder="resnet50",
        default_decoder="unet", adapter="torchgeo_resnet50_moco",
        normalization_mode="ssl4eo_s1", allowed_modality_sets=(("vv", "vh"),),
        foundation_input_size=224, description="SSL4EO-S12 Sentinel-1 MoCo ResNet-50.",
    ),
    "ssl4eo_s1_decur": WeightSourceSpec(
        name="ssl4eo_s1_decur", provider="torchgeo", default_encoder="resnet50",
        default_decoder="unet", adapter="torchgeo_resnet50_decur",
        normalization_mode="ssl4eo_s1", allowed_modality_sets=(("vv", "vh"),),
        foundation_input_size=224, description="DeCUR Sentinel-1 ResNet-50.",
    ),
    "fgmae_sar_vit_small": WeightSourceSpec(
        name="fgmae_sar_vit_small", provider="torchgeo", default_encoder="vit_small_patch16_224",
        default_decoder="segformer", adapter="torchgeo_vit_small_fgmae",
        normalization_mode="ssl4eo_s1", allowed_modality_sets=(("vv", "vh"),),
        foundation_input_size=224, description="FG-MAE Sentinel-1 ViT-S/16.",
    ),
    "fgmae_sar_vit_base": WeightSourceSpec(
        name="fgmae_sar_vit_base", provider="torchgeo", default_encoder="vit_base_patch16_224",
        default_decoder="segformer", adapter="torchgeo_vit_base_fgmae",
        normalization_mode="ssl4eo_s1", allowed_modality_sets=(("vv", "vh"),),
        foundation_input_size=224, description="FG-MAE Sentinel-1 ViT-B/16.",
    ),
    "croma_sar_base": WeightSourceSpec(
        name="croma_sar_base", provider="torchgeo", default_encoder="croma_sar_base",
        default_decoder="segformer", adapter="croma_sar_base",
        normalization_mode="robust_percentile", allowed_modality_sets=(("vv", "vh"),),
        foundation_input_size=120, description="CROMA base SAR encoder.",
    ),
    "terramind_v1_tiny": WeightSourceSpec(
        name="terramind_v1_tiny", provider="terratorch", default_encoder="terramind_v1_tiny",
        default_decoder="segformer", adapter="terramind",
        normalization_mode="terramind_v1",
        allowed_modality_sets=(("vv", "vh"), ("vv", "vh", "dem")),
        foundation_input_size=224, description="TerraMind v1 Tiny, S1GRD with optional DEM.",
    ),
    "terramind_v1_small": WeightSourceSpec(
        name="terramind_v1_small", provider="terratorch", default_encoder="terramind_v1_small",
        default_decoder="segformer", adapter="terramind",
        normalization_mode="terramind_v1",
        allowed_modality_sets=(("vv", "vh"), ("vv", "vh", "dem")),
        foundation_input_size=224, description="TerraMind v1 Small, S1GRD with optional DEM.",
    ),
    "terramind_v1_base": WeightSourceSpec(
        name="terramind_v1_base", provider="terratorch", default_encoder="terramind_v1_base",
        default_decoder="segformer", adapter="terramind",
        normalization_mode="terramind_v1",
        allowed_modality_sets=(("vv", "vh"), ("vv", "vh", "dem")),
        foundation_input_size=224, description="TerraMind v1 Base, S1GRD with optional DEM.",
    ),
}


def normalize_source_name(value: str | None) -> str:
    key = str(value or "random").strip().lower().replace("-", "_")
    aliases = {
        "none": "random", "scratch": "random", "image_net": "imagenet",
        "terramind_tiny": "terramind_v1_tiny", "terramind_small": "terramind_v1_small",
        "terramind_base": "terramind_v1_base", "croma": "croma_sar_base",
        "ssl4eo_moco": "ssl4eo_s1_moco", "ssl4eo_decur": "ssl4eo_s1_decur",
        "fgmae_sar_small": "fgmae_sar_vit_small", "fgmae_sar_base": "fgmae_sar_vit_base",
    }
    return aliases.get(key, key)


def get_weight_source(value: str) -> WeightSourceSpec:
    key = normalize_source_name(value)
    if key not in SOURCES:
        raise ValueError(f"Unknown weights source {value!r}. Available: {', '.join(sorted(SOURCES))}")
    return SOURCES[key]


def list_weight_sources() -> list[dict]:
    return [SOURCES[name].to_dict() for name in sorted(SOURCES)]


def parse_candidate_spec(value: str) -> CandidateSpec:
    """Parse SOURCE[:DECODER[:ENCODER[:LABEL]]].

    Foundation providers normally need only SOURCE.  ImageNet/random candidates should
    include decoder and encoder, for example ``imagenet:unet:resnet34``.
    """
    parts = [part.strip() for part in str(value).split(":")]
    if not parts or not parts[0] or len(parts) > 4:
        raise ValueError(
            "Candidate specifications must be SOURCE[:DECODER[:ENCODER[:LABEL]]], "
            "for example terramind_v1_tiny or imagenet:unet:resnet34."
        )
    while len(parts) < 4:
        parts.append("")
    return CandidateSpec(
        weights_source=normalize_source_name(parts[0]),
        decoder=parts[1] or None,
        encoder=parts[2] or None,
        label=parts[3] or None,
    )


def _safe_label(value: str) -> str:
    import re
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("_")
    return label or "candidate"


def resolve_model_spec(
    *,
    weights_source: str,
    modalities: Sequence[str],
    encoder: str | None = None,
    decoder: str | None = None,
    pretrained: bool | None = None,
    label: str | None = None,
) -> ResolvedModelSpec:
    source = get_weight_source(weights_source)
    mods = tuple(canonicalize_modalities(modalities))
    if not source.supports(mods):
        allowed = ["+".join(item) for item in source.allowed_modality_sets]
        raise ValueError(
            f"weights_source={source.name!r} does not support modalities={list(mods)}. "
            f"Allowed modality sets: {allowed}"
        )

    resolved_encoder = str(encoder or source.default_encoder or "").strip()
    resolved_decoder = str(decoder or source.default_decoder or "").strip().lower().replace("-", "_")
    if not resolved_encoder or not resolved_decoder:
        raise ValueError(
            f"weights_source={source.name!r} requires an explicit encoder and decoder. "
            "Use --encoder and --decoder."
        )

    if source.adapter not in {"timm"}:
        if encoder and source.default_encoder and encoder != source.default_encoder:
            raise ValueError(
                f"weights_source={source.name!r} requires encoder={source.default_encoder!r}; "
                f"got {encoder!r}."
            )
        resolved_encoder = str(source.default_encoder)
        if decoder and source.default_decoder and decoder != source.default_decoder:
            # Decoder overrides are deliberately allowed for feature-compatible adapters.
            resolved_decoder = str(decoder)

    use_pretrained = bool(pretrained) if pretrained is not None else source.name != "random"
    if source.name == "random":
        use_pretrained = False
    if source.name not in {"random", "imagenet"} and not use_pretrained:
        # Evaluation reconstructs provider architectures with pretrained=False before
        # loading the complete flood-mapping checkpoint.  This is valid and avoids downloads.
        pass

    default_label = source.name if source.name not in {"random", "imagenet"} else f"{source.name}_{resolved_decoder}_{resolved_encoder}"
    return ResolvedModelSpec(
        weights_source=source.name,
        provider=source.provider,
        encoder=resolved_encoder,
        decoder=resolved_decoder,
        pretrained=use_pretrained,
        modalities=mods,
        normalization_mode=source.normalization_mode,
        adapter=source.adapter,
        foundation_input_size=source.foundation_input_size,
        label=_safe_label(label or default_label),
    )


def resolve_candidate(candidate: CandidateSpec, modalities: Sequence[str], *, legacy_pretrained: bool | None = None) -> ResolvedModelSpec:
    return resolve_model_spec(
        weights_source=candidate.weights_source,
        modalities=modalities,
        encoder=candidate.encoder,
        decoder=candidate.decoder,
        pretrained=legacy_pretrained,
        label=candidate.label,
    )


def apply_resolved_model_to_config(
    config,
    resolved: ResolvedModelSpec,
    *,
    evaluation: bool = False,
    force_normalization: bool = False,
):
    """Apply one resolved registry entry to a TrainConfig in-place.

    Provider adapters always impose their provider input convention.  Ordinary
    ImageNet/random encoders retain the normalization already declared by the base
    configuration unless the caller explicitly asks to force the registry default.
    This preserves authoritative configurations written by older package releases.
    """
    config.model.weights_source = resolved.weights_source
    config.model.encoder = resolved.encoder
    config.model.decoder = resolved.decoder
    config.model.pretrained = False if evaluation else bool(resolved.pretrained)
    config.model.foundation_input_size = int(resolved.foundation_input_size or config.image_size)
    if resolved.modalities:
        config.data.input_modalities = list(resolved.modalities)
        config.data.in_channels = len(resolved.modalities)
        config.data.include_dem = "dem" in resolved.modalities

    provider_mode = resolved.normalization_mode in {"terramind_v1", "ssl4eo_s1"}
    if provider_mode or force_normalization:
        config.data.normalization_mode = resolved.normalization_mode
    if provider_mode:
        config.data.normalization_stats_path = None
        # Provider-pretrained SAR inputs should not receive brightness/noise transforms.
        config.data.augmentation_profile = "geometric"
        config.data.disable_sar_noise = True
    if hasattr(config.data, "refresh_cache_hash"):
        config.data.refresh_cache_hash()
    return config
