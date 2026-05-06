from __future__ import annotations

from dataclasses import dataclass

from .exceptions import UnknownStyleError


@dataclass(frozen=True)
class GenerationStyle:
    exaggeration: float
    cfg_weight: float


STYLE_PRESETS = {
    "neutral": GenerationStyle(exaggeration=0.5, cfg_weight=0.5),
    "warm": GenerationStyle(exaggeration=0.6, cfg_weight=0.45),
    "dramatic": GenerationStyle(exaggeration=0.8, cfg_weight=0.55),
}


def resolve_style(
    style: str,
    *,
    exaggeration: float | None = None,
    cfg_weight: float | None = None,
) -> GenerationStyle:
    try:
        preset = STYLE_PRESETS[style]
    except KeyError as exc:
        raise UnknownStyleError(style, sorted(STYLE_PRESETS)) from exc

    return GenerationStyle(
        exaggeration=preset.exaggeration if exaggeration is None else exaggeration,
        cfg_weight=preset.cfg_weight if cfg_weight is None else cfg_weight,
    )
