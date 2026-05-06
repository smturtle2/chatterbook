from __future__ import annotations

from pathlib import Path
from typing import Any

from .epub import extract_chapters
from .exceptions import OutputExistsError, UnsupportedLanguageError
from .styles import resolve_style

SUPPORTED_LANGUAGES = {
    "ar",
    "da",
    "de",
    "el",
    "en",
    "es",
    "fi",
    "fr",
    "he",
    "hi",
    "it",
    "ja",
    "ko",
    "ms",
    "nl",
    "no",
    "pl",
    "pt",
    "ru",
    "sv",
    "sw",
    "tr",
    "zh",
}


def convert_epub(
    epub_path: str | Path,
    output_dir: str | Path,
    *,
    language: str,
    voice_path: str | Path | None = None,
    style: str = "neutral",
    device: str | None = None,
    t3_model: str | None = None,
    overwrite: bool = False,
    exaggeration: float | None = None,
    cfg_weight: float | None = None,
    temperature: float = 0.8,
    repetition_penalty: float = 1.2,
    min_p: float = 0.05,
    top_p: float = 1.0,
) -> list[Path]:
    """Convert an EPUB into chapter WAV files.

    The Chatterbox model is loaded lazily so importing chatterbook stays cheap.
    """
    language_id = language.lower()
    if language_id not in SUPPORTED_LANGUAGES:
        raise UnsupportedLanguageError(language, sorted(SUPPORTED_LANGUAGES))

    epub_path = Path(epub_path)
    output_dir = Path(output_dir)
    voice_prompt_path = _resolve_voice_path(voice_path)
    generation_style = resolve_style(
        style,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
    )

    chapters = extract_chapters(epub_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = [output_dir / chapter.filename for chapter in chapters]
    existing_paths = [path for path in output_paths if path.exists()]
    if existing_paths and not overwrite:
        raise OutputExistsError(existing_paths)

    model = _load_model(device=device, t3_model=t3_model)
    torchaudio = _import_torchaudio()

    written_paths: list[Path] = []
    for chapter, output_path in zip(chapters, output_paths, strict=True):
        wav = model.generate(
            chapter.text,
            language_id=language_id,
            audio_prompt_path=str(voice_prompt_path) if voice_prompt_path else None,
            exaggeration=generation_style.exaggeration,
            cfg_weight=generation_style.cfg_weight,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
        )
        torchaudio.save(str(output_path), wav, model.sr)
        written_paths.append(output_path)

    return written_paths


def _resolve_voice_path(voice_path: str | Path | None) -> Path | None:
    if voice_path is None:
        return None

    path = Path(voice_path)
    if not path.is_file():
        raise FileNotFoundError(f"voice_path does not exist: {path}")
    return path


def _load_model(*, device: str | None, t3_model: str | None) -> Any:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    selected_device = device or _default_device()
    return ChatterboxMultilingualTTS.from_pretrained(
        device=selected_device,
        t3_model=t3_model,
    )


def _default_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _import_torchaudio() -> Any:
    import torchaudio

    return torchaudio
