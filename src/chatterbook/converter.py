from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Literal

from .epub import extract_chapters, get_book_title
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
    output_path: str | Path | None = None,
    *,
    language: str,
    voice_path: str | Path | None = None,
    style: str = "neutral",
    output_format: Literal["m4b", "wav"] = "m4b",
    bitrate: str = "128k",
    keep_temp: bool = False,
    show_progress: bool = True,
    device: str | None = None,
    t3_model: str | None = None,
    overwrite: bool = False,
    max_chars: int = 300,
    exaggeration: float | None = None,
    cfg_weight: float | None = None,
    temperature: float = 0.8,
    repetition_penalty: float = 1.2,
    min_p: float = 0.05,
    top_p: float = 1.0,
) -> Path | list[Path]:
    """Convert an EPUB into an M4B audiobook or chapter WAV files.

    The Chatterbox model is loaded lazily so importing chatterbook stays cheap.
    """
    language_id = language.lower()
    if language_id not in SUPPORTED_LANGUAGES:
        raise UnsupportedLanguageError(language, sorted(SUPPORTED_LANGUAGES))

    epub_path = Path(epub_path)
    book_title = get_book_title(epub_path)
    output = _resolve_output_path(
        output_path,
        output_format=output_format,
        book_title=book_title,
    )
    voice_prompt_path = _resolve_voice_path(voice_path)
    generation_style = resolve_style(
        style,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
    )

    chapters = extract_chapters(epub_path)
    if output_format == "wav":
        return _convert_epub_to_wavs(
            chapters,
            output,
            language_id=language_id,
            voice_prompt_path=voice_prompt_path,
            generation_style=generation_style,
            device=device,
            t3_model=t3_model,
            overwrite=overwrite,
            max_chars=max_chars,
            show_progress=show_progress,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
        )

    if output_format != "m4b":
        raise ValueError("output_format must be 'm4b' or 'wav'")

    return _convert_epub_to_m4b(
        chapters,
        output,
        book_title=book_title,
        language_id=language_id,
        voice_prompt_path=voice_prompt_path,
        generation_style=generation_style,
        device=device,
        t3_model=t3_model,
        overwrite=overwrite,
        max_chars=max_chars,
        bitrate=bitrate,
        keep_temp=keep_temp,
        show_progress=show_progress,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        min_p=min_p,
        top_p=top_p,
    )


def _convert_epub_to_wavs(
    chapters: list[Any],
    output_dir: Path,
    *,
    language_id: str,
    voice_prompt_path: Path | None,
    generation_style: Any,
    device: str | None,
    t3_model: str | None,
    overwrite: bool,
    max_chars: int,
    show_progress: bool,
    temperature: float,
    repetition_penalty: float,
    min_p: float,
    top_p: float,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = [output_dir / chapter.filename for chapter in chapters]
    existing_paths = [path for path in output_paths if path.exists()]
    if existing_paths and not overwrite:
        raise OutputExistsError(existing_paths)

    model = _load_model(device=device, t3_model=t3_model)
    torchaudio = _import_torchaudio()

    written_paths: list[Path] = []
    for chapter, output_path in _progress(
        list(zip(chapters, output_paths, strict=True)),
        desc="Chapters",
        unit="chapter",
        colour="cyan",
        enabled=show_progress,
    ):
        wavs = []
        texts = _split_blocks(chapter.blocks, max_chars=max_chars)
        for text in _progress(
            texts,
            desc=chapter.title,
            unit="chunk",
            colour="magenta",
            enabled=show_progress,
            leave=False,
        ):
            wavs.append(
                model.generate(
                    text,
                    language_id=language_id,
                    audio_prompt_path=str(voice_prompt_path)
                    if voice_prompt_path
                    else None,
                    exaggeration=generation_style.exaggeration,
                    cfg_weight=generation_style.cfg_weight,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    min_p=min_p,
                    top_p=top_p,
                )
            )
        wav = _concat_wavs(wavs)
        torchaudio.save(str(output_path), wav, model.sr)
        written_paths.append(output_path)

    return written_paths


def _resolve_output_path(
    output_path: str | Path | None,
    *,
    output_format: str,
    book_title: str,
) -> Path:
    if output_format not in {"m4b", "wav"}:
        raise ValueError("output_format must be 'm4b' or 'wav'")

    default_name = _safe_filename(book_title)
    if output_path is None:
        if output_format == "m4b":
            return Path.cwd() / f"{default_name}.m4b"
        return Path.cwd() / default_name

    path = Path(output_path)
    if output_format == "m4b" and (path.is_dir() or path.suffix == ""):
        return path / f"{default_name}.m4b"
    return path


def _convert_epub_to_m4b(
    chapters: list[Any],
    output_path: Path,
    *,
    book_title: str,
    language_id: str,
    voice_prompt_path: Path | None,
    generation_style: Any,
    device: str | None,
    t3_model: str | None,
    overwrite: bool,
    max_chars: int,
    bitrate: str,
    keep_temp: bool,
    show_progress: bool,
    temperature: float,
    repetition_penalty: float,
    min_p: float,
    top_p: float,
) -> Path:
    if output_path.exists() and not overwrite:
        raise OutputExistsError([output_path])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_context: tempfile.TemporaryDirectory[str] | None = None
    if keep_temp:
        temp_dir = output_path.with_suffix(output_path.suffix + ".tmp")
        if temp_dir.exists() and not overwrite:
            raise OutputExistsError([temp_dir])
        shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True)
    else:
        temp_context = tempfile.TemporaryDirectory(
            prefix=f"{output_path.stem}-", dir=output_path.parent
        )
        temp_dir = Path(temp_context.name)

    try:
        chapter_paths, chapter_durations = _write_chapter_wavs(
            chapters,
            temp_dir,
            language_id=language_id,
            voice_prompt_path=voice_prompt_path,
            generation_style=generation_style,
            device=device,
            t3_model=t3_model,
            max_chars=max_chars,
            show_progress=show_progress,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
        )
        metadata_path = temp_dir / "metadata.txt"
        concat_path = temp_dir / "concat.txt"
        metadata_path.write_text(
            _build_ffmetadata(book_title, chapters, chapter_durations),
            encoding="utf-8",
        )
        concat_path.write_text(
            "".join(f"file '{_escape_concat_path(path)}'\n" for path in chapter_paths),
            encoding="utf-8",
        )
        _run_ffmpeg_m4b(
            concat_path,
            metadata_path,
            output_path,
            bitrate=bitrate,
            overwrite=overwrite,
        )
    finally:
        if temp_context is not None:
            temp_context.cleanup()

    return output_path


def _duration_ms(wav: Any, sample_rate: int) -> int:
    return round(wav.shape[-1] * 1000 / sample_rate)


def _build_ffmetadata(
    book_title: str,
    chapters: list[Any],
    chapter_durations: list[int],
) -> str:
    lines = [";FFMETADATA1", f"title={_escape_ffmetadata(book_title)}"]
    start = 0
    for chapter, duration in zip(chapters, chapter_durations, strict=True):
        end = start + duration
        lines.extend(
            [
                "[CHAPTER]",
                "TIMEBASE=1/1000",
                f"START={start}",
                f"END={end}",
                f"title={_escape_ffmetadata(chapter.title)}",
            ]
        )
        start = end
    return "\n".join(lines) + "\n"


def _escape_ffmetadata(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("=", "\\=")
        .replace(";", "\\;")
        .replace("#", "\\#")
    )


def _escape_concat_path(path: Path) -> str:
    return str(path).replace("'", "'\\''")


def _run_ffmpeg_m4b(
    concat_path: Path,
    metadata_path: Path,
    output_path: Path,
    *,
    bitrate: str,
    overwrite: bool,
) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-y" if overwrite else "-n",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-i",
        str(metadata_path),
        "-map_metadata",
        "1",
        "-map_chapters",
        "1",
        "-c:a",
        "aac",
        "-b:a",
        bitrate,
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError("ffmpeg is required for M4B output") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip()
        raise RuntimeError(f"ffmpeg failed to create M4B: {detail}") from exc


def _write_chapter_wavs(
    chapters: list[Any],
    output_dir: Path,
    *,
    language_id: str,
    voice_prompt_path: Path | None,
    generation_style: Any,
    device: str | None,
    t3_model: str | None,
    max_chars: int,
    show_progress: bool,
    temperature: float,
    repetition_penalty: float,
    min_p: float,
    top_p: float,
) -> tuple[list[Path], list[int]]:
    model = _load_model(device=device, t3_model=t3_model)
    torchaudio = _import_torchaudio()
    chapter_paths: list[Path] = []
    chapter_durations: list[int] = []

    for chapter in _progress(
        chapters,
        desc="Chapters",
        unit="chapter",
        colour="cyan",
        enabled=show_progress,
    ):
        wavs = []
        texts = _split_blocks(chapter.blocks, max_chars=max_chars)
        for text in _progress(
            texts,
            desc=chapter.title,
            unit="chunk",
            colour="magenta",
            enabled=show_progress,
            leave=False,
        ):
            wavs.append(
                model.generate(
                    text,
                    language_id=language_id,
                    audio_prompt_path=str(voice_prompt_path)
                    if voice_prompt_path
                    else None,
                    exaggeration=generation_style.exaggeration,
                    cfg_weight=generation_style.cfg_weight,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    min_p=min_p,
                    top_p=top_p,
                )
            )

        wav = _concat_wavs(wavs)
        output_path = output_dir / chapter.filename
        torchaudio.save(str(output_path), wav, model.sr)
        chapter_paths.append(output_path)
        chapter_durations.append(_duration_ms(wav, model.sr))

    return chapter_paths, chapter_durations


def _safe_filename(value: str) -> str:
    import re

    value = re.sub(r"[\\/:*?\"<>|]+", " ", value)
    value = " ".join(value.split()).strip()
    return value or "book"


def _progress(
    items: Any,
    *,
    desc: str,
    unit: str,
    colour: str,
    enabled: bool,
    leave: bool = True,
) -> Any:
    if not enabled:
        return items

    from tqdm.auto import tqdm

    return tqdm(items, desc=desc, unit=unit, colour=colour, leave=leave)


def _resolve_voice_path(voice_path: str | Path | None) -> Path | None:
    if voice_path is None:
        return None

    path = Path(voice_path)
    if not path.is_file():
        raise FileNotFoundError(f"voice_path does not exist: {path}")
    return path


def _split_text(text: str, *, max_chars: int) -> list[str]:
    return _split_blocks([text], max_chars=max_chars)


def _split_blocks(blocks: list[str], *, max_chars: int) -> list[str]:
    if max_chars < 100:
        raise ValueError("max_chars must be at least 100")

    chunks: list[str] = []
    current = ""
    for block in blocks:
        block = " ".join(block.split())
        if not block:
            continue
        if len(block) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_oversized(block, max_chars=max_chars))
        elif not current:
            current = block
        elif len(current) + 1 + len(block) <= max_chars:
            current = f"{current} {block}"
        else:
            chunks.append(current)
            current = block

    if current:
        chunks.append(current)

    return chunks


def _split_oversized(text: str, *, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    words = text.split()
    current = ""
    for word in words:
        if len(word) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(
                word[index : index + max_chars]
                for index in range(0, len(word), max_chars)
            )
        elif not current:
            current = word
        elif len(current) + 1 + len(word) <= max_chars:
            current = f"{current} {word}"
        else:
            chunks.append(current)
            current = word

    if current:
        chunks.append(current)

    return chunks


def _concat_wavs(wavs: list[Any]) -> Any:
    if not wavs:
        raise ValueError("No audio chunks were generated")
    if len(wavs) == 1:
        return wavs[0]

    import torch

    return torch.cat(wavs, dim=-1)


def _load_model(*, device: str | None, t3_model: str | None) -> Any:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    selected_device = device or _default_device()
    kwargs = {"device": selected_device}
    if t3_model is not None:
        kwargs["t3_model"] = t3_model
    return ChatterboxMultilingualTTS.from_pretrained(**kwargs)


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
