from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .epub import extract_chapters, get_book_title
from .exceptions import GenerationError, OutputExistsError, UnsupportedLanguageError
from .styles import resolve_style

SCHEMA_VERSION = 1
DEFAULT_MAX_CHARS = 300
DEFAULT_COMMA_PAUSE_MS = 120
DEFAULT_SENTENCE_PAUSE_MS = 300
DEFAULT_PARAGRAPH_PAUSE_MS = 600
DEFAULT_DIALOGUE_PAUSE_MS = 300

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


@dataclass(frozen=True)
class AudioSegment:
    text: str
    kind: Literal["narration", "dialogue"] = "narration"
    pause_after_ms: int = 0

    @property
    def is_dialogue(self) -> bool:
        return self.kind == "dialogue"

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "kind": self.kind,
            "pause_after_ms": self.pause_after_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AudioSegment:
        return cls(
            text=str(data["text"]),
            kind=_segment_kind(data.get("kind", "narration")),
            pause_after_ms=int(data.get("pause_after_ms", 0)),
        )


@dataclass(frozen=True)
class BookParagraph:
    index: int
    text: str
    segments: list[AudioSegment]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "segments": [segment.to_dict() for segment in self.segments],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BookParagraph:
        return cls(
            index=int(data["index"]),
            text=str(data["text"]),
            segments=[
                AudioSegment.from_dict(segment)
                for segment in data.get("segments", data.get("sentences", []))
            ],
        )


@dataclass(frozen=True)
class BookChapter:
    index: int
    title: str
    filename: str
    paragraphs: list[BookParagraph]
    segments: list[AudioSegment]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "title": self.title,
            "filename": self.filename,
            "paragraphs": [paragraph.to_dict() for paragraph in self.paragraphs],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BookChapter:
        paragraphs = [
            BookParagraph.from_dict(paragraph)
            for paragraph in data.get("paragraphs", [])
        ]
        return cls(
            index=int(data["index"]),
            title=str(data["title"]),
            filename=str(data["filename"]),
            paragraphs=paragraphs,
            segments=_flatten_paragraph_segments(paragraphs),
        )


class Book:
    """Serializable EPUB book representation ready for TTS conversion."""

    def __init__(
        self,
        epub_path: str | Path,
        *,
        max_chars: int = DEFAULT_MAX_CHARS,
        comma_pause_ms: int = DEFAULT_COMMA_PAUSE_MS,
        sentence_pause_ms: int = DEFAULT_SENTENCE_PAUSE_MS,
        paragraph_pause_ms: int = DEFAULT_PARAGRAPH_PAUSE_MS,
        dialogue_pause_ms: int = DEFAULT_DIALOGUE_PAUSE_MS,
    ) -> None:
        _validate_pause_values(
            comma_pause_ms=comma_pause_ms,
            sentence_pause_ms=sentence_pause_ms,
            paragraph_pause_ms=paragraph_pause_ms,
            dialogue_pause_ms=dialogue_pause_ms,
        )
        if max_chars < 100:
            raise ValueError("max_chars must be at least 100")

        self.source_path = Path(epub_path)
        self.title = get_book_title(self.source_path)
        self.max_chars = max_chars
        self.comma_pause_ms = comma_pause_ms
        self.sentence_pause_ms = sentence_pause_ms
        self.paragraph_pause_ms = paragraph_pause_ms
        self.dialogue_pause_ms = dialogue_pause_ms
        self.chapters = _build_book_chapters(
            extract_chapters(self.source_path),
            max_chars=max_chars,
            comma_pause_ms=comma_pause_ms,
            sentence_pause_ms=sentence_pause_ms,
            paragraph_pause_ms=paragraph_pause_ms,
            dialogue_pause_ms=dialogue_pause_ms,
        )
        self.total_segments = _count_segments(self.chapters)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "title": self.title,
            "source_path": str(self.source_path) if self.source_path else None,
            "max_chars": self.max_chars,
            "pause_defaults": {
                "comma_pause_ms": self.comma_pause_ms,
                "sentence_pause_ms": self.sentence_pause_ms,
                "paragraph_pause_ms": self.paragraph_pause_ms,
                "dialogue_pause_ms": self.dialogue_pause_ms,
            },
            "chapters": [chapter.to_dict() for chapter in self.chapters],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Book:
        schema_version = int(data.get("schema_version", 0))
        if schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported book schema version: {schema_version}")

        pause_defaults = data.get("pause_defaults", {})
        book = cls.__new__(cls)
        book.source_path = (
            Path(data["source_path"]) if data.get("source_path") is not None else None
        )
        book.title = str(data["title"])
        book.max_chars = int(data.get("max_chars", DEFAULT_MAX_CHARS))
        book.comma_pause_ms = int(
            pause_defaults.get("comma_pause_ms", DEFAULT_COMMA_PAUSE_MS)
        )
        book.sentence_pause_ms = int(
            pause_defaults.get("sentence_pause_ms", DEFAULT_SENTENCE_PAUSE_MS)
        )
        book.paragraph_pause_ms = int(
            pause_defaults.get("paragraph_pause_ms", DEFAULT_PARAGRAPH_PAUSE_MS)
        )
        book.dialogue_pause_ms = int(
            pause_defaults.get("dialogue_pause_ms", DEFAULT_DIALOGUE_PAUSE_MS)
        )
        book.chapters = [
            BookChapter.from_dict(chapter) for chapter in data.get("chapters", [])
        ]
        book.total_segments = _count_segments(book.chapters)
        return book

    def convert(
        self,
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
        batch_size: int = 8,
        overwrite: bool = False,
        speed: float = 0.9,
        exaggeration: float | None = None,
        cfg_weight: float | None = None,
        dialogue_exaggeration: float = 0.7,
        dialogue_cfg_weight: float = 0.45,
        temperature: float = 0.8,
        repetition_penalty: float = 1.2,
        min_p: float = 0.05,
        top_p: float = 1.0,
    ) -> Path | list[Path]:
        language_id = language.lower()
        if language_id not in SUPPORTED_LANGUAGES:
            raise UnsupportedLanguageError(language, sorted(SUPPORTED_LANGUAGES))
        if speed <= 0:
            raise ValueError("speed must be greater than 0")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        output = _resolve_output_path(
            output_path,
            output_format=output_format,
            book_title=self.title,
        )
        voice_prompt_path = _resolve_voice_path(voice_path)
        generation_style = resolve_style(
            style,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )

        if output_format == "wav":
            return _convert_book_to_wavs(
                self,
                output,
                language_id=language_id,
                voice_prompt_path=voice_prompt_path,
                generation_style=generation_style,
                device=device,
                t3_model=t3_model,
                batch_size=batch_size,
                overwrite=overwrite,
                speed=speed,
                dialogue_exaggeration=dialogue_exaggeration,
                dialogue_cfg_weight=dialogue_cfg_weight,
                show_progress=show_progress,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )

        if output_format != "m4b":
            raise ValueError("output_format must be 'm4b' or 'wav'")

        return _convert_book_to_m4b(
            self,
            output,
            language_id=language_id,
            voice_prompt_path=voice_prompt_path,
            generation_style=generation_style,
            device=device,
            t3_model=t3_model,
            batch_size=batch_size,
            overwrite=overwrite,
            bitrate=bitrate,
            keep_temp=keep_temp,
            speed=speed,
            dialogue_exaggeration=dialogue_exaggeration,
            dialogue_cfg_weight=dialogue_cfg_weight,
            show_progress=show_progress,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
        )


def _build_book_chapters(
    chapters: list[Any],
    *,
    max_chars: int,
    comma_pause_ms: int,
    sentence_pause_ms: int,
    paragraph_pause_ms: int,
    dialogue_pause_ms: int,
) -> list[BookChapter]:
    book_chapters = []
    for chapter_index, chapter in enumerate(chapters, start=1):
        paragraphs = []
        for paragraph_index, block in enumerate(chapter.blocks, start=1):
            text = " ".join(block.split())
            if not text:
                continue
            paragraphs.append(
                BookParagraph(
                    index=paragraph_index,
                    text=text,
                    segments=_build_audio_segments(
                        [text],
                        max_chars=max_chars,
                        comma_pause_ms=comma_pause_ms,
                        sentence_pause_ms=sentence_pause_ms,
                        paragraph_pause_ms=paragraph_pause_ms,
                        dialogue_pause_ms=dialogue_pause_ms,
                    ),
                )
            )
        book_chapters.append(
            BookChapter(
                index=chapter_index,
                title=chapter.title,
                filename=chapter.filename,
                paragraphs=paragraphs,
                segments=_flatten_paragraph_segments(paragraphs),
            )
        )
    return book_chapters


def _flatten_paragraph_segments(paragraphs: list[BookParagraph]) -> list[AudioSegment]:
    return [
        segment
        for paragraph in paragraphs
        for segment in paragraph.segments
    ]


def _count_segments(chapters: list[BookChapter]) -> int:
    return sum(len(chapter.segments) for chapter in chapters)


def _convert_book_to_wavs(
    book: Book,
    output_dir: Path,
    *,
    language_id: str,
    voice_prompt_path: Path | None,
    generation_style: Any,
    device: str | None,
    t3_model: str | None,
    batch_size: int,
    overwrite: bool,
    speed: float,
    dialogue_exaggeration: float,
    dialogue_cfg_weight: float,
    show_progress: bool,
    temperature: float,
    repetition_penalty: float,
    min_p: float,
    top_p: float,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = [output_dir / chapter.filename for chapter in book.chapters]
    existing_paths = [path for path in output_paths if path.exists()]
    if existing_paths and not overwrite:
        raise OutputExistsError(existing_paths)

    model = _load_model(device=device, t3_model=t3_model)
    torchaudio = _import_torchaudio()
    progress = _book_progress(book, enabled=show_progress, output_format="wav")

    written_paths: list[Path] = []
    try:
        for chapter, output_path in zip(book.chapters, output_paths, strict=True):
            wav, duration_ms = _render_chapter_wav(
                chapter,
                model,
                language_id=language_id,
                voice_prompt_path=voice_prompt_path,
                generation_style=generation_style,
                batch_size=batch_size,
                dialogue_exaggeration=dialogue_exaggeration,
                dialogue_cfg_weight=dialogue_cfg_weight,
                show_progress=show_progress,
                progress=progress,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )
            torchaudio.save(str(output_path), wav, model.sr)
            if speed != 1.0:
                _run_ffmpeg_wav_speed(output_path, speed=speed)
            written_paths.append(output_path)
            _ = duration_ms
    finally:
        if progress is not None:
            progress.close()

    return written_paths


def _convert_book_to_m4b(
    book: Book,
    output_path: Path,
    *,
    language_id: str,
    voice_prompt_path: Path | None,
    generation_style: Any,
    device: str | None,
    t3_model: str | None,
    batch_size: int,
    overwrite: bool,
    bitrate: str,
    keep_temp: bool,
    speed: float,
    dialogue_exaggeration: float,
    dialogue_cfg_weight: float,
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
            book,
            temp_dir,
            language_id=language_id,
            voice_prompt_path=voice_prompt_path,
            generation_style=generation_style,
            device=device,
            t3_model=t3_model,
            batch_size=batch_size,
            dialogue_exaggeration=dialogue_exaggeration,
            dialogue_cfg_weight=dialogue_cfg_weight,
            show_progress=show_progress,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
        )
        metadata_path = temp_dir / "metadata.txt"
        concat_path = temp_dir / "concat.txt"
        metadata_durations = _tempo_adjusted_durations(chapter_durations, speed=speed)
        metadata_path.write_text(
            _build_ffmetadata(book.title, book.chapters, metadata_durations),
            encoding="utf-8",
        )
        concat_path.write_text(
            "".join(f"file '{_escape_concat_path(path)}'\n" for path in chapter_paths),
            encoding="utf-8",
        )
        if show_progress:
            print("Creating M4B with ffmpeg...")
        _run_ffmpeg_m4b(
            concat_path,
            metadata_path,
            output_path,
            bitrate=bitrate,
            overwrite=overwrite,
            speed=speed,
        )
    finally:
        if temp_context is not None:
            temp_context.cleanup()

    return output_path


def _write_chapter_wavs(
    book: Book,
    output_dir: Path,
    *,
    language_id: str,
    voice_prompt_path: Path | None,
    generation_style: Any,
    device: str | None,
    t3_model: str | None,
    batch_size: int,
    dialogue_exaggeration: float,
    dialogue_cfg_weight: float,
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
    progress = _book_progress(book, enabled=show_progress, output_format="m4b")

    try:
        for chapter in book.chapters:
            wav, duration_ms = _render_chapter_wav(
                chapter,
                model,
                language_id=language_id,
                voice_prompt_path=voice_prompt_path,
                generation_style=generation_style,
                batch_size=batch_size,
                dialogue_exaggeration=dialogue_exaggeration,
                dialogue_cfg_weight=dialogue_cfg_weight,
                show_progress=show_progress,
                progress=progress,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )
            output_path = output_dir / chapter.filename
            torchaudio.save(str(output_path), wav, model.sr)
            chapter_paths.append(output_path)
            chapter_durations.append(duration_ms)
    finally:
        if progress is not None:
            progress.close()

    return chapter_paths, chapter_durations


def _render_chapter_wav(
    chapter: BookChapter,
    model: Any,
    *,
    language_id: str,
    voice_prompt_path: Path | None,
    generation_style: Any,
    batch_size: int,
    dialogue_exaggeration: float,
    dialogue_cfg_weight: float,
    show_progress: bool,
    progress: Any,
    temperature: float,
    repetition_penalty: float,
    min_p: float,
    top_p: float,
) -> tuple[Any, int]:
    wavs = []
    rendered_count = 0
    for batch in _audio_batches(chapter.segments, batch_size=batch_size):
        exaggeration = (
            dialogue_exaggeration
            if batch[0].is_dialogue
            else generation_style.exaggeration
        )
        cfg_weight = (
            dialogue_cfg_weight if batch[0].is_dialogue else generation_style.cfg_weight
        )
        prompt_path = _prepare_batch_conditionals(
            model,
            voice_prompt_path=voice_prompt_path,
            exaggeration=exaggeration,
        )
        for segment in batch:
            try:
                wav = _generate_audio(
                    model,
                    segment.text,
                    show_progress=show_progress,
                    language_id=language_id,
                    audio_prompt_path=prompt_path,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    min_p=min_p,
                    top_p=top_p,
                )
            except Exception as exc:
                raise GenerationError(chapter.title, segment.text) from exc
            wavs.append(wav)
            if segment.pause_after_ms:
                wavs.append(_silence(segment.pause_after_ms, model.sr, like=wav))
            rendered_count += 1
            if progress is not None:
                progress.set_postfix(
                    chapter=chapter.title[:24],
                    segment=f"{rendered_count}/{len(chapter.segments)}",
                    batch=len(batch),
                    refresh=False,
                )
                progress.update(1)

    wav = _concat_wavs(wavs)
    return wav, _duration_ms(wav, model.sr)


def _audio_batches(
    segments: list[AudioSegment],
    *,
    batch_size: int,
) -> list[list[AudioSegment]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    batches: list[list[AudioSegment]] = []
    current: list[AudioSegment] = []
    for segment in segments:
        if (
            current
            and (segment.kind != current[-1].kind or len(current) >= batch_size)
        ):
            batches.append(current)
            current = []
        current.append(segment)
    if current:
        batches.append(current)
    return batches


def _prepare_batch_conditionals(
    model: Any,
    *,
    voice_prompt_path: Path | None,
    exaggeration: float,
) -> str | None:
    if voice_prompt_path is None:
        return None
    if not hasattr(model, "prepare_conditionals"):
        return str(voice_prompt_path)

    model.prepare_conditionals(str(voice_prompt_path), exaggeration=exaggeration)
    return None


def _build_audio_segments(
    blocks: list[str],
    *,
    max_chars: int,
    comma_pause_ms: int = DEFAULT_COMMA_PAUSE_MS,
    sentence_pause_ms: int = DEFAULT_SENTENCE_PAUSE_MS,
    paragraph_pause_ms: int = DEFAULT_PARAGRAPH_PAUSE_MS,
    dialogue_pause_ms: int = DEFAULT_DIALOGUE_PAUSE_MS,
) -> list[AudioSegment]:
    _validate_pause_values(
        comma_pause_ms=comma_pause_ms,
        sentence_pause_ms=sentence_pause_ms,
        paragraph_pause_ms=paragraph_pause_ms,
        dialogue_pause_ms=dialogue_pause_ms,
    )
    if max_chars < 100:
        raise ValueError("max_chars must be at least 100")

    segments: list[AudioSegment] = []
    for block in blocks:
        block = " ".join(block.split())
        if not block:
            continue

        paragraph_segments = _split_dialogue(
            block,
            max_chars=max_chars,
            comma_pause_ms=comma_pause_ms,
            sentence_pause_ms=sentence_pause_ms,
        )
        paragraph_segments = _compact_audio_segments(
            paragraph_segments,
            max_chars=max_chars,
        )
        if not paragraph_segments:
            continue

        start_index = len(segments)
        for segment in paragraph_segments:
            if segments and (segment.is_dialogue != segments[-1].is_dialogue):
                previous = segments[-1]
                segments[-1] = AudioSegment(
                    text=previous.text,
                    kind=previous.kind,
                    pause_after_ms=max(previous.pause_after_ms, dialogue_pause_ms),
                )
            segments.append(segment)

        last = segments[-1]
        segments[-1] = AudioSegment(
            text=last.text,
            kind=last.kind,
            pause_after_ms=max(last.pause_after_ms, paragraph_pause_ms),
        )
        if start_index < len(segments) - 1:
            previous = segments[-2]
            if previous.is_dialogue != last.is_dialogue:
                segments[-2] = AudioSegment(
                    text=previous.text,
                    kind=previous.kind,
                    pause_after_ms=max(previous.pause_after_ms, dialogue_pause_ms),
                )

    return segments


def _compact_audio_segments(
    segments: list[AudioSegment],
    *,
    max_chars: int,
) -> list[AudioSegment]:
    compacted: list[AudioSegment] = []
    current: AudioSegment | None = None

    for segment in segments:
        if current is None:
            current = segment
            continue

        combined_text = f"{current.text} {segment.text}"
        if (
            current.kind == segment.kind
            and current.pause_after_ms == 0
            and len(combined_text) <= max_chars
        ):
            current = AudioSegment(
                text=combined_text,
                kind=current.kind,
                pause_after_ms=segment.pause_after_ms,
            )
            continue

        compacted.append(current)
        current = segment

    if current is not None:
        compacted.append(current)
    return compacted


def _split_dialogue(
    text: str,
    *,
    max_chars: int,
    comma_pause_ms: int,
    sentence_pause_ms: int,
) -> list[AudioSegment]:
    quote_pairs = {
        "“": "”",
        '"': '"',
        "‘": "’",
    }
    segments: list[AudioSegment] = []
    index = 0

    while index < len(text):
        next_quote_index = -1
        next_quote = ""
        for quote in quote_pairs:
            found = text.find(quote, index)
            if found != -1 and (next_quote_index == -1 or found < next_quote_index):
                next_quote_index = found
                next_quote = quote

        if next_quote_index == -1:
            segments.extend(
                _segments_from_text(
                    text[index:],
                    kind="narration",
                    max_chars=max_chars,
                    comma_pause_ms=comma_pause_ms,
                    sentence_pause_ms=sentence_pause_ms,
                )
            )
            break

        segments.extend(
            _segments_from_text(
                text[index:next_quote_index],
                kind="narration",
                max_chars=max_chars,
                comma_pause_ms=comma_pause_ms,
                sentence_pause_ms=sentence_pause_ms,
            )
        )

        closing_quote = quote_pairs[next_quote]
        closing_index = text.find(closing_quote, next_quote_index + 1)
        if closing_index == -1:
            segments.extend(
                _segments_from_text(
                    text[next_quote_index:],
                    kind="narration",
                    max_chars=max_chars,
                    comma_pause_ms=comma_pause_ms,
                    sentence_pause_ms=sentence_pause_ms,
                )
            )
            break

        segments.extend(
            _segments_from_text(
                text[next_quote_index : closing_index + 1],
                kind="dialogue",
                max_chars=max_chars,
                comma_pause_ms=comma_pause_ms,
                sentence_pause_ms=sentence_pause_ms,
            )
        )
        index = closing_index + 1

    return [segment for segment in segments if segment.text]


def _segments_from_text(
    text: str,
    *,
    kind: Literal["narration", "dialogue"],
    max_chars: int,
    comma_pause_ms: int,
    sentence_pause_ms: int,
) -> list[AudioSegment]:
    text = " ".join(text.split())
    if not text:
        return []

    segments: list[AudioSegment] = []
    for sentence in _split_sentences(text):
        pause_after_ms = _punctuation_pause(
            sentence,
            comma_pause_ms=comma_pause_ms,
            sentence_pause_ms=sentence_pause_ms,
        )
        chunks = _split_oversized(sentence, max_chars=max_chars)
        for chunk in chunks[:-1]:
            segments.append(AudioSegment(text=chunk, kind=kind, pause_after_ms=0))
        segments.append(
            AudioSegment(text=chunks[-1], kind=kind, pause_after_ms=pause_after_ms)
        )
    return segments


def _split_sentences(text: str) -> list[str]:
    sentence_endings = ".?!。！？"
    parts: list[str] = []
    start = 0
    for index, char in enumerate(text):
        if char not in sentence_endings:
            continue
        end = index + 1
        while end < len(text) and text[end] in '"”’':
            end += 1
        part = text[start:end].strip()
        if part:
            parts.append(part)
        start = end

    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _punctuation_pause(
    text: str,
    *,
    comma_pause_ms: int,
    sentence_pause_ms: int,
) -> int:
    stripped = text.rstrip('"”’')
    if not stripped:
        return 0
    if stripped[-1] in ".?!。！？":
        return sentence_pause_ms
    if stripped[-1] in ",，、;；:":
        return comma_pause_ms
    return 0


def _validate_pause_values(
    *,
    comma_pause_ms: int,
    sentence_pause_ms: int,
    paragraph_pause_ms: int,
    dialogue_pause_ms: int,
) -> None:
    values = {
        "comma_pause_ms": comma_pause_ms,
        "sentence_pause_ms": sentence_pause_ms,
        "paragraph_pause_ms": paragraph_pause_ms,
        "dialogue_pause_ms": dialogue_pause_ms,
    }
    for name, value in values.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative")


def _segment_kind(value: Any) -> Literal["narration", "dialogue"]:
    if value not in {"narration", "dialogue"}:
        raise ValueError(f"Unknown segment kind: {value}")
    return value


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
    speed: float,
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
    ]
    if speed != 1.0:
        command.extend(["-filter:a", f"atempo={speed}"])
    command.extend(
        [
            "-c:a",
            "aac",
            "-b:a",
            bitrate,
            str(output_path),
        ]
    )
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError("ffmpeg is required for M4B output") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip()
        raise RuntimeError(f"ffmpeg failed to create M4B: {detail}") from exc


def _safe_filename(value: str) -> str:
    value = re.sub(r"[\\/:*?\"<>|]+", " ", value)
    value = " ".join(value.split()).strip()
    return value or "book"


def _tempo_adjusted_durations(chapter_durations: list[int], *, speed: float) -> list[int]:
    if speed <= 0:
        raise ValueError("speed must be greater than 0")
    if speed == 1.0:
        return chapter_durations
    return [round(duration / speed) for duration in chapter_durations]


def _run_ffmpeg_wav_speed(output_path: Path, *, speed: float) -> None:
    if speed <= 0:
        raise ValueError("speed must be greater than 0")

    temp_path = output_path.with_suffix(output_path.suffix + ".tempo.wav")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i",
        str(output_path),
        "-filter:a",
        f"atempo={speed}",
        str(temp_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError("ffmpeg is required for speed adjustment") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip()
        raise RuntimeError(f"ffmpeg failed to adjust WAV speed: {detail}") from exc
    temp_path.replace(output_path)


def _silence(duration_ms: int, sample_rate: int, *, like: Any | None = None) -> Any:
    import torch

    samples = round(sample_rate * duration_ms / 1000)
    kwargs = {}
    if like is not None:
        kwargs["device"] = like.device
        kwargs["dtype"] = like.dtype
    return torch.zeros(1, samples, **kwargs)


def _book_progress(book: Book, *, enabled: bool, output_format: str) -> Any:
    if not enabled:
        return None

    from tqdm.auto import tqdm

    return tqdm(
        total=book.total_segments,
        desc=f"EPUB -> {output_format.upper()}",
        unit="segment",
        colour="green",
        dynamic_ncols=True,
    )


def _generate_audio(
    model: Any,
    text: str,
    *,
    show_progress: bool,
    **kwargs: Any,
) -> Any:
    with _suppress_model_progress(enabled=show_progress):
        return model.generate(text, **kwargs)


@contextmanager
def _suppress_model_progress(*, enabled: bool) -> Any:
    if not enabled:
        with nullcontext():
            yield
        return

    with open("/dev/null", "w", encoding="utf-8") as sink:
        with redirect_stdout(sink), redirect_stderr(sink), _suppress_chatterbox_logs():
            yield


@contextmanager
def _suppress_chatterbox_logs() -> Any:
    logger_names = [
        "chatterbox.models.t3.inference.alignment_stream_analyzer",
    ]
    previous_levels = {}
    previous_disabled = {}
    for name in logger_names:
        logger = logging.getLogger(name)
        previous_levels[name] = logger.level
        previous_disabled[name] = logger.disabled
        logger.setLevel(logging.ERROR)
        logger.disabled = True

    try:
        yield
    finally:
        for name in logger_names:
            logger = logging.getLogger(name)
            logger.setLevel(previous_levels[name])
            logger.disabled = previous_disabled[name]


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
    _install_pkg_resources_shim()
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    selected_device = device or _default_device()
    kwargs = {"device": selected_device}
    if t3_model is not None:
        kwargs["t3_model"] = t3_model
    return ChatterboxMultilingualTTS.from_pretrained(**kwargs)


def _install_pkg_resources_shim() -> None:
    import importlib.resources
    import sys
    import types

    if "pkg_resources" in sys.modules:
        return

    module = types.ModuleType("pkg_resources")

    def resource_filename(package_or_requirement: str, resource_name: str) -> str:
        return str(importlib.resources.files(package_or_requirement) / resource_name)

    module.resource_filename = resource_filename
    sys.modules["pkg_resources"] = module


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
