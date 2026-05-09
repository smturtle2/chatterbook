from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import chatterbook.converter as converter
from chatterbook import Book


@pytest.fixture(autouse=True)
def fake_book_title(monkeypatch):
    monkeypatch.setattr(converter, "get_book_title", lambda _: "테스트 책")


class FakeModel:
    sr = 24000

    def __init__(self) -> None:
        self.calls = []

    def generate(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return torch.zeros(1, 10)


class BatchFakeModel(FakeModel):
    def __init__(self) -> None:
        super().__init__()
        self.conditionals = []

    def prepare_conditionals(self, audio_prompt_path, **kwargs):
        self.conditionals.append((audio_prompt_path, kwargs))


class FailingModel:
    sr = 24000

    def generate(self, text, **kwargs):
        raise RuntimeError("model failed")


class FakeTorchaudio:
    def __init__(self) -> None:
        self.saved = []

    def save(self, path, wav, sample_rate):
        self.saved.append((path, wav, sample_rate))
        Path(path).write_bytes(b"wav")


def fake_book_dict() -> dict:
    return {
        "schema_version": 1,
        "title": "테스트 책",
        "source_path": None,
        "max_chars": 300,
        "pause_defaults": {
            "comma_pause_ms": 120,
            "sentence_pause_ms": 300,
            "paragraph_pause_ms": 600,
            "dialogue_pause_ms": 300,
        },
        "chapters": [
            {
                "index": 1,
                "title": "Intro",
                "filename": "001-intro.wav",
                "paragraphs": [
                    {
                        "index": 1,
                        "text": "hello",
                        "segments": [
                            {
                                "text": "hello",
                                "kind": "narration",
                                "pause_after_ms": 600,
                            }
                        ],
                    }
                ],
            },
            {
                "index": 2,
                "title": "Next",
                "filename": "002-next.wav",
                "paragraphs": [
                    {
                        "index": 1,
                        "text": "world",
                        "segments": [
                            {
                                "text": "world",
                                "kind": "narration",
                                "pause_after_ms": 600,
                            }
                        ],
                    }
                ],
            },
        ],
    }


def batch_book_dict() -> dict:
    data = fake_book_dict()
    data["chapters"] = [
        {
            "index": 1,
            "title": "Intro",
            "filename": "001-intro.wav",
            "paragraphs": [
                {
                    "index": 1,
                    "text": "one two three",
                    "segments": [
                        {
                            "text": "one",
                            "kind": "narration",
                            "pause_after_ms": 0,
                        },
                        {
                            "text": "two",
                            "kind": "narration",
                            "pause_after_ms": 0,
                        },
                        {
                            "text": "three",
                            "kind": "narration",
                            "pause_after_ms": 0,
                        },
                    ],
                }
            ],
        }
    ]
    return data


def test_book_serializes_paragraphs_dialogue_and_pauses(monkeypatch, tmp_path):
    monkeypatch.setattr(
        converter,
        "extract_chapters",
        lambda _: [
            SimpleNamespace(
                title="Intro",
                text="ignored",
                blocks=[
                    "렌은 말했다, “불.” 아무 일도 없었다.",
                    "다음 문단이다.",
                ],
                filename="001-intro.wav",
            )
        ],
    )

    book = Book(tmp_path / "book.epub")
    data = book.to_dict()

    assert data["schema_version"] == 1
    assert data["title"] == "테스트 책"
    assert data["chapters"][0]["title"] == "Intro"
    assert data["chapters"][0]["paragraphs"][0]["segments"] == [
        {"text": "렌은 말했다,", "kind": "narration", "pause_after_ms": 300},
        {"text": "“불.”", "kind": "dialogue", "pause_after_ms": 300},
        {"text": "아무 일도 없었다.", "kind": "narration", "pause_after_ms": 600},
    ]
    assert data["chapters"][0]["paragraphs"][1]["segments"] == [
        {"text": "다음 문단이다.", "kind": "narration", "pause_after_ms": 600}
    ]


def test_book_round_trips_from_dict():
    data = fake_book_dict()

    assert Book.from_dict(data).to_dict() == data


def test_book_from_dict_precomputes_runtime_audio_plan():
    book = Book.from_dict(batch_book_dict())

    assert book.total_segments == 3
    assert [segment.text for segment in book.chapters[0].segments] == [
        "one",
        "two",
        "three",
    ]


def test_straight_apostrophe_does_not_create_dialogue(monkeypatch, tmp_path):
    monkeypatch.setattr(
        converter,
        "extract_chapters",
        lambda _: [
            SimpleNamespace(
                title="Intro",
                text="ignored",
                blocks=["I'm here."],
                filename="001-intro.wav",
            )
        ],
    )

    book = Book(tmp_path / "book.epub")
    segments = book.to_dict()["chapters"][0]["paragraphs"][0]["segments"]

    assert segments == [
        {"text": "I'm here.", "kind": "narration", "pause_after_ms": 600}
    ]


def test_book_convert_passes_voice_path_and_style(monkeypatch, tmp_path):
    fake_model = FakeModel()
    fake_torchaudio = FakeTorchaudio()
    voice_path = tmp_path / "voice.wav"
    voice_path.write_bytes(b"voice")

    monkeypatch.setattr(converter, "_load_model", lambda **_: fake_model)
    monkeypatch.setattr(converter, "_import_torchaudio", lambda: fake_torchaudio)

    paths = Book.from_dict(fake_book_dict()).convert(
        tmp_path / "audio",
        language="ko",
        voice_path=voice_path,
        style="warm",
        output_format="wav",
        speed=1.0,
    )

    assert [path.name for path in paths] == ["001-intro.wav", "002-next.wav"]
    assert fake_model.calls[0][1]["audio_prompt_path"] == str(voice_path)
    assert fake_model.calls[0][1]["language_id"] == "ko"
    assert fake_model.calls[0][1]["exaggeration"] == 0.6
    assert fake_model.calls[0][1]["cfg_weight"] == 0.45
    assert len(fake_torchaudio.saved) == 2


def test_book_convert_prepares_voice_conditionals_per_batch(monkeypatch, tmp_path):
    fake_model = BatchFakeModel()
    voice_path = tmp_path / "voice.wav"
    voice_path.write_bytes(b"voice")

    monkeypatch.setattr(converter, "_load_model", lambda **_: fake_model)
    monkeypatch.setattr(converter, "_import_torchaudio", lambda: FakeTorchaudio())

    Book.from_dict(batch_book_dict()).convert(
        tmp_path / "audio",
        language="ko",
        voice_path=voice_path,
        output_format="wav",
        speed=1.0,
        batch_size=2,
    )

    assert fake_model.conditionals == [
        (str(voice_path), {"exaggeration": 0.5}),
        (str(voice_path), {"exaggeration": 0.5}),
    ]
    assert [call[0] for call in fake_model.calls] == ["one", "two", "three"]
    assert all(call[1]["audio_prompt_path"] is None for call in fake_model.calls)


def test_audio_batches_group_by_kind_and_size():
    segments = [
        converter.AudioSegment("one"),
        converter.AudioSegment("two"),
        converter.AudioSegment("three"),
        converter.AudioSegment("quote", kind="dialogue"),
    ]

    batches = converter._audio_batches(segments, batch_size=2)

    assert [[segment.text for segment in batch] for batch in batches] == [
        ["one", "two"],
        ["three"],
        ["quote"],
    ]


def test_explicit_generation_values_override_style(monkeypatch, tmp_path):
    fake_model = FakeModel()
    monkeypatch.setattr(converter, "_load_model", lambda **_: fake_model)
    monkeypatch.setattr(converter, "_import_torchaudio", lambda: FakeTorchaudio())

    Book.from_dict(fake_book_dict()).convert(
        tmp_path / "audio",
        language="en",
        style="dramatic",
        output_format="wav",
        speed=1.0,
        exaggeration=0.7,
        cfg_weight=0.3,
    )

    call = fake_model.calls[0][1]
    assert call["exaggeration"] == 0.7
    assert call["cfg_weight"] == 0.3


def test_book_convert_rejects_missing_voice_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        Book.from_dict(fake_book_dict()).convert(
            tmp_path / "audio",
            language="ko",
            output_format="wav",
            speed=1.0,
            voice_path=tmp_path / "missing.wav",
        )


def test_book_convert_rejects_existing_output(tmp_path):
    output_dir = tmp_path / "audio"
    output_dir.mkdir()
    (output_dir / "001-intro.wav").write_bytes(b"exists")

    with pytest.raises(FileExistsError):
        Book.from_dict(fake_book_dict()).convert(
            output_dir,
            language="ko",
            output_format="wav",
            speed=1.0,
        )


def test_book_convert_reports_generation_context(monkeypatch, tmp_path):
    monkeypatch.setattr(converter, "_load_model", lambda **_: FailingModel())
    monkeypatch.setattr(converter, "_import_torchaudio", lambda: FakeTorchaudio())

    with pytest.raises(converter.GenerationError) as exc_info:
        Book.from_dict(fake_book_dict()).convert(
            tmp_path / "audio",
            language="ko",
            output_format="wav",
            speed=1.0,
        )

    message = str(exc_info.value)
    assert "Intro" in message
    assert "hello" in message


def test_book_convert_defaults_m4b_name_from_title(monkeypatch, tmp_path):
    fake_model = FakeModel()
    commands = []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(converter, "_load_model", lambda **_: fake_model)
    monkeypatch.setattr(converter, "_import_torchaudio", lambda: FakeTorchaudio())

    def fake_run(command, **kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"m4b")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(converter.subprocess, "run", fake_run)

    path = Book.from_dict(fake_book_dict()).convert(language="ko")

    assert path == tmp_path / "테스트 책.m4b"
    assert path.read_bytes() == b"m4b"
    assert commands[0][-1] == str(path)


def test_book_convert_m4b_uses_output_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(converter, "_load_model", lambda **_: FakeModel())
    monkeypatch.setattr(converter, "_import_torchaudio", lambda: FakeTorchaudio())
    monkeypatch.setattr(
        converter.subprocess,
        "run",
        lambda command, **_: Path(command[-1]).write_bytes(b"m4b"),
    )

    output_dir = tmp_path / "out"
    path = Book.from_dict(fake_book_dict()).convert(output_dir, language="ko")

    assert path == output_dir / "테스트 책.m4b"


def test_build_ffmetadata_preserves_chapter_titles():
    chapters = [
        SimpleNamespace(title="제1장 = 시작"),
        SimpleNamespace(title="제2장 # 끝"),
    ]

    metadata = converter._build_ffmetadata("책; 제목", chapters, [1000, 2500])

    assert "title=책\\; 제목" in metadata
    assert "START=0\nEND=1000" in metadata
    assert "START=1000\nEND=3500" in metadata
    assert "title=제1장 \\= 시작" in metadata
    assert "title=제2장 \\# 끝" in metadata


def test_build_audio_segments_splits_sentences_and_oversized_chunks():
    segments = converter._build_audio_segments(
        ["첫 문장입니다. " + ("긴문장 " * 30)],
        max_chars=120,
        paragraph_pause_ms=600,
    )

    assert len(segments) > 1
    assert all(len(segment.text) <= 120 for segment in segments)
    assert segments[-1].pause_after_ms == 600


def test_build_audio_segments_preserves_sentence_pauses_by_default():
    segments = converter._build_audio_segments(
        ["첫 문장입니다. 둘째 문장입니다. 셋째 문장입니다."],
        max_chars=120,
        paragraph_pause_ms=600,
    )

    assert segments == [
        converter.AudioSegment(
            "첫 문장입니다.",
            kind="narration",
            pause_after_ms=300,
        ),
        converter.AudioSegment(
            "둘째 문장입니다.",
            kind="narration",
            pause_after_ms=300,
        ),
        converter.AudioSegment(
            "셋째 문장입니다.",
            kind="narration",
            pause_after_ms=600,
        ),
    ]


def test_build_audio_segments_keeps_dialogue_boundaries_when_compacting():
    segments = converter._build_audio_segments(
        ["렌은 말했다. “불.” 아무 일도 없었다. 다음 문장이다."],
        max_chars=120,
        paragraph_pause_ms=600,
        dialogue_pause_ms=300,
    )

    assert segments == [
        converter.AudioSegment("렌은 말했다.", kind="narration", pause_after_ms=300),
        converter.AudioSegment("“불.”", kind="dialogue", pause_after_ms=300),
        converter.AudioSegment(
            "아무 일도 없었다.",
            kind="narration",
            pause_after_ms=300,
        ),
        converter.AudioSegment(
            "다음 문장이다.",
            kind="narration",
            pause_after_ms=600,
        ),
    ]


def test_build_audio_segments_compacts_adjacent_dialogue_only_with_dialogue():
    segments = converter._build_audio_segments(
        ['“불.” “물.” 그는 숨을 골랐다. “바람.”'],
        max_chars=120,
        paragraph_pause_ms=600,
        dialogue_pause_ms=300,
    )

    assert segments == [
        converter.AudioSegment("“불.”", kind="dialogue", pause_after_ms=300),
        converter.AudioSegment("“물.”", kind="dialogue", pause_after_ms=300),
        converter.AudioSegment("그는 숨을 골랐다.", kind="narration", pause_after_ms=300),
        converter.AudioSegment("“바람.”", kind="dialogue", pause_after_ms=600),
    ]


def test_build_audio_segments_compacts_only_when_no_pause_would_be_lost():
    segments = converter._compact_audio_segments(
        [
            converter.AudioSegment("first", kind="narration", pause_after_ms=0),
            converter.AudioSegment("second", kind="narration", pause_after_ms=300),
            converter.AudioSegment("third", kind="narration", pause_after_ms=0),
        ],
        max_chars=120,
    )

    assert segments == [
        converter.AudioSegment(
            "first second",
            kind="narration",
            pause_after_ms=300,
        ),
        converter.AudioSegment("third", kind="narration", pause_after_ms=0),
    ]


def test_audio_batches_never_mix_narration_and_dialogue():
    segments = [
        converter.AudioSegment("narration one", kind="narration"),
        converter.AudioSegment("narration two", kind="narration"),
        converter.AudioSegment("dialogue one", kind="dialogue"),
        converter.AudioSegment("dialogue two", kind="dialogue"),
        converter.AudioSegment("narration three", kind="narration"),
    ]

    batches = converter._audio_batches(segments, batch_size=8)

    assert [[segment.kind for segment in batch] for batch in batches] == [
        ["narration", "narration"],
        ["dialogue", "dialogue"],
        ["narration"],
    ]


def test_silence_matches_reference_tensor():
    reference = torch.ones(1, 10, dtype=torch.float64)

    silence = converter._silence(500, 24000, like=reference)

    assert silence.shape == (1, 12000)
    assert silence.dtype == reference.dtype
    assert torch.all(silence == 0)


def test_tempo_adjusted_durations():
    assert converter._tempo_adjusted_durations([900, 1800], speed=0.9) == [
        1000,
        2000,
    ]


def test_run_ffmpeg_m4b_adds_tempo_filter(monkeypatch, tmp_path):
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(converter.subprocess, "run", fake_run)

    converter._run_ffmpeg_m4b(
        tmp_path / "concat.txt",
        tmp_path / "metadata.txt",
        tmp_path / "book.m4b",
        bitrate="128k",
        overwrite=True,
        speed=0.9,
    )

    assert "-filter:a" in commands[0]
    assert "atempo=0.9" in commands[0]


def test_book_progress_uses_single_total_bar(monkeypatch):
    calls = []

    class FakeTqdm:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setitem(
        __import__("sys").modules,
        "tqdm.auto",
        SimpleNamespace(tqdm=FakeTqdm),
    )

    progress = converter._book_progress(
        Book.from_dict(fake_book_dict()),
        enabled=True,
        output_format="m4b",
    )

    assert isinstance(progress, FakeTqdm)
    assert len(calls) == 1
    assert calls[0]["desc"] == "EPUB -> M4B"
    assert calls[0]["unit"] == "segment"
    assert calls[0]["colour"] == "green"
    assert calls[0]["total"] == 2


def test_pkg_resources_shim_supports_resource_filename(monkeypatch):
    monkeypatch.delitem(sys.modules, "pkg_resources", raising=False)

    converter._install_pkg_resources_shim()
    import pkg_resources

    path = pkg_resources.resource_filename("chatterbook", "__init__.py")

    assert path.endswith("chatterbook/__init__.py")


def test_generate_audio_suppresses_model_output(capsys):
    class LoudModel:
        def generate(self, text, **kwargs):
            print("Sampling: noisy stdout")
            print("Sampling: noisy stderr", file=sys.stderr)
            logging.getLogger(
                "chatterbox.models.t3.inference.alignment_stream_analyzer"
            ).warning("forcing EOS token")
            return text, kwargs

    result = converter._generate_audio(
        LoudModel(),
        "hello",
        show_progress=True,
        language_id="ko",
    )

    captured = capsys.readouterr()
    assert result == ("hello", {"language_id": "ko"})
    assert captured.out == ""
    assert captured.err == ""


def test_generate_audio_keeps_model_output_when_progress_disabled(capsys):
    class LoudModel:
        def generate(self, text, **kwargs):
            print("visible")
            return text

    result = converter._generate_audio(LoudModel(), "hello", show_progress=False)

    captured = capsys.readouterr()
    assert result == "hello"
    assert captured.out == "visible\n"
