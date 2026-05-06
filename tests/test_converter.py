from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import logging
import sys

import pytest
import torch

import chatterbook.converter as converter
from chatterbook import convert_epub


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


def test_convert_epub_passes_voice_path_and_style(monkeypatch, tmp_path):
    fake_model = FakeModel()
    fake_torchaudio = FakeTorchaudio()
    voice_path = tmp_path / "voice.wav"
    voice_path.write_bytes(b"voice")

    monkeypatch.setattr(
        converter,
        "extract_chapters",
        lambda _: [
            SimpleNamespace(
                title="Intro",
                text="hello",
                blocks=["hello"],
                filename="001-intro.wav",
            ),
            SimpleNamespace(
                title="Next",
                text="world",
                blocks=["world"],
                filename="002-next.wav",
            ),
        ],
    )
    monkeypatch.setattr(converter, "_load_model", lambda **_: fake_model)
    monkeypatch.setattr(converter, "_import_torchaudio", lambda: fake_torchaudio)

    paths = convert_epub(
        tmp_path / "book.epub",
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


def test_explicit_generation_values_override_style(monkeypatch, tmp_path):
    fake_model = FakeModel()
    monkeypatch.setattr(
        converter,
        "extract_chapters",
        lambda _: [
            SimpleNamespace(
                title="Intro",
                text="hello",
                blocks=["hello"],
                filename="001-intro.wav",
            )
        ],
    )
    monkeypatch.setattr(converter, "_load_model", lambda **_: fake_model)
    monkeypatch.setattr(converter, "_import_torchaudio", lambda: FakeTorchaudio())

    convert_epub(
        tmp_path / "book.epub",
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


def test_convert_epub_rejects_missing_voice_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        convert_epub(
            tmp_path / "book.epub",
            tmp_path / "audio",
            language="ko",
            output_format="wav",
            speed=1.0,
            voice_path=tmp_path / "missing.wav",
        )


def test_convert_epub_rejects_existing_output(monkeypatch, tmp_path):
    monkeypatch.setattr(
        converter,
        "extract_chapters",
        lambda _: [
            SimpleNamespace(
                title="Intro",
                text="hello",
                blocks=["hello"],
                filename="001-intro.wav",
            )
        ],
    )
    output_dir = tmp_path / "audio"
    output_dir.mkdir()
    (output_dir / "001-intro.wav").write_bytes(b"exists")

    with pytest.raises(FileExistsError):
        convert_epub(
            tmp_path / "book.epub",
            output_dir,
            language="ko",
            output_format="wav",
            speed=1.0,
        )


def test_convert_epub_splits_long_chapters(monkeypatch, tmp_path):
    fake_model = FakeModel()
    monkeypatch.setattr(
        converter,
        "extract_chapters",
        lambda _: [
            SimpleNamespace(
                title="Intro",
                text="문장입니다. " * 80,
                blocks=["문장입니다. " * 80],
                filename="001-intro.wav",
            )
        ],
    )
    monkeypatch.setattr(converter, "_load_model", lambda **_: fake_model)
    monkeypatch.setattr(converter, "_import_torchaudio", lambda: FakeTorchaudio())

    convert_epub(
        tmp_path / "book.epub",
        tmp_path / "audio",
        language="ko",
        output_format="wav",
        speed=1.0,
        max_chars=120,
    )

    assert len(fake_model.calls) > 1
    assert all(len(text) <= 120 for text, _ in fake_model.calls)


def test_convert_epub_reports_generation_context(monkeypatch, tmp_path):
    monkeypatch.setattr(
        converter,
        "extract_chapters",
        lambda _: [
            SimpleNamespace(
                title="Intro",
                text="hello",
                blocks=["문제가 나는 문장입니다."],
                filename="001-intro.wav",
            )
        ],
    )
    monkeypatch.setattr(converter, "_load_model", lambda **_: FailingModel())
    monkeypatch.setattr(converter, "_import_torchaudio", lambda: FakeTorchaudio())

    with pytest.raises(converter.GenerationError) as exc_info:
        convert_epub(
            tmp_path / "book.epub",
            tmp_path / "audio",
            language="ko",
            output_format="wav",
            speed=1.0,
        )

    message = str(exc_info.value)
    assert "Intro" in message
    assert "문제가 나는 문장입니다." in message


def test_convert_epub_defaults_m4b_name_from_epub_title(monkeypatch, tmp_path):
    fake_model = FakeModel()
    fake_torchaudio = FakeTorchaudio()
    commands = []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        converter,
        "extract_chapters",
        lambda _: [
            SimpleNamespace(
                title="제1장",
                text="hello",
                blocks=["hello"],
                filename="001-1.wav",
            )
        ],
    )
    monkeypatch.setattr(converter, "_load_model", lambda **_: fake_model)
    monkeypatch.setattr(converter, "_import_torchaudio", lambda: fake_torchaudio)

    def fake_run(command, **kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"m4b")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(converter.subprocess, "run", fake_run)

    path = convert_epub(tmp_path / "book.epub", language="ko")

    assert path == tmp_path / "테스트 책.m4b"
    assert path.read_bytes() == b"m4b"
    assert commands[0][-1] == str(path)


def test_convert_epub_m4b_uses_output_directory(monkeypatch, tmp_path):
    fake_model = FakeModel()
    monkeypatch.setattr(
        converter,
        "extract_chapters",
        lambda _: [
            SimpleNamespace(
                title="제1장",
                text="hello",
                blocks=["hello"],
                filename="001-1.wav",
            )
        ],
    )
    monkeypatch.setattr(converter, "_load_model", lambda **_: fake_model)
    monkeypatch.setattr(converter, "_import_torchaudio", lambda: FakeTorchaudio())
    monkeypatch.setattr(
        converter.subprocess,
        "run",
        lambda command, **_: Path(command[-1]).write_bytes(b"m4b"),
    )

    output_dir = tmp_path / "out"
    path = convert_epub(tmp_path / "book.epub", output_dir, language="ko")

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


def test_build_audio_segments_preserves_paragraphs_and_dialogue():
    segments = converter._build_audio_segments(
        [
            "렌은 말했다. “불.” 아무 일도 일어나지 않았다.",
            "다음 문단이다.",
        ],
        max_chars=300,
        paragraph_pause_ms=600,
        dialogue_pause_ms=300,
    )

    assert segments == [
        converter.AudioSegment("렌은 말했다.", is_dialogue=False, pause_after_ms=300),
        converter.AudioSegment("“불.”", is_dialogue=True, pause_after_ms=300),
        converter.AudioSegment(
            "아무 일도 일어나지 않았다.",
            is_dialogue=False,
            pause_after_ms=600,
        ),
        converter.AudioSegment("다음 문단이다.", is_dialogue=False, pause_after_ms=600),
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


def test_epub_progress_uses_single_total_bar(monkeypatch):
    calls = []

    class FakeTqdm:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setitem(
        __import__("sys").modules,
        "tqdm.auto",
        SimpleNamespace(tqdm=FakeTqdm),
    )
    chapters = [
        SimpleNamespace(blocks=["a", "b"]),
        SimpleNamespace(blocks=["c " * 80]),
    ]

    progress = converter._epub_progress(
        chapters,
        max_chars=120,
        paragraph_pause_ms=600,
        dialogue_pause_ms=300,
        enabled=True,
    )

    assert isinstance(progress, FakeTqdm)
    assert len(calls) == 1
    assert calls[0]["desc"] == "EPUB"
    assert calls[0]["unit"] == "chunk"
    assert calls[0]["colour"] == "green"
    assert calls[0]["total"] == 4


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
