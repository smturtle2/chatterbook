from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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
        max_chars=120,
    )

    assert len(fake_model.calls) > 1
    assert all(len(text) <= 120 for text, _ in fake_model.calls)


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
