from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import chatterbook.converter as converter
from chatterbook import convert_epub


class FakeModel:
    sr = 24000

    def __init__(self) -> None:
        self.calls = []

    def generate(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return [[0.0]]


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
            SimpleNamespace(text="hello", filename="001-intro.wav"),
            SimpleNamespace(text="world", filename="002-next.wav"),
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
        lambda _: [SimpleNamespace(text="hello", filename="001-intro.wav")],
    )
    monkeypatch.setattr(converter, "_load_model", lambda **_: fake_model)
    monkeypatch.setattr(converter, "_import_torchaudio", lambda: FakeTorchaudio())

    convert_epub(
        tmp_path / "book.epub",
        tmp_path / "audio",
        language="en",
        style="dramatic",
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
            voice_path=tmp_path / "missing.wav",
        )


def test_convert_epub_rejects_existing_output(monkeypatch, tmp_path):
    monkeypatch.setattr(
        converter,
        "extract_chapters",
        lambda _: [SimpleNamespace(text="hello", filename="001-intro.wav")],
    )
    output_dir = tmp_path / "audio"
    output_dir.mkdir()
    (output_dir / "001-intro.wav").write_bytes(b"exists")

    with pytest.raises(FileExistsError):
        convert_epub(tmp_path / "book.epub", output_dir, language="ko")
