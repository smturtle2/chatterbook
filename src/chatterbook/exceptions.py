from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence


class ChatterbookError(Exception):
    """Base exception for chatterbook errors."""


class UnsupportedLanguageError(ChatterbookError, ValueError):
    def __init__(self, language: str, supported_languages: Sequence[str]) -> None:
        supported = ", ".join(supported_languages)
        super().__init__(
            f"Unsupported language '{language}'. Supported languages: {supported}"
        )


class UnknownStyleError(ChatterbookError, ValueError):
    def __init__(self, style: str, supported_styles: Sequence[str]) -> None:
        supported = ", ".join(supported_styles)
        super().__init__(f"Unknown style '{style}'. Supported styles: {supported}")


class OutputExistsError(ChatterbookError, FileExistsError):
    def __init__(self, paths: Sequence[Path]) -> None:
        joined = ", ".join(str(path) for path in paths)
        super().__init__(f"Output file already exists: {joined}")
