from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup
from ebooklib import epub


@dataclass(frozen=True)
class Chapter:
    title: str
    text: str
    filename: str


def extract_chapters(epub_path: str | Path) -> list[Chapter]:
    book = epub.read_epub(str(epub_path))
    chapters: list[Chapter] = []

    for itemref in book.spine:
        item_id = itemref[0]
        item = book.get_item_with_id(item_id)
        if item is None:
            continue
        if isinstance(item, epub.EpubNav):
            continue

        html = item.get_content()
        title, text = _extract_html_text(html)
        if not text:
            continue

        chapter_number = len(chapters) + 1
        title = title or f"Chapter {chapter_number}"
        filename = f"{chapter_number:03d}-{_slugify(title)}.wav"
        chapters.append(Chapter(title=title, text=text, filename=filename))

    return chapters


def _extract_html_text(html: bytes) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav"]):
        tag.decompose()

    heading = soup.find(["h1", "h2", "h3"])
    title = heading.get_text(" ", strip=True) if heading else ""
    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return title, text


def _slugify(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return value or "chapter"
