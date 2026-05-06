from __future__ import annotations

from ebooklib import epub

from chatterbook.epub import extract_chapters


def test_extract_chapters_uses_spine_order(tmp_path):
    book = epub.EpubBook()
    book.set_identifier("fixture")
    book.set_title("Fixture")
    book.set_language("en")

    first = epub.EpubHtml(title="First", file_name="first.xhtml", lang="en")
    first.content = "<html><body><h1>First Title</h1><p>Hello first.</p></body></html>"
    second = epub.EpubHtml(title="Second", file_name="second.xhtml", lang="en")
    second.content = "<html><body><h1>Second Title</h1><p>Hello second.</p></body></html>"

    book.add_item(first)
    book.add_item(second)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", first, second]

    path = tmp_path / "fixture.epub"
    epub.write_epub(str(path), book)

    chapters = extract_chapters(path)

    assert [chapter.title for chapter in chapters] == ["First Title", "Second Title"]
    assert [chapter.filename for chapter in chapters] == [
        "001-first-title.wav",
        "002-second-title.wav",
    ]
    assert chapters[0].text == "First Title Hello first."
