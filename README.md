# chatterbook

Convert EPUB books into M4B audiobooks with Chatterbox Multilingual TTS.

```python
from chatterbook import Book

book = Book("book.epub")
book.convert(
    language="ko",
    voice_path="voices/narrator.wav",
    style="warm",
    speed=0.9,
)
```

`Book` reads the EPUB immediately and builds a JSON-compatible representation
with title, chapters, paragraphs, narration/dialogue segments, and pause timing.
You can serialize and restore that representation without reading the EPUB again:

```python
data = book.to_dict()
book = Book.from_dict(data)
book.convert("audiobooks/book.m4b", language="ko")
```

By default, the output filename is read from the EPUB title metadata and written
as `Title.m4b` in the current directory. You can also pass a path or directory.

`voice_path` is an optional short WAV reference clip for voice cloning. If it is
omitted, Chatterbox's bundled default conditionals are used.

Chapters are split into paragraph and speech segments when the `Book` is built.
Quoted dialogue is marked separately and generated with slightly more expressive
settings. Commas, sentence endings, paragraph breaks, and narration/dialogue
transitions add short pauses to improve audiobook pacing.

The default audiobook pacing is intentionally slower than raw TTS:

- `speed=0.9`
- `comma_pause_ms=120`
- `sentence_pause_ms=300`
- `paragraph_pause_ms=600`
- `dialogue_pause_ms=300`

M4B output requires `ffmpeg` on your `PATH`. During conversion, chatterbook shows
one colored `tqdm` progress bar for the whole EPUB. Pass `show_progress=False`
to disable it.

To export chapter WAV files instead of one M4B:

```python
book.convert(
    "audio",
    language="ko",
    output_format="wav",
)
```

## Styles

- `neutral`: balanced default
- `warm`: slightly softer narration
- `dramatic`: more expressive narration

You can override a style with explicit generation values:

```python
book.convert(
    language="ko",
    exaggeration=0.7,
    cfg_weight=0.3,
)
```
