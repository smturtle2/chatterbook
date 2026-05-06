# chatterbook

Convert EPUB books into M4B audiobooks with Chatterbox Multilingual TTS.

```python
from chatterbook import convert_epub

convert_epub(
    "book.epub",
    language="ko",
    voice_path="voices/narrator.wav",
    style="warm",
    max_chars=300,
)
```

By default, the output filename is read from the EPUB title metadata and written
as `Title.m4b` in the current directory. You can also pass a path:

```python
convert_epub("book.epub", "audiobooks/book.m4b", language="ko")
```

`voice_path` is an optional short WAV reference clip for voice cloning. If it is
omitted, Chatterbox's bundled default conditionals are used.

Long chapters are split on EPUB paragraph boundaries and adjacent short
paragraphs are grouped up to `max_chars`, then assembled into one chapterized
M4B.

M4B output requires `ffmpeg` on your `PATH`. During conversion, chatterbook shows
colored `tqdm` progress bars for chapters and text chunks. Pass
`show_progress=False` to disable them.

To export chapter WAV files instead of one M4B:

```python
convert_epub(
    "book.epub",
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
convert_epub(
    "book.epub",
    language="ko",
    exaggeration=0.7,
    cfg_weight=0.3,
)
```
