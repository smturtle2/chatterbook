# chatterbook

Convert EPUB books into chapter WAV audio with Chatterbox Multilingual TTS.

```python
from chatterbook import convert_epub

convert_epub(
    "book.epub",
    output_dir="audio",
    language="ko",
    voice_path="voices/narrator.wav",
    style="warm",
)
```

`voice_path` is an optional short WAV reference clip for voice cloning. If it is
omitted, Chatterbox's bundled default conditionals are used.

## Styles

- `neutral`: balanced default
- `warm`: slightly softer narration
- `dramatic`: more expressive narration

You can override a style with explicit generation values:

```python
convert_epub(
    "book.epub",
    output_dir="audio",
    language="ko",
    exaggeration=0.7,
    cfg_weight=0.3,
)
```
