"""Build data/dev_book.txt: 50 consecutive sentences from a public-domain novel.

The dev set is for quick iteration with ``scripts/dev.py``. ``data/benchmark.jsonl``
stays the held-out test, so method changes are not tuned on the reported numbers.

Source: David Graham Phillips, "White Magic: A Novel" (1910), Project Gutenberg
eBook #65432, public domain in the USA. The excerpt is the start of chapter X.
The book is obscure on purpose: a famous one may be memorised by the model
(Pride and Prejudice scored 15 points higher in a spot check).

Typographic quotes become ASCII and italics markers are dropped, matching the
ASCII-only benchmark. Sentences are split per paragraph with the benchmark's
splitter and written one per line. The resulting file is committed, so
experiments do not depend on the download.

    uv run python scripts/build_dev_set.py
"""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

from build_dataset import sentences

URL = "https://www.gutenberg.org/cache/epub/65432/pg65432.txt"
CHAPTER_HEADING = "\nX\n\nBEATRICE IN CHAINS\n"
SENTENCES = 50
ASCII = {"“": '"', "”": '"', "‘": "'", "’": "'", "_": ""}
ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    with urllib.request.urlopen(URL, timeout=60) as resp:
        book = resp.read().decode("utf-8").replace("\r\n", "\n")
    chapter = book[book.index(CHAPTER_HEADING) + len(CHAPTER_HEADING) :]
    for src, dst in ASCII.items():
        chapter = chapter.replace(src, dst)
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", chapter) if p.strip()]
    picked = [s for p in paragraphs for s in sentences(p)][:SENTENCES]
    out = ROOT / "data" / "dev_book.txt"
    out.write_text("".join(s + "\n" for s in picked), encoding="utf-8")
    words = sum(len(re.findall(r"\w+", s)) for s in picked)
    print(f"wrote {len(picked)} sentences, {words} words to {out}")


if __name__ == "__main__":
    main()
