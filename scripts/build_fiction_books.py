"""Build data/fiction_books.json: the LoRA book split plus many more English fiction training books.

    uv run --frozen python scripts/build_fiction_books.py --train-books 300

The first ten entries are data/lora_books.json unchanged and in the same order, so
`finetune prepare` shuffles the validation and test books exactly as before and
the earlier validation samples stay a prefix of the new ones. New books are drawn
in a seeded random order from the Project Gutenberg catalog: English texts with a
fiction subject, no poetry or drama, at most two per author, none by the authors
of the validation, test or dev books, and at least --min-sentences sentence units.

Project Gutenberg blocks automated downloads from its main site, so books come
from the PGLAF mirror (byte-identical files) with a delay between requests. Raw
files are cached where `finetune prepare` looks for them; checksums are pinned.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from neural_compression.finetune import book_sentences, digest  # noqa: E402

MIRROR = "https://gutenberg.pglaf.org"
CATALOG = f"{MIRROR}/cache/epub/feeds/pg_catalog.csv"
# Validation/test authors (Wells, Carroll, Wilde, Stoker) and the dev book's author.
EXCLUDED_AUTHORS = ("Wells, H. G.", "Carroll, Lewis", "Wilde, Oscar", "Stoker, Bram", "Phillips, David Graham")
EXCLUDED_IDS = {65432}  # dev book, White Magic


def fetch(url: str, delay: float) -> bytes:
    time.sleep(delay)
    request = urllib.request.Request(url, headers={"User-Agent": "dynapress-research (book list build)"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def candidates(catalog: bytes, seed: int) -> list[dict]:
    rows = csv.DictReader(io.StringIO(catalog.decode("utf-8")))
    out, seen_titles = [], set()
    for r in rows:
        subjects = r["Subjects"].lower()
        if r["Type"] != "Text" or r["Language"] != "en" or "fiction" not in subjects:
            continue
        if re.search(r"poetry|drama|plays", subjects) or int(r["Text#"]) in EXCLUDED_IDS:
            continue
        if any(a in r["Authors"] for a in EXCLUDED_AUTHORS):
            continue
        key = (re.sub(r"\W+", " ", r["Title"]).strip().lower(), r["Authors"])
        if key in seen_titles:  # another edition of the same book
            continue
        seen_titles.add(key)
        out.append({"id": int(r["Text#"]), "title": " ".join(r["Title"].split()), "authors": r["Authors"]})
    out.sort(key=lambda b: b["id"])
    random.Random(seed).shuffle(out)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", type=Path, default=ROOT / "data/lora_books.json")
    p.add_argument("--out", type=Path, default=ROOT / "data/fiction_books.json")
    p.add_argument("--cache", type=Path, default=ROOT / "data/fiction/books", help="the prepare --out directory's books/")
    p.add_argument("--train-books", type=int, default=300, help="training books in total, including the base ones")
    p.add_argument("--min-sentences", type=int, default=500)
    p.add_argument("--per-author", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--delay", type=float, default=1.0, help="seconds between downloads")
    args = p.parse_args()

    base = json.loads(args.base.read_text(encoding="utf-8"))["books"]
    base_ids = {int(b["id"].removeprefix("gutenberg-")) for b in base}
    # Same files from the mirror; the pinned checksums still apply.
    books = [{**b, "url": re.sub(r"^https://www\.gutenberg\.org", MIRROR, b["url"])} for b in base]
    args.cache.mkdir(parents=True, exist_ok=True)
    catalog = fetch(CATALOG, 0)
    per_author: dict[str, int] = {}
    need = args.train_books - sum(b["split"] == "train" for b in base)
    skipped = 0
    for c in candidates(catalog, args.seed):
        if need <= 0:
            break
        if c["id"] in base_ids or per_author.get(c["authors"], 0) >= args.per_author:
            continue
        url = f"{MIRROR}/cache/epub/{c['id']}/pg{c['id']}.txt"
        cached = args.cache / f"{digest(url.encode())}.txt"
        try:
            raw = cached.read_bytes() if cached.exists() else fetch(url, args.delay)
            n = len(set(book_sentences(raw.decode("utf-8-sig").replace("\r\n", "\n"))))
        except Exception as e:  # missing file, bad encoding, incomplete wrapper
            print(f"skip {c['id']}: {type(e).__name__}: {e}", flush=True)
            skipped += 1
            continue
        if not cached.exists():
            cached.write_bytes(raw)
        if n < args.min_sentences:
            skipped += 1
            continue
        per_author[c["authors"]] = per_author.get(c["authors"], 0) + 1
        books.append({"id": f"gutenberg-{c['id']}", "title": c["title"], "split": "train", "url": url, "sha256": digest(raw)})
        need -= 1
        print(f"{len(books) - len(base):4d} {c['id']:>6} {n:6d} {c['title'][:60]}", flush=True)
    if need > 0:
        raise SystemExit(f"only found {args.train_books - need} training books")
    info = {
        "note": "data/lora_books.json plus English fiction training books from the Project Gutenberg catalog; see scripts/build_fiction_books.py",
        "catalog_sha256": digest(catalog),
        "seed": args.seed,
        "books": books,
    }
    args.out.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {len(books)} books, skipped {skipped} candidates")


if __name__ == "__main__":
    main()
