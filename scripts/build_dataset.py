"""Build data/benchmark.jsonl from public datasets plus a hand-written ambiguity set.

Rows are fetched through the Hugging Face datasets-server ``/rows`` API (no full
download needed). Sentence selection is deterministic (fixed offsets, fixed
filters, seeded sampling). The resulting file is committed, so experiments do
not depend on the API staying the same.

Sources:
  conversational  roskoN/dailydialog (test)                      CC BY-NC-SA 4.0
  technical       philschmid/markdown-documentation-transformers HF Transformers docs, Apache-2.0
  scientific      allenai/scitldr, config Abstract (test)        Apache-2.0
  names_numbers   wikimedia/wikipedia 20231101.en                CC BY-SA 4.0
  ambiguous       data/ambiguous.txt (hand-written for this project)
"""

from __future__ import annotations

import json
import random
import re
import urllib.parse
import urllib.request
from pathlib import Path

PER_CATEGORY = 25
SEED = 1234
API = "https://datasets-server.huggingface.co/rows"
ROOT = Path(__file__).resolve().parent.parent


def fetch_rows(dataset: str, config: str, split: str, offset: int, length: int) -> list[dict]:
    rows = []
    while length > 0:
        n = min(100, length)
        query = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split, "offset": offset, "length": n})
        with urllib.request.urlopen(f"{API}?{query}", timeout=60) as resp:
            rows += [r["row"] for r in json.load(resp)["rows"]]
        offset, length = offset + n, length - n
    return rows


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", text) if s.strip()]


def ok(s: str, lo: int = 8, hi: int = 32) -> bool:
    n = len(re.findall(r"\w+", s))
    return lo <= n <= hi and s.isascii() and s[-1] in ".!?" and not re.search(r"https?:|[\[\]{}<>|*#=\\]|  ", s)


def detok_dailydialog(s: str) -> str:
    s = s.replace("’", "'")
    s = re.sub(r" ' (m|s|t|re|ve|ll|d)\b", r"'\1", s)
    s = re.sub(r" ([,.!?;:%])", r"\1", s)
    s = re.sub(r"\$ (\d)", r"$\1", s)
    return re.sub(r"\s+", " ", s).strip()


def conversational() -> list[str]:
    rows = fetch_rows("roskoN/dailydialog", "full", "test", 0, 300)
    return [u for r in rows for u in map(detok_dailydialog, r["utterances"]) if ok(u, 6, 30)]


def technical() -> list[str]:
    out = []
    for r in fetch_rows("philschmid/markdown-documentation-transformers", "default", "train", 0, 200):
        md = re.sub(r"```.*?```", "", r["markdown"], flags=re.S)
        md = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", md)  # links -> text
        md = md.replace("`", "")
        for line in md.splitlines():
            if line.strip() and not re.match(r"\s*([#|>\-*<!]|\d+\.)", line):
                out += [s for s in sentences(line) if ok(s)]
    return out


def scientific() -> list[str]:
    rows = fetch_rows("allenai/scitldr", "Abstract", "test", 0, 300)
    return [s.strip() for r in rows for s in r["source"] if ok(s.strip(), 8, 35)]


def names_numbers() -> list[str]:
    out = []
    for r in fetch_rows("wikimedia/wikipedia", "20231101.en", "train", 5000, 300):
        for para in r["text"].split("\n"):
            for s in sentences(para):
                caps = re.findall(r"(?<!^)(?<![.!?] )\b[A-Z][a-z]+", s)
                if ok(s, 8, 35) and re.search(r"\d", s) and len(caps) >= 2:
                    out.append(s)
    return out


def main() -> None:
    rng = random.Random(SEED)
    categories = {
        "conversational": conversational,
        "technical": technical,
        "scientific": scientific,
        "names_numbers": names_numbers,
    }
    rows = []
    for name, fn in categories.items():
        pool = sorted(set(fn()))
        picked = rng.sample(pool, min(PER_CATEGORY, len(pool)))
        print(f"{name}: {len(pool)} candidates, picked {len(picked)}")
        rows += [{"category": name, "text": t} for t in picked]
    ambiguous = [l.strip() for l in (ROOT / "data" / "ambiguous.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
    rows += [{"category": "ambiguous", "text": t} for t in ambiguous[:PER_CATEGORY]]
    out = ROOT / "data" / "benchmark.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
