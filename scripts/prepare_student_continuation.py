"""Prepare document-disjoint continuation data without opening the book test split.

Run from the repository: uv run --frozen python scripts/prepare_student_continuation.py
The original books, validation rows and teacher gzip must already be recovered.
Raw text stays under ignored data/student; only the manifest is published.
DailyDialog is CC BY-NC-SA 4.0: this continuation checkpoint is research-only,
not a commercially cleared model. Transformers documentation is Apache-2.0.
"""

from __future__ import annotations

import json
import random
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_dataset import detok_dailydialog, ok, sentences
from neural_compression.finetune import book_sentences, digest
from neural_compression.student_train import read_jsonl, teacher_sentences

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/student/continuation"
SEED = 731


def key(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def main() -> None:
    raw_dir = OUT / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    provenance = []
    pools = {"train": [], "validation": []}
    recorded = ROOT / "results/student/continuation/data_manifest.json"
    expected = {s["url"]: s["sha256"] for s in json.loads(recorded.read_text())["sources"]} if recorded.exists() else {}

    def cached(url: str) -> bytes:
        path = raw_dir / digest(url.encode())
        if not path.exists():
            with urllib.request.urlopen(url, timeout=90) as response:
                path.write_bytes(response.read())
        data = path.read_bytes()
        if url in expected and digest(data) != expected[url]:
            raise ValueError(f"source checksum changed: {url}")
        provenance.append({"url": url, "sha256": digest(data)})
        return data

    def rows(dataset: str, config: str, split: str, start: int, count: int) -> list[dict]:
        out = []
        for offset in range(start, start + count, 100):
            query = urllib.parse.urlencode(dict(dataset=dataset, config=config, split=split, offset=offset, length=min(100, start + count - offset)))
            batch = json.loads(cached("https://datasets-server.huggingface.co/rows?" + query))
            out += [r["row"] for r in batch["rows"]]
        return out

    def add(split: str, domain: str, document: str, texts: list[str]) -> None:
        pools[split] += [{"book_id": domain, "document": document, "text": t} for t in texts]

    for split, source_split, count in [("train", "train", 600), ("validation", "validation", 120)]:
        for row in rows("roskoN/dailydialog", "full", source_split, 0, count):
            texts = [s for t in row["utterances"] for s in book_sentences(detok_dailydialog(t)) if ok(s, 5, 48)]
            add(split, "dialogue", "dailydialog:" + row["id"], texts)

    for split, start, count in [("train", 200, 80), ("validation", 280, 40)]:
        for row in rows("philschmid/markdown-documentation-transformers", "default", "train", start, count):
            md = re.sub(r"```.*?```", "", row["markdown"], flags=re.S)
            md = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", md).replace("`", "")
            texts = []
            for line in md.splitlines():
                if line.strip() and not re.match(r"\s*([#|>\-*<!]|\d+\.)", line):
                    texts += [s for s in sentences(line) if ok(s, 5, 48)]
            # Version changes of a documentation page are still the same document.
            document = re.sub(r"/v[\d.]+/", "/", row["url"])
            add(split, "technical", document, texts)

    for split, number in [("train", 1342), ("validation", 161)]:
        url = f"https://www.gutenberg.org/cache/epub/{number}/pg{number}.txt"
        text = cached(url).decode("utf-8-sig").replace("\r\n", "\n")
        add(split, "fiction", f"gutenberg-{number}", book_sentences(text))

    original = read_jsonl(ROOT / "data/student/books_all/train.jsonl")
    prior_validation = read_jsonl(ROOT / "data/student/books_all/validation.jsonl")
    teacher = teacher_sentences(OUT / "teacher.jsonl.gz", {r["text"] for r in prior_validation})
    prior_texts = {key(r["text"]) for r in original + prior_validation + read_jsonl(ROOT / "data/benchmark.jsonl")}
    prior_texts.update(map(key, teacher))
    owners = {}
    documents = {}
    for split, samples in pools.items():
        for row in samples:
            owners.setdefault(key(row["text"]), set()).add(split)
            documents.setdefault(row["document"], set()).add(split)
    picked = {}
    for split in pools:
        selected = []
        for domain in ("dialogue", "technical", "fiction"):
            eligible = {}
            for r in pools[split]:
                k = key(r["text"])
                if r["book_id"] == domain and k not in prior_texts and len(owners[k]) == 1 and len(documents[r["document"]]) == 1:
                    eligible.setdefault(k, r)
            pool = sorted(eligible.values(), key=lambda r: r["text"])
            n = min(2000 if split == "train" else 40, len(pool))
            if n == 0:
                raise ValueError(f"no eligible {domain} {split} rows")
            chosen = rng.sample(pool, n)
            if split == "train":
                chosen = [{**r, "weight": 20.0} for r in chosen]
            selected += chosen
            print(f"{split} {domain}: {len(pool)} eligible, {n} selected", flush=True)
        picked[split] = selected
    picked["train"] = original + picked["train"]
    files = {}
    for split, samples in picked.items():
        data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in samples).encode()
        (OUT / f"{split}.jsonl").write_bytes(data)
        files[split] = {"rows": len(samples), "sha256": digest(data), "documents": sorted({r.get("document", r["book_id"]) for r in samples})}
    manifest = {
        "seed": SEED,
        "sources": provenance,
        "licenses": {
            "dialogue": {"license": "CC BY-NC-SA 4.0", "url": "https://huggingface.co/datasets/roskoN/dailydialog", "authors": "Li et al., DailyDialog (IJCNLP 2017)", "restriction": "Noncommercial research; no commercial clearance asserted for resulting checkpoint"},
            "technical": {"license": "Apache-2.0", "url": "https://huggingface.co/datasets/philschmid/markdown-documentation-transformers", "authors": "Hugging Face Transformers documentation; dataset by Philipp Schmid"},
            "fiction": {"license": "Project Gutenberg public domain in USA; check local rights", "authors": "Jane Austen, Pride and Prejudice (train); Sense and Sensibility (development)"},
        },
        "files": files,
        "original_train_sha256": digest((ROOT / "data/student/books_all/train.jsonl").read_bytes()),
        "teacher_sha256": digest((OUT / "teacher.jsonl.gz").read_bytes()),
        "exclusions": "Normalized exact sentence matches to prior train/validation, teacher text and benchmark removed from new data; shared documents/sentences removed across new splits. No original test file accessed. Near duplicates and base-pretraining overlap are not ruled out.",
        "weights": {"original_book_sentence": 3.0, "new_real_sentence": 20.0, "teacher_sentence": 1.0},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
