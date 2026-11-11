"""Book-disjoint prefix-restoration LoRA experiment: prepare, train, evaluate.

Run with ``python -m neural_compression.finetune --help``. Training dependencies
are optional (``uv sync --group training``); inference uses the existing decoder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import urllib.request
from pathlib import Path

from .compressor import (
    MAX_SUFFIX,
    compress,
    compress_with_lengths,
    has_digit,
    kept_chars,
    split_text,
)
from .decoder import Decoder, _span, sentence_units
from .models import DEFAULT_MODEL, DEFAULT_REVISION

SPLITS = ("train", "validation", "test")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def book_sentences(text: str) -> list[str]:
    """Strip Gutenberg wrappers and split using the real decoder's unit boundaries."""
    start = re.search(
        r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n",
        text,
        re.IGNORECASE,
    )
    end = re.search(
        r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", text, re.IGNORECASE
    )
    if start or end:
        if not start or not end or start.end() >= end.start():
            raise ValueError("incomplete Gutenberg wrapper")
        text = text[start.end() : end.start()]
    out = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        c = compress(paragraph, 1)
        for unit in sentence_units(c):
            sentence = "".join(
                c.prefixes[i] + c.separators[i + 1] for i in unit
            ).strip()
            if unit.start == 0:
                sentence = c.separators[0] + sentence
            # Exclude headings, fragments and very short dialogue for this pilot.
            if 5 <= len(unit) <= 48 and re.search(r"[.!?][\"'’”)]*$", sentence):
                out.append(sentence)
    return out


def prepare(manifest: Path, out: Path, seed: int = 0, per_book: int = 2000) -> dict:
    if per_book < 1:
        raise ValueError("per_book must be positive")
    books = json.loads(manifest.read_text(encoding="utf-8"))["books"]
    if {b["split"] for b in books} != set(SPLITS):
        raise ValueError("manifest must contain train, validation and test books")
    ids = [b["id"] for b in books]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate book id: each book belongs to exactly one split")
    raw_dir = out / "books"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows = {s: [] for s in SPLITS}
    hashes, metadata = set(), []
    for book in books:
        if "path" in book:
            raw = (manifest.parent / book["path"]).read_bytes()
        else:
            cached = raw_dir / f"{digest(book['url'].encode())}.txt"
            if not cached.exists():
                with urllib.request.urlopen(book["url"], timeout=60) as response:
                    cached.write_bytes(response.read())
            raw = cached.read_bytes()
        if book.get("sha256") and digest(raw) != book["sha256"]:
            raise ValueError(f"source checksum mismatch for {book['id']}")
        text = raw.decode("utf-8-sig").replace("\r\n", "\n")
        normalized_hash = digest(re.sub(r"\s+", " ", text).strip().encode())
        if normalized_hash in hashes:
            raise ValueError("duplicate book contents, even under different ids")
        hashes.add(normalized_hash)
        sentences = list(dict.fromkeys(book_sentences(text)))
        rng.shuffle(sentences)
        rows[book["split"]] += [
            {"book_id": book["id"], "text": s} for s in sentences[:per_book]
        ]
        metadata.append(
            {**book, "sha256": digest(raw), "available_sentences": len(sentences)}
        )
    # Remove shared sentences from EVERY split, not just from training.
    owners: dict[str, set[str]] = {}
    for split, samples in rows.items():
        for row in samples:
            key = re.sub(r"\s+", " ", row["text"]).strip().casefold()
            owners.setdefault(key, set()).add(split)
    for split in SPLITS:
        rows[split] = [
            r
            for r in rows[split]
            if len(owners[re.sub(r"\s+", " ", r["text"]).strip().casefold()]) == 1
        ]
        if not rows[split]:
            raise ValueError(f"no eligible sentences in {split}")
        if {r["book_id"] for r in rows[split]} != {
            b["id"] for b in books if b["split"] == split
        }:
            raise ValueError(f"a book has no eligible sentences in {split}")
    files = {}
    for split in SPLITS:
        data = "".join(
            json.dumps(r, ensure_ascii=False) + "\n" for r in rows[split]
        ).encode()
        (out / f"{split}.jsonl").write_bytes(data)
        files[split] = {"sentences": len(rows[split]), "sha256": digest(data)}
    info = {"seed": seed, "per_book": per_book, "books": metadata, "splits": files}
    (out / "manifest.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    return info


def training_example(text: str, tokenizer, rng: random.Random, mixed: bool) -> dict:
    pct = rng.choice([25, 50, 75])
    c = compress(text, pct / 100)
    if mixed:
        words = split_text(text)[1]
        lengths = [
            len(w)
            if has_digit(w)
            else rng.randint(
                max(1, kept_chars(len(w), 25), len(w) - MAX_SUFFIX), len(w)
            )
            for w in words
        ]
        c = compress_with_lengths(text, lengths, pct)
    decoder = Decoder(None, "full")
    request = decoder._request(c, c.slots(), range(len(c.prefixes)))
    # Match inference's separately tokenized shared prefix and request exactly.
    prompt = tokenizer(decoder._fewshot_prefix(pct), add_special_tokens=False).input_ids
    prompt += tokenizer(request.prompt, add_special_tokens=False).input_ids
    # Leading separators are copied outside the model, just as in Decoder.decode.
    answer_text = _span(c.separators, split_text(text)[1], range(len(c.prefixes)))
    answer = tokenizer(" " + answer_text + "\n", add_special_tokens=False).input_ids
    return {
        "input_ids": prompt + answer,
        "attention_mask": [1] * (len(prompt) + len(answer)),
        "labels": [-100] * len(prompt) + answer,
    }


def encode_rows(rows: list[dict], tokenizer, seed: int, max_length: int) -> list[dict]:
    rng = random.Random(seed)
    examples = []
    for row in rows:
        for mixed in (False, True):
            example = training_example(row["text"], tokenizer, rng, mixed)
            # Never silently truncate an answer or its prompt.
            if len(example["input_ids"]) <= max_length:
                examples.append(example)
    if not examples:
        raise ValueError("all examples exceed max_length")
    return examples


def collate(examples: list[dict], pad_id: int) -> dict:
    import torch

    width = max(len(e["input_ids"]) for e in examples)
    return {
        key: torch.tensor([e[key] + [fill] * (width - len(e[key])) for e in examples])
        for key, fill in (
            ("input_ids", pad_id),
            ("attention_mask", 0),
            ("labels", -100),
        )
    }


def train(args, callbacks: list | None = None) -> None:
    import functools

    import peft
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("training output must be empty; use a new run directory")
    info = json.loads((args.data / "manifest.json").read_text())
    # Read and hash only train/validation. Test text is never loaded by training.
    splits = {}
    for split in ("train", "validation"):
        path = args.data / f"{split}.jsonl"
        if digest(path.read_bytes()) != info["splits"][split]["sha256"]:
            raise ValueError(f"{split} checksum mismatch; prepare data again")
        splits[split] = read_rows(path)
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = {
        s: encode_rows(rows, tokenizer, args.seed, args.max_length)
        for s, rows in splits.items()
    }
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, dtype=torch.float32
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        ),
    )
    model.print_trainable_parameters()
    training_args = TrainingArguments(
        output_dir=str(args.out / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=2e-4,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accumulation,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        use_cpu=args.cpu,
        logging_steps=10,
        dataloader_pin_memory=not args.cpu,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        data_collator=functools.partial(collate, pad_id=tokenizer.pad_token_id),
        callbacks=callbacks,
    )
    result = trainer.train()
    metrics = trainer.evaluate()
    model.save_pretrained(args.out / "adapter")
    tokenizer.save_pretrained(args.out / "adapter")
    # Merged weights use the unmodified load_model/benchmark/compress/decompress APIs.
    merged = model.merge_and_unload().eval()
    merged.config.use_cache = True
    merged.save_pretrained(args.out / "merged")
    tokenizer.save_pretrained(args.out / "merged")
    meta = {
        "base_model": args.model,
        "revision": args.revision,
        "seed": args.seed,
        "data": info,
        "training_args": training_args.to_dict(),
        "train_metrics": result.metrics,
        "validation_metrics": metrics,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "log_history": trainer.state.log_history,
        "encoded_examples": {s: len(v) for s, v in encoded.items()},
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "selection": "lowest validation answer-token loss; confirm with validation adaptive-100% bytes before opening test",
    }
    (args.out / "experiment.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )


def evaluate(args) -> None:
    from .benchmark import (
        Sample,
        adaptive_corpus_sizes,
        export,
        run_adaptive,
        run_fixed,
    )
    from .models import load_model

    if args.split == "test" and not args.final_test:
        raise ValueError(
            "test is held out: pass --final-test only after freezing the experiment on validation"
        )
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("evaluation output must be empty; use a new directory")
    meta = json.loads((args.run / "experiment.json").read_text())
    path = args.data / f"{args.split}.jsonl"
    if digest(path.read_bytes()) != meta["data"]["splits"][args.split]["sha256"]:
        raise ValueError("evaluation data differs from the recorded experiment")
    rows = read_rows(path)
    # Equal deterministic sample counts from each book, not just the first book.
    counts, samples = {}, []
    for row in rows:
        book = row["book_id"]
        if counts.get(book, 0) < args.per_book:
            samples.append(Sample(book, row["text"]))
            counts[book] = counts.get(book, 0) + 1
    summary = {}
    for label, name, revision in (
        ("base", meta["base_model"], meta["revision"]),
        ("lora", str(args.run / "merged"), None),
    ):
        lm = load_model(
            name,
            revision=revision,
            device=args.device,
            threads=args.threads,
            seed=meta["seed"],
        )
        measured = run_fixed(
            lm, samples, 0.5, "full", beam_size=args.beam, batch_size=args.batch_size
        )
        measured += run_adaptive(
            lm,
            samples,
            "full",
            target_accuracy=1.0,
            beam_size=args.beam,
            batch_size=args.batch_size,
        )
        adaptive = [r for r in measured if r["mode"] == "adaptive@1"]
        if not all(
            r["sentence_exact"] and r["verified_matches_search"] for r in adaptive
        ):
            raise RuntimeError(f"{label}: independent adaptive decode was not exact")
        export(
            measured,
            args.out / label,
            {
                **lm.describe(),
                "split": args.split,
                "books": counts,
                "beam": args.beam,
                "batch_size": args.batch_size,
                "device": str(lm.device),
                "experiment": meta,
            },
        )
        summary[label] = {
            **adaptive_corpus_sizes(measured),
            "fixed_word_accuracy": sum(
                r["word_correct"] for r in measured if r["mode"] == "fixed"
            )
            / sum(r["word_total"] for r in measured if r["mode"] == "fixed"),
            "adaptive_mean_rounds": sum(r["rounds"] for r in adaptive) / len(adaptive),
            "adaptive_compress_seconds": sum(r["compress_seconds"] for r in adaptive),
            "adaptive_decode_seconds": sum(r["decode_seconds"] for r in adaptive),
        }
        del lm
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary["adapter_bytes"] = sum(
        p.stat().st_size for p in (args.run / "adapter").glob("*") if p.is_file()
    )
    (args.out / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser(
        "prepare",
        help="download/cache books and create deterministic book-disjoint splits",
    )
    p.add_argument("--manifest", type=Path, default=Path("data/lora_books.json"))
    p.add_argument("--out", type=Path, default=Path("data/lora"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--per-book", type=int, default=2000)
    p = commands.add_parser(
        "train", help="train LoRA, save adapter and merged decoder-compatible model"
    )
    p.add_argument("--data", type=Path, default=Path("data/lora"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--revision", default=DEFAULT_REVISION)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accumulation", type=int, default=8)
    p.add_argument("--cpu", action="store_true")
    p = commands.add_parser(
        "evaluate",
        help="paired base/LoRA fixed-50 and independently verified adaptive-100 benchmark",
    )
    p.add_argument("--data", type=Path, default=Path("data/lora"))
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--final-test", action="store_true")
    p.add_argument("--per-book", type=int, default=25)
    p.add_argument("--beam", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()
    if args.command == "prepare":
        print(
            json.dumps(
                prepare(args.manifest, args.out, args.seed, args.per_book), indent=2
            )
        )
    elif args.command == "train":
        train(args)
    else:
        if args.per_book < 1:
            parser.error("--per-book must be positive")
        evaluate(args)


if __name__ == "__main__":
    main()
