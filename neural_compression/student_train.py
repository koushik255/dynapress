"""Train and evaluate the single-pass student decoder (``student.py``).

    python -m neural_compression.student_train train --out results/student/runs/cpu1 --teacher data/student/gen-*/teacher.jsonl.gz
    python -m neural_compression.student_train evaluate --model results/student/runs/cpu1/model --out results/student/cpu1

Training text is the ground-truth training-book sentences plus sentences sampled
from the teacher (SmolLM2-360M continuing training-book passages, see
``scripts/modal_student.py``); every sentence is its own ground truth. Each time
a sentence is drawn it gets a fresh random payload, fixed-ratio or adaptive-style,
built with the real ``compressor`` functions, so the student trains on exactly
the slots and length ranges the decoder will see. The loss is cross-entropy over
the same candidate set the decoder searches (plus the OOV option).

Evaluation uses the pilot's 50 validation sentences (the first 25 per validation
book), never the test split. Model selection uses the next 100 sentences per
validation book instead.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch

from .adaptive import compress_adaptive_many
from .benchmark import merge_payloads, zstd_size
from .compressor import WORD_RE, Compressed, compress, compress_with_lengths, has_digit, split_text, start_lengths
from .finetune import book_sentences
from .student import INT_KEYS, CharNgram, Config, StudentDecoder, StudentModel, Vocab, unit_features

MAX_WORD_LEN = 30


# ---------------------------------------------------------------- data


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def teacher_sentences(path: Path, exclude: set[str]) -> list[str]:
    """Complete sentences from teacher continuations, cut with the decoder's unit boundaries."""
    out = []
    for row in read_jsonl(path):
        for s in book_sentences(row["text"].replace("\r", "")):
            if s not in exclude:
                out.append(s)
    return list(dict.fromkeys(out))


def validation_samples(path: Path, start: int, count: int) -> list[str]:
    """Rows ``start``..``start+count`` of each validation book, in file order (the pilot used 0..25)."""
    seen: dict[str, int] = {}
    out = []
    for row in read_jsonl(path):
        n = seen.get(row["book_id"], 0)
        if start <= n < start + count:
            out.append(row["text"])
        seen[row["book_id"]] = n + 1
    return out


def build_vocab(sentences: list[str], min_count: int) -> Vocab:
    counts = Counter(
        w.lower() for s in sentences for w in WORD_RE.findall(s) if not has_digit(w) and len(w) <= MAX_WORD_LEN
    )
    words = [w for w, c in counts.items() if c >= min_count and len(w.lower()) == len(w)]
    return Vocab(words, [counts[w] for w in words])


def sample_payload(text: str, rng: random.Random) -> Compressed:
    """A random fixed-ratio or adaptive-style payload of ``text`` (numbers kept whole)."""
    if rng.random() < 0.35:
        return compress(text, rng.choice([25, 34, 50, 50, 50, 67, 75]) / 100)
    words = split_text(text)[1]
    lengths = []
    for w, k0 in zip(words, start_lengths(words, 25)):
        r = rng.random()
        lengths.append(k0 if r < 0.45 else len(w) if r < 0.6 else rng.randint(k0, len(w)))
    return compress_with_lengths(text, lengths, 25)


def example(text: str, rng: random.Random, vocab: Vocab) -> tuple[dict, list[int]]:
    doc = sample_payload(text, rng)
    slots = doc.slots()
    f = unit_features(doc, slots, range(len(slots)), vocab)
    words = split_text(text)[1]
    gold = []
    for slot, w in zip(slots, words):
        if slot.complete:
            gold.append(-100)
        else:
            gold.append(vocab.index.get(w.lower(), len(vocab)))  # in vocab => among the candidates
    return f, gold


def collate(batch: list[tuple[dict, list[int]]]) -> dict:
    B = len(batch)
    L = max(len(g) for _, g in batch)
    t = {k: torch.zeros(B, L, dtype=torch.long) for k in INT_KEYS}
    pad = torch.ones(B, L, dtype=torch.bool)
    spans = torch.zeros(B, L, 4, dtype=torch.long)
    gold = torch.full((B, L), -100, dtype=torch.long)
    char_ids, offsets = [], []
    for b, (f, g) in enumerate(batch):
        n = len(g)
        for k in INT_KEYS:
            t[k][b, :n] = torch.tensor(f[k])
        pad[b, :n] = False
        spans[b, :n] = torch.tensor(f["spans"])
        gold[b, :n] = torch.tensor(g)
        ends = f["char_offsets"][1:] + [len(f["char_ids"])]
        for j in range(L):
            offsets.append(len(char_ids))
            if j < n:
                char_ids += f["char_ids"][f["char_offsets"][j] : ends[j]]
    t["pad"] = pad
    t["char_ids"] = torch.tensor(char_ids)
    t["char_offsets"] = torch.tensor(offsets)
    t["spans"] = spans.view(B * L, 4)
    t["gold"] = gold.view(B * L)
    return t


class Batches(torch.utils.data.IterableDataset):
    def __init__(self, sentences: list[str], weights: list[float], vocab: Vocab, batch_size: int, seed: int):
        self.sentences, self.weights, self.vocab, self.batch_size, self.seed = sentences, weights, vocab, batch_size, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed * 1000 + (info.id if info else 0))
        while True:
            # Draw 8 batches at once and group them by length, so little of each batch is padding.
            texts = sorted(rng.choices(self.sentences, weights=self.weights, k=8 * self.batch_size), key=len)
            chunks = [texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
            rng.shuffle(chunks)
            for chunk in chunks:
                yield collate([example(s, rng, self.vocab) for s in chunk])


def train_loss(model: StudentModel, t: dict, vlen: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Cross-entropy over each truncated word's candidate set plus OOV; the same scores the decoder uses."""
    B, L = t["case"].shape
    V, d = model.word.weight.shape
    E = model.word.weight
    a, b, lo, hi = t["spans"].unbind(1)
    sel = t["gold"] != -100  # truncated words
    known = ~sel & (b > a)  # complete words in the vocabulary: span (idx, idx + 1)
    vidx = torch.arange(V, device=vlen.device)
    a, b, lo, hi = a[sel, None], b[sel, None], lo[sel, None], hi[sel, None]
    mask = (vidx >= a) & (vidx < b) & (vlen >= lo) & (vlen <= hi)
    w = torch.softmax(model.prior.masked_fill(~mask, -math.inf), 1).nan_to_num(0.0)
    bag = torch.zeros(B * L, d, device=E.device)
    bag = bag.index_put((sel.nonzero()[:, 0],), (w @ E).float()).index_put((known.nonzero()[:, 0],), E[t["spans"][known, 0]].float())
    h = model.encode(t, bag.view(B, L, d)).reshape(B * L, d)
    hs = h[sel]
    logits = (model.out(hs) @ E.T + model.bias).float().masked_fill(~mask, -math.inf)
    logits = torch.cat([logits, model.oov(hs).float()], 1)
    gold = t["gold"][sel]
    loss = torch.nn.functional.cross_entropy(logits, gold)
    correct = int((logits.argmax(1) == gold).sum())
    return loss, correct, int(sel.sum())


# ---------------------------------------------------------------- evaluation


def fixed_accuracy(decoder: StudentDecoder, texts: list[str], retention: float = 0.5) -> dict:
    payloads = [compress(t, retention).to_bytes() for t in texts]
    t0 = time.perf_counter()
    results = decoder.decode_many(payloads)
    seconds = time.perf_counter() - t0
    words = [split_text(t)[1] for t in texts]
    total = sum(len(w) for w in words)
    correct = sum(sum(a == b for a, b in zip(r.words, w)) for r, w in zip(results, words))
    return {"word_accuracy": correct / total, "words": total, "ms_per_word": 1000 * seconds / total, "sentence_exact": sum(r.text == t for r, t in zip(results, texts))}


def adaptive_eval(decoder: StudentDecoder, texts: list[str]) -> tuple[dict, list[dict]]:
    t0 = time.perf_counter()
    results = compress_adaptive_many(texts, decoder, 0.25, 1.0)
    compress_seconds = time.perf_counter() - t0
    # Independent check: a fresh decode of every final payload, then of the merged corpus payload.
    t0 = time.perf_counter()
    verified = decoder.decode_many([r.payload for r in results])
    decode_seconds = time.perf_counter() - t0
    merged = merge_payloads([Compressed.from_bytes(r.payload) for r in results])
    corpus = "\n".join(texts)
    merged_ok = decoder.decode(merged.to_bytes()).text == corpus
    words = sum(len(split_text(t)[1]) for t in texts)
    rows = [
        {
            "original": t,
            "decoded": v.text,
            "exact": v.text == t,
            "matches_search": v.text == r.decoded_text,
            "rounds": r.rounds,
            "payload_bytes": len(r.payload),
            "payload_hex": r.payload.hex(),
        }
        for t, r, v in zip(texts, results, verified)
    ]
    summary = {
        "sentences": len(texts),
        "words": words,
        "all_exact": all(r["exact"] and r["matches_search"] for r in rows),
        "merged_corpus_exact": merged_ok,
        "payload_bytes": len(merged.to_bytes()),
        "payload_zstd_bytes": len(merged.to_bytes(use_zstd=True)),
        "original_zstd_bytes": zstd_size(corpus.encode()),
        "prefix_chars": sum(len(p) for p in merged.prefixes),
        "complete_words": sum(merged.complete),
        "mean_rounds": sum(r.rounds for r in results) / len(results),
        "compress_seconds": compress_seconds,
        "decode_seconds": decode_seconds,
        "decode_ms_per_word": 1000 * decode_seconds / words,
    }
    return summary, rows


def evaluate(args) -> None:
    torch.set_num_threads(args.threads)
    decoder = StudentDecoder.load(args.model, args.device)
    texts = validation_samples(args.data / "validation.jsonl", args.start, args.per_book)
    decoder.decode_many([compress(texts[0], 0.5).to_bytes()])  # warm-up
    fixed = fixed_accuracy(decoder, texts)
    summary, rows = adaptive_eval(decoder, texts)
    params = sum(p.numel() for p in decoder.model.parameters())
    files = [p for p in Path(args.model).iterdir() if p.is_file()]
    out = {
        "model": str(args.model),
        "device": args.device,
        "threads": args.threads,
        "torch": torch.__version__,
        "samples": f"validation rows {args.start}..{args.start + args.per_book} per book",
        "parameters": params,
        "vocab_size": len(decoder.vocab),
        "model_dir_bytes": sum(p.stat().st_size for p in files),
        "fixed50": fixed,
        "adaptive100": summary,
        "ngram_fallbacks": decoder.fallbacks,
    }
    if args.device == "cuda":
        out["gpu"] = torch.cuda.get_device_name()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    (args.out / "adaptive_rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(json.dumps(out, indent=2))


# ---------------------------------------------------------------- training


def train(args, deadline: float | None = None, log=lambda s: print(s, flush=True)) -> dict:
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    books = [r["text"] for r in read_jsonl(args.data / "train.jsonl")]
    val_rows = read_jsonl(args.data / "validation.jsonl")
    exclude = {r["text"] for r in val_rows}
    teacher = []
    for path in args.teacher or []:
        teacher += teacher_sentences(Path(path), exclude)
    log(f"book sentences {len(books)}, teacher sentences {len(teacher)}")
    vocab = build_vocab(books + teacher, args.min_count)
    log(f"vocab {len(vocab)}")
    sentences = books + teacher
    # Draw book sentences ``book_weight`` times as often as a teacher sentence.
    weights = [args.book_weight] * len(books) + [1.0] * len(teacher)
    cfg = Config(len(vocab), args.dim, args.layers, args.heads, args.dim * 4, args.dropout)
    model = StudentModel(cfg, vocab.logprior).to(device)
    log(f"parameters {sum(p.numel() for p in model.parameters()):,}")
    vlen = torch.tensor(vocab.lens, device=device)
    loader = torch.utils.data.DataLoader(
        Batches(sentences, weights, vocab, args.batch_size, args.seed), batch_size=None, num_workers=args.workers, persistent_workers=args.workers > 0
    )
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    dev_texts = validation_samples(args.data / "validation.jsonl", 25, args.dev_per_book)
    ngram = CharNgram(vocab.words, vocab.counts.tolist())
    history = []
    t0 = time.time()
    run_loss = run_correct = run_total = 0.0
    use_amp = device.type == "cuda"
    best = None
    for step, batch in enumerate(loader, 1):
        model.train()
        # Warmup, then cosine decay over whichever runs out first: steps or the time budget.
        progress = step / args.steps if deadline is None else max(step / args.steps, (time.time() - t0) / (deadline - t0))
        lr = args.lr * min(1.0, step / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
        for group in opt.param_groups:
            group["lr"] = lr
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            loss, correct, total = train_loss(model, batch, vlen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        run_loss += float(loss.detach()) * total
        run_correct += correct
        run_total += total
        last = step >= args.steps or progress >= 1.0
        if step % args.log_every == 0 or last:
            entry = {"step": step, "loss": run_loss / run_total, "train_acc": run_correct / run_total, "seconds": time.time() - t0, "lr": lr}
            run_loss = run_correct = run_total = 0.0
            if step % args.eval_every == 0 or last:
                decoder = StudentDecoder(model, vocab, ngram, args.device)
                entry["dev_fixed50"] = fixed_accuracy(decoder, dev_texts)["word_accuracy"]
                if best is None or entry["dev_fixed50"] > best[0]:
                    best = (entry["dev_fixed50"], step)
                    decoder.save(args.out / "model", {"step": step, "dev_fixed50": entry["dev_fixed50"]})
            history.append(entry)
            log(json.dumps(entry))
        if last:
            break
    info = {"args": {k: str(v) for k, v in vars(args).items()}, "books": len(books), "teacher": len(teacher), "vocab": len(vocab), "best": best, "history": history}
    (args.out / "train.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("train")
    p.add_argument("--data", type=Path, default=Path("data/student/books_all"))
    p.add_argument("--teacher", nargs="*", default=[])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-count", type=int, default=2)
    p.add_argument("--book-weight", type=float, default=3.0)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--dev-per-book", type=int, default=100)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p = commands.add_parser("evaluate")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("data/lora"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--per-book", type=int, default=25)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.command == "train":
        if args.out.exists() and any(args.out.iterdir()):
            parser.error("--out must be empty")
        args.out.mkdir(parents=True, exist_ok=True)
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
