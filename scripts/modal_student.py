"""Modal jobs for the single-pass student decoder (``neural_compression.student``).

    uv run --with modal modal run scripts/modal_student.py::generate   # teacher text, time-capped
    uv run --with modal modal run scripts/modal_student.py::train      # GPU training + validation

``generate`` samples continuations of training-book passages from the teacher
(SmolLM2-360M, pinned revision) for a fixed wall-clock budget and stores them on
the ``neural-compression-hf-cache`` volume. Only train-split sentences are
uploaded as prompts. Both jobs use one GPU, no retries, a hard function timeout,
and an in-loop deadline well inside it, so the worst-case cost is bounded by
``timeout x hourly rate`` (printed before launch).
"""

from __future__ import annotations

import gzip
import json
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
# A10G GPU + 4 cores + 16 GiB, at Modal's list prices (USD/hour).
HOURLY_USD = 1.10 + 4 * 0.0473 + 16 * 0.008
GEN_TIMEOUT = 50 * 60
GEN_SECONDS = 36 * 60  # stop sampling here; the rest is startup, model load, and writing

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.15.0",
        "transformers==5.22.2",
        "peft==0.19.1",
        "accelerate==1.15.0",
        "zstandard>=0.23",
        "numpy",
    )
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4"})
    .add_local_dir(ROOT / "neural_compression", "/root/neural_compression")
    .add_local_file(ROOT / "data/lora/train.jsonl", "/data/lora/train.jsonl")
)
cache = modal.Volume.from_name("neural-compression-hf-cache")
app = modal.App("neural-compression-student", image=image)


def make_prompts(rows: list[dict], n: int, seed: int) -> list[tuple[str, str]]:
    """``n`` prompts of one or two random sentences from the same training book."""
    rng = random.Random(seed)
    by_book: dict[str, list[str]] = {}
    for r in rows:
        by_book.setdefault(r["book_id"], []).append(r["text"])
    books = sorted(by_book)
    out = []
    for _ in range(n):
        book = rng.choice(books)
        out.append((book, " ".join(rng.sample(by_book[book], rng.choice([1, 2])))))
    return out


def sample_batch(model, tokenizer, prompts: list[str], max_new_tokens: int, seed: int) -> list[str]:
    """Teacher continuations (temperature 1, top-p 0.95) of ``prompts``, prompt text removed."""
    import torch

    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(
            **enc,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.batch_decode(out[:, enc.input_ids.shape[1] :], skip_special_tokens=True)


def load_teacher(device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from neural_compression.models import DEFAULT_MODEL, DEFAULT_REVISION

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, revision=DEFAULT_REVISION, dtype=dtype).to(device).eval()
    return model, tokenizer


@app.function(gpu="A10G", cpu=(4, 4), memory=(16384, 16384), max_containers=1, retries=0, timeout=GEN_TIMEOUT, volumes={"/cache": cache})
def generate_remote(run_id: str, batch_size: int = 384, max_new_tokens: int = 160) -> dict:
    sys.path.insert(0, "/root")
    import torch

    started = time.monotonic()
    root = Path("/cache/student") / run_id
    root.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(line) for line in open("/data/lora/train.jsonl", encoding="utf-8")]
    model, tokenizer = load_teacher("cuda")
    out_path = root / "teacher.jsonl.gz"
    tokens = batches = 0
    gen_start = time.monotonic()
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        while time.monotonic() - started < GEN_SECONDS:
            prompts = make_prompts(rows, batch_size, seed=batches)
            texts = sample_batch(model, tokenizer, [p for _, p in prompts], max_new_tokens, seed=batches)
            for (book, _), text in zip(prompts, texts):
                f.write(json.dumps({"book_id": book, "text": text}, ensure_ascii=False) + "\n")
            batches += 1
            tokens += batch_size * max_new_tokens
            if batches % 10 == 0:
                rate = tokens / (time.monotonic() - gen_start)
                print(f"batch {batches}: {tokens} tokens, {rate:.0f} tok/s", flush=True)
    elapsed = time.monotonic() - started
    status = {
        "run_id": run_id,
        "batches": batches,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "generated_token_slots": tokens,
        "gpu": torch.cuda.get_device_name(),
        "seconds": elapsed,
        "estimated_compute_usd": elapsed * HOURLY_USD / 3600,
        "remote": str(out_path),
    }
    (root / "status.json").write_text(json.dumps(status, indent=2))
    cache.commit()
    return {"status": status, "data": out_path.read_bytes()}


TRAIN_TIMEOUT = 95 * 60
TRAIN_HOURLY_USD = 1.10 + 8 * 0.0473 + 16 * 0.008  # A10G + 8 cores (data workers) + 16 GiB
TRAIN_SECONDS = 62 * 60  # training stops here (best dev checkpoint kept); then GPU evaluation

train_image = image.add_local_dir(ROOT / "data/student/books_all", "/data/books_all", ignore=["books", "books/**", "test.jsonl"]).add_local_file(
    ROOT / "data/lora/validation.jsonl", "/data/lora/validation.jsonl"
)


@app.function(
    image=train_image, gpu="A10G", cpu=(8, 8), memory=(16384, 16384), max_containers=1, retries=0, timeout=TRAIN_TIMEOUT, volumes={"/cache": cache}
)
def train_remote(run_id: str, teacher: str, options: dict) -> dict:
    sys.path.insert(0, "/root")
    from types import SimpleNamespace

    import torch

    from neural_compression import student_train

    started = time.time()
    root = Path("/cache/student") / run_id
    root.mkdir(parents=True, exist_ok=False)
    options = dict(options)
    train_seconds = min(options.pop("train_seconds", TRAIN_SECONDS), TRAIN_SECONDS)
    args = SimpleNamespace(
        data=Path("/data/books_all"),
        teacher=[teacher] if teacher else [],  # "": books only (ablation)
        out=root,
        min_count=2,
        book_weight=3.0,
        dim=512,
        layers=8,
        heads=8,
        dropout=0.1,
        batch_size=256,
        lr=6e-4,
        warmup=2000,
        steps=200000,
        log_every=500,
        eval_every=4000,
        dev_per_book=100,
        workers=6,
        threads=1,
        device="cuda",
        seed=0,
    )
    vars(args).update(options)
    error = None
    lines: list[str] = []

    def log(s: str) -> None:
        print(s, flush=True)
        lines.append(s)

    try:
        student_train.train(args, deadline=started + train_seconds, log=log)
        cache.commit()
        for device in ("cuda", "cpu"):
            student_train.evaluate(
                SimpleNamespace(model=root / "model", data=Path("/data/lora"), out=root / f"eval-{device}", start=0, per_book=25, device=device, threads=4)
            )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        log(error)
    elapsed = time.time() - started
    status = {
        "run_id": run_id,
        "status": "failed" if error else "complete",
        "error": error,
        "gpu": torch.cuda.get_device_name(),
        "seconds": elapsed,
        "estimated_compute_usd": elapsed * TRAIN_HOURLY_USD / 3600,
        "options": options,
    }
    (root / "status.json").write_text(json.dumps(status, indent=2))
    (root / "log.txt").write_text("\n".join(lines))
    cache.commit()
    files = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    return {"status": status, "files": files}


@app.local_entrypoint()
def train(teacher: str = "", options: str = "{}"):
    run_id = "train-" + datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out = ROOT / "results/student/runs" / run_id
    out.mkdir(parents=True, exist_ok=False)
    print(f"One A10G, no retries, timeout {TRAIN_TIMEOUT // 60} min: worst case ${TRAIN_TIMEOUT * TRAIN_HOURLY_USD / 3600:.2f}", flush=True)
    result = train_remote.remote(run_id, teacher, json.loads(options))
    for name, data in result["files"].items():
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        (out / name).write_bytes(data)
    print(json.dumps(result["status"], indent=2))


@app.local_entrypoint()
def generate():
    run_id = "gen-" + datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out = ROOT / "data/student" / run_id
    out.mkdir(parents=True, exist_ok=False)
    print(f"One A10G, no retries, timeout {GEN_TIMEOUT // 60} min: worst case ${GEN_TIMEOUT * HOURLY_USD / 3600:.2f}", flush=True)
    result = generate_remote.remote(run_id)
    (out / "teacher.jsonl.gz").write_bytes(result["data"])
    (out / "status.json").write_text(json.dumps(result["status"], indent=2))
    print(json.dumps(result["status"], indent=2))
