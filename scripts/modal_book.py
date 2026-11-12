"""Compare the saved LoRA and base model on the complete Alice book body.

    uv run --with modal modal run scripts/modal_book.py

No training or held-out test data. One GPU, no retries, a three-hour timeout,
and calibration/progress guards against a $5 budget including $0.75 reserve.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
PILOT = "pilot-20261110-181019"
HOURLY_USD = 1.1 + 2 * 0.0473 + 8 * 0.008
BUDGET_USD = 5.0
RESERVE_USD = 0.75
TIMEOUT_SECONDS = 3 * 3600
BATCH_UNITS = 64

image = (
    modal.Image.debian_slim(python_version="3.14")
    .pip_install("torch==2.15.0", "transformers==5.22.2", "numpy")
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "2"})
    .add_local_dir(ROOT / "neural_compression", "/root/neural_compression")
)
cache = modal.Volume.from_name("neural-compression-hf-cache")
app = modal.App("neural-compression-whole-book", image=image)


def book_body() -> tuple[bytes, dict]:
    books = json.loads((ROOT / "data/lora_books.json").read_text())["books"]
    book = next(b for b in books if b["id"] == "gutenberg-11")
    name = hashlib.sha256(book["url"].encode()).hexdigest() + ".txt"
    raw = (ROOT / "data/lora/books" / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != book["sha256"]:
        raise ValueError("book source checksum mismatch")
    start = re.search(rb"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n", raw, re.I)
    end = re.search(rb"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK", raw, re.I)
    if not start or not end or start.end() >= end.start():
        raise ValueError("incomplete Gutenberg wrapper")
    body = raw[start.end():end.start()]
    return body, {**book, "body_sha256": hashlib.sha256(body).hexdigest(), "body_bytes": len(body)}


def book_units(text: str) -> list[str]:
    from neural_compression.compressor import compress
    from neural_compression.decoder import sentence_units

    doc = compress(text, 1)
    units = []
    for span in sentence_units(doc):
        unit = doc.separators[0] if span.start == 0 else ""
        unit += "".join(doc.prefixes[i] + doc.separators[i + 1] for i in span)
        units.append(unit)
    if "".join(units) != text:
        raise ValueError("unit boundaries changed the book")
    return units


@app.function(
    gpu=["L4", "A10G"], cpu=(2, 2), memory=(8192, 8192),
    max_containers=1, retries=0, timeout=TIMEOUT_SECONDS,
    startup_timeout=600, volumes={"/cache": cache},
)
def run_book(run_id: str, body: bytes, book: dict) -> dict:
    sys.path.insert(0, "/root")
    import gc
    import traceback

    import torch
    import transformers
    from compression import zstd

    from neural_compression.adaptive import compress_adaptive_many
    from neural_compression.benchmark import merge_payloads
    from neural_compression.compressor import Compressed, split_text
    from neural_compression.decoder import Decoder
    from neural_compression.models import load_model

    started = time.monotonic()
    root = Path("/cache/lora") / run_id
    root.mkdir(parents=True, exist_ok=False)
    text = body.decode("utf-8")
    units = book_units(text)
    pilot = Path("/cache/lora") / PILOT / "run"
    experiment = json.loads((pilot / "experiment.json").read_text())
    summary = {
        "book": book, "decoder_units": len(units),
        "words": len(split_text(text)[1]), "pilot": PILOT,
        "beam": 4, "batch_size": 16, "context": "full (sentence-sized units)",
        "torch": torch.__version__, "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(), "models": {},
        "note": "Model weights excluded from payload sizes. No training or book-wide vocabulary adaptation.",
    }
    (root / "original.txt").write_bytes(body)
    t0 = time.perf_counter()
    baseline = zstd.compress(body, level=19)
    encode_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    assert zstd.decompress(baseline) == body
    summary["zstd"] = {
        "bytes": len(baseline), "encode_seconds": encode_seconds,
        "decode_seconds": time.perf_counter() - t0, "exact": True,
    }
    (root / "plain.zst").write_bytes(baseline)

    def save():
        (root / "comparison.json").write_text(json.dumps(summary, indent=2))
        cache.commit()

    def guard(projected_seconds: float):
        elapsed = time.monotonic() - started
        projected = elapsed + projected_seconds
        if projected > TIMEOUT_SECONDS - 300 or projected * HOURLY_USD / 3600 + RESERVE_USD > BUDGET_USD:
            raise RuntimeError("Budget guard: projected remaining work exceeds the budget/time limit")

    error = None
    try:
        for label, model, revision in (
            ("base", experiment["base_model"], experiment["revision"]),
            ("lora", str(pilot / "merged"), None),
        ):
            guard(300)
            lm = load_model(model, revision=revision, device="cuda", threads=2, seed=experiment["seed"])
            decoder = Decoder(lm, "full", beam_size=4, batch_size=16)
            payloads = [None] * len(units)
            rounds = [0] * len(units)
            # Spread calibration across the whole book, not just short headings.
            n = min(BATCH_UNITS, len(units))
            calibration = [i * (len(units) - 1) // max(1, n - 1) for i in range(n)]
            remaining = [i for i in range(len(units)) if i not in set(calibration)]
            batches = [calibration] + [remaining[i:i + BATCH_UNITS] for i in range(0, len(remaining), BATCH_UNITS)]
            encode_total = 0.0
            finished = 0
            calibration_decode = 0.0
            for batch_no, indices in enumerate(batches):
                if batch_no:
                    guard(1.5 * encode_total / finished * len(indices) + 300)
                t0 = time.perf_counter()
                results = compress_adaptive_many([units[i] for i in indices], decoder)
                encode_total += time.perf_counter() - t0
                for i, result in zip(indices, results):
                    if result.decoded_text != units[i]:
                        raise RuntimeError(f"adaptive reconstruction differs at unit {i}")
                    payloads[i], rounds[i] = result.payload, result.rounds
                finished += len(indices)
                if batch_no == 0:
                    t0 = time.perf_counter()
                    verified = decoder.decode_many([payloads[i] for i in indices])
                    calibration_decode = time.perf_counter() - t0
                    if any(v.text != units[i] for i, v in zip(indices, verified)):
                        raise RuntimeError("independent calibration decode differs")
                encode_per_unit = encode_total / finished
                decode_per_unit = calibration_decode / len(calibration)
                remaining_seconds = (len(units) - finished) * encode_per_unit + len(units) * decode_per_unit
                if label == "base":
                    remaining_seconds += len(units) * (encode_per_unit + decode_per_unit) + 60
                projected = time.monotonic() - started + 1.5 * remaining_seconds + 120
                summary["models"][label] = {
                    "status": "compressing", "finished_units": finished,
                    "encode_seconds": encode_total,
                    "projected_total_seconds_with_margin": projected,
                    "projected_usd_with_reserve": projected * HOURLY_USD / 3600 + RESERVE_USD,
                    "model": lm.describe(),
                }
                (root / f"{label}-progress.json").write_text(json.dumps({
                    "payload_hex": [p.hex() if p is not None else None for p in payloads],
                    "rounds": rounds,
                }))
                print(f"{label}: {finished}/{len(units)} units; projected ${projected * HOURLY_USD / 3600 + RESERVE_USD:.2f} including reserve", flush=True)
                save()
                guard(1.5 * remaining_seconds + 120)
            merged = merge_payloads([Compressed.from_bytes(p) for p in payloads], joiner="")
            packed = merged.to_bytes(use_zstd=True)
            (root / f"{label}.ncp.zst").write_bytes(packed)
            # Fresh decoder gets only one serialized book payload and the model.
            del decoder, results
            gc.collect()
            decoder = Decoder(lm, "full", beam_size=4, batch_size=16)
            t0 = time.perf_counter()
            decoded = decoder.decode(packed).text.encode("utf-8")
            decode_seconds = time.perf_counter() - t0
            if decoded != body:
                raise RuntimeError(f"{label}: complete book failed byte-exact independent decode")
            (root / f"{label}.decoded.txt").write_bytes(decoded)
            summary["models"][label].update({
                "status": "complete", "payload_bytes": len(merged.to_bytes()),
                "payload_zstd_bytes": len(packed), "decode_seconds": decode_seconds,
                "mean_rounds": sum(rounds) / len(rounds), "exact": True,
                "decoded_sha256": hashlib.sha256(decoded).hexdigest(),
            })
            save()
            del decoder, lm, verified
            gc.collect()
            torch.cuda.empty_cache()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (root / "error.txt").write_text(traceback.format_exc())
        print(traceback.format_exc(), flush=True)
    finally:
        elapsed = time.monotonic() - started
        gpu_rate = 0.8 if torch.cuda.get_device_name().removeprefix("NVIDIA ") == "L4" else 1.1
        summary.update({
            "status": "failed" if error else "complete", "error": error,
            "gpu_seconds": elapsed,
            "estimated_compute_usd": elapsed / 3600 * (gpu_rate + 2 * 0.0473 + 8 * 0.008),
            "budget_usd": BUDGET_USD, "reserve_usd": RESERVE_USD,
            "cost_note": "Runtime estimate, not an invoice; retained volume storage billed separately.",
        })
        save()
    return {"summary": summary, "files": {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}}


@app.local_entrypoint()
def main():
    body, book = book_body()
    run_id = "book-alice-" + datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out = ROOT / "results/lora" / run_id
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(ROOT / "neural_compression", out / "source/neural_compression", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copyfile(__file__, out / "source/modal_book.py")
    print(f"Whole book: {len(body)} bytes; $5 cap; outputs: {out}", flush=True)
    result = run_book.remote(run_id, body, book)
    for name, data in result["files"].items():
        (out / name).write_bytes(data)
    print(json.dumps(result["summary"], indent=2))
    if result["summary"]["status"] != "complete":
        raise RuntimeError(f"Book comparison stopped; inspect {out}")
