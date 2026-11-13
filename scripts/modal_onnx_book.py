"""Budget-limited base-model ONNX CUDA book reconstruction experiment.

    uv run --frozen --with modal modal run --detach scripts/modal_onnx_book.py

Use --detach: without it Modal cancels the run when the local client disconnects
(run 2 was cancelled mid-encode that way). Results land on the volume regardless.
One L40S, no retries. A separate receiver process decodes the serialized payload
at a different batch size. Calibration selects the whole book if it fits, or a
clearly labelled contiguous prefix. Source bytes are never normalized.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
if ROOT.name == "scripts":
    ROOT = ROOT.parent
# https://modal.com/pricing: L40S + 2 CPU cores + 16 GiB, per second.
HOURLY_USD = (0.000542 + 2 * 0.0000131 + 16 * 0.00000222) * 3600
TIMEOUT_SECONDS = 3900
STARTUP_SECONDS = 600
BUDGET_USD = 3.0
image = (
    modal.Image.debian_slim(python_version="3.14")
    .pip_install("onnxruntime-gpu[cuda,cudnn]==1.31.1", "tokenizers==0.22.2", "huggingface-hub", "numpy", "modal==1.6.4")
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "2"})
    .add_local_dir(ROOT / "neural_compression", "/root/neural_compression")
    .add_local_file(__file__, "/root/modal_onnx_book.py")
)
cache = modal.Volume.from_name("neural-compression-hf-cache", create_if_missing=True)
app = modal.App("dynapress-onnx-cuda-book", image=image)


def units(text: str) -> list[str]:
    from neural_compression.compressor import compress
    from neural_compression.decoder import sentence_units

    doc = compress(text, 1)
    out = [
        (doc.separators[0] if span.start == 0 else "")
        + "".join(doc.prefixes[i] + doc.separators[i + 1] for i in span)
        for span in sentence_units(doc)
    ]
    if "".join(out) != text:
        raise ValueError("unit splitting changed source bytes")
    return out


def worker(phase: str, root: Path):
    """Each phase has its own model/session/process; receiver reads no originals."""
    import onnxruntime as ort

    from neural_compression.compressor import Compressed, split_text
    from neural_compression.decoder import Decoder
    from neural_compression.models import load_model

    ort.preload_dlls(directory="")
    device = "cpu" if phase == "cpu" else "cuda"
    started = time.monotonic()
    if phase == "encode":
        # Enable profiling for one real prefix, then stop it before the book run.
        from unittest.mock import patch

        original_options = ort.SessionOptions

        def profiled_options():
            options = original_options()
            options.enable_profiling = True
            options.profile_file_prefix = str(root / "cuda-profile")
            return options

        with patch.object(ort, "SessionOptions", profiled_options):
            lm = load_model(backend="onnx", device=device, threads=2)
        lm.backend.prefix(lm.encode("A short CUDA execution probe."))
        profile = json.loads(Path(lm.backend.session.end_profiling()).read_text())
        counts = Counter(e.get("args", {}).get("provider") for e in profile if e.get("args", {}).get("provider"))
        ops = Counter(e.get("args", {}).get("op_name") for e in profile if e.get("args", {}).get("provider") == "CUDAExecutionProvider")
        if not any(ops.get(op) for op in ("Gemm", "FusedGemm", "MatMul")):
            raise RuntimeError("No CUDA matrix multiplication in ORT profile; refusing a CPU-only result")
        (root / "runtime.json").write_text(json.dumps({
            "ort": ort.__version__, "model": lm.describe(),
            "available_providers": ort.get_available_providers(),
            "session_providers": lm.backend.session.get_providers(),
            "provider_options": lm.backend.session.get_provider_options(),
            "profile_node_events": dict(counts), "cuda_ops": dict(ops),
            "load_and_probe_seconds": time.monotonic() - started,
            "cache_note": "Weights on GPU, KV cache and beam reordering on host; transfers included in timings.",
        }, indent=2))
        from neural_compression.adaptive import compress_adaptive_many
        from neural_compression.benchmark import merge_payloads

        text = (root / "book.txt").read_bytes().decode("utf-8")
        texts = units(text)
        decoder = Decoder(lm, "full", beam_size=4, batch_size=16)
        n = min(32, len(texts))
        calibration_ids = [i * (len(texts) - 1) // max(1, n - 1) for i in range(n)]
        t0 = time.monotonic()
        calibration = compress_adaptive_many([texts[i] for i in calibration_ids], decoder)
        calibration_seconds = time.monotonic() - t0
        # Reserve 15 min for independent replay/output and leave 2x encode margin.
        available = max(0, float(os.environ["ENCODE_DEADLINE"]) - time.monotonic())
        per_word = calibration_seconds / max(1, sum(len(split_text(texts[i])[1]) for i in calibration_ids))
        allowance = int(available / max(per_word * 2, 1e-6))
        selected, words = [], 0
        for t in texts:
            count = len(split_text(t)[1])
            if words + count > allowance:
                break
            selected.append(t)
            words += count
        if len(selected) < n:
            raise RuntimeError("Calibration cannot fit even 32 contiguous units within encode budget")
        payloads = []
        rounds = []
        calibration_by_id = dict(zip(calibration_ids, calibration))
        encode_seconds = calibration_seconds
        for start in range(0, len(selected), 32):
            indices = list(range(start, min(start + 32, len(selected))))
            pending = [i for i in indices if i not in calibration_by_id]
            t0 = time.monotonic()
            results = compress_adaptive_many([selected[i] for i in pending], decoder) if pending else []
            encode_seconds += time.monotonic() - t0
            found = {**calibration_by_id, **dict(zip(pending, results))}
            for i in indices:
                result = found[i]
                if result.decoded_text != selected[i]:
                    raise RuntimeError(f"Adaptive acceptance not byte-exact at unit {i}")
                payloads.append(result.payload)
                rounds.append(result.rounds)
            # Every checkpoint is independently replayable even after interruption.
            packed = merge_payloads([Compressed.from_bytes(p) for p in payloads], joiner="").to_bytes(use_zstd=True)
            (root / "payload.ncp.zst").write_bytes(packed)
            original = "".join(selected[:len(payloads)]).encode("utf-8")
            (root / "selected.txt").write_bytes(original)
            (root / "payloads.json").write_text(json.dumps([p.hex() for p in payloads]))
            (root / "encode.json").write_text(json.dumps({
                "selected_units": len(payloads), "planned_units": len(selected), "book_units": len(texts),
                "whole_book": len(payloads) == len(texts), "words": len(split_text(original.decode())[1]),
                "original_bytes": len(original), "original_sha256": hashlib.sha256(original).hexdigest(),
                "payload_bytes": len(packed), "summed_payload_bytes": sum(map(len, payloads)),
                "encode_seconds_including_calibration": encode_seconds,
                "mean_rounds": sum(rounds) / len(rounds), "calibration_seconds": calibration_seconds,
                "calibration_units": n, "batch_size": 16, "beam": 4,
            }, indent=2))
            print(f"Encoded {len(payloads)}/{len(selected)} selected units ({len(texts)} whole book)", flush=True)
            if time.monotonic() >= float(os.environ["ENCODE_DEADLINE"]):
                break
    else:
        lm = load_model(backend="onnx", device=device, threads=2)
        decoder = Decoder(lm, "full", beam_size=4, batch_size=7)
        t0 = time.monotonic()
        if phase == "decode":
            decoded = decoder.decode((root / "payload.ncp.zst").read_bytes()).text.encode("utf-8")
            (root / "decoded.txt").write_bytes(decoded)
            result = {"decoded_bytes": len(decoded), "decoded_sha256": hashlib.sha256(decoded).hexdigest()}
        else:
            payloads = [bytes.fromhex(p) for p in json.loads((root / "payloads.json").read_text())[:8]]
            decoded = [r.text for r in decoder.decode_many(payloads)]
            (root / "cpu-decoded.json").write_text(json.dumps(decoded))
            result = {"units": len(decoded)}
        result.update({"seconds": time.monotonic() - t0, "load_and_decode_seconds": time.monotonic() - started, "batch_size": 7})
        (root / f"{phase}.json").write_text(json.dumps(result, indent=2))


@app.function(
    gpu="L40S", cpu=(2, 2), memory=(16384, 16384), max_containers=1,
    retries=0, timeout=TIMEOUT_SECONDS, startup_timeout=STARTUP_SECONDS,
    volumes={"/cache": cache},
)
def run_book(run_id: str, body: bytes, book: dict) -> dict:
    started = time.monotonic()
    root = Path("/cache/onnx-book") / run_id
    root.mkdir(parents=True, exist_ok=False)
    (root / "book.txt").write_bytes(body)
    env = {**os.environ, "PYTHONPATH": "/root", "ENCODE_DEADLINE": str(started + 2600)}
    summary = {"book": book, "run_id": run_id, "budget_usd": BUDGET_USD,
               "timeout_seconds": TIMEOUT_SECONDS, "hourly_usd": HOURLY_USD, "phases": {},
               "status": "incomplete"}  # stays if the input is cancelled (not an Exception)
    summary["gpu"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], text=True).strip()
    try:
        for phase, limit in (("encode", 2700), ("decode", 800), ("cpu", 180)):
            remaining = min(limit, started + TIMEOUT_SECONDS - 90 - time.monotonic())
            if remaining <= 0:
                raise RuntimeError("No verification budget remaining")
            with (root / f"{phase}.log").open("w") as log:
                try:
                    p = subprocess.run([sys.executable, "/root/modal_onnx_book.py", phase, str(root)],
                                       stdout=log, stderr=subprocess.STDOUT, env=env, timeout=remaining)
                    summary["phases"][phase] = {"returncode": p.returncode}
                except subprocess.TimeoutExpired:
                    summary["phases"][phase] = {"timeout": True}
                    if phase != "encode" or not (root / "payload.ncp.zst").exists():
                        raise
            print((root / f"{phase}.log").read_text()[-4000:], flush=True)
            cache.commit()
            if summary["phases"][phase].get("returncode", 0) != 0:
                raise RuntimeError(f"{phase} subprocess failed")
            if phase == "decode":
                summary["fresh_process_byte_exact"] = (root / "decoded.txt").read_bytes() == (root / "selected.txt").read_bytes()
                if not summary["fresh_process_byte_exact"]:
                    raise RuntimeError("Fresh CUDA receiver did not reconstruct byte-exactly")
            if phase == "cpu":
                original_units = units((root / "selected.txt").read_bytes().decode())
                got = json.loads((root / "cpu-decoded.json").read_text())
                summary["cpu_cross_device_subset"] = {"exact": sum(a == b for a, b in zip(got, original_units)), "total": len(got)}
        summary["status"] = "complete"
    except Exception as exc:
        summary.update({"status": "incomplete", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        elapsed = time.monotonic() - started
        summary.update({"runtime_seconds": elapsed, "estimated_compute_usd": elapsed * HOURLY_USD / 3600,
                        "cost_note": "Runtime estimate, not invoice; image build/storage excluded."})
        for key in ("runtime", "encode", "decode", "cpu"):
            path = root / f"{key}.json"
            if path.exists():
                summary[key] = json.loads(path.read_text())
        (root / "summary.json").write_text(json.dumps(summary, indent=2))
        cache.commit()
    return {"summary": summary, "files": {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}}


@app.local_entrypoint()
def main():
    # Reuse the existing checksum-pinned Alice preparation, without its LoRA run.
    from scripts.modal_book import book_body

    body, book = book_body()
    run_id = "onnx-cuda-alice-" + datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out = ROOT / "results/lora" / run_id
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(ROOT / "neural_compression", out / "source/neural_compression", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copyfile(__file__, out / "source/modal_onnx_book.py")
    print(f"One L40S, no retries; {len(body)} source bytes; max compute+startup estimate ${(TIMEOUT_SECONDS + STARTUP_SECONDS) * HOURLY_USD / 3600:.2f}; output {out}", flush=True)
    result = run_book.remote(run_id, body, book)
    for name, data in result["files"].items():
        (out / name).write_bytes(data)
    print(json.dumps(result["summary"], indent=2))
    if result["summary"]["status"] != "complete":
        raise RuntimeError(f"Experiment incomplete; inspect {out}")


if __name__ == "__main__":
    worker(sys.argv[1], Path(sys.argv[2]))
