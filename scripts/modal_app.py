"""Run the benchmark on Modal GPUs, one container per configuration.

Each (retention, context) configuration and each adaptive target runs in its
own GPU container in parallel. The rows are merged locally, then exported and
plotted exactly as the local ``benchmark`` command would.

    uv run --with modal modal run scripts/modal_app.py                     # full grid, all samples
    uv run --with modal modal run scripts/modal_app.py --limit-per-category 1 --contexts "0 full" --retentions 0.5 --adaptive-targets ""

Model weights are cached in the Modal volume ``neural-compression-hf-cache``.
GPU results can differ slightly from CPU results (different floating-point
kernels can flip near-ties in beam search). The device is recorded in the metadata.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
# Fallback order: Modal starts each container on the first type with free capacity.
GPUS = ["L4", "A10G", "L40S", "T4"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.15.0", "transformers==5.22.2", "zstandard>=0.23", "numpy")
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(ROOT / "neural_compression", "/root/neural_compression")
)
cache = modal.Volume.from_name("neural-compression-hf-cache", create_if_missing=True)
app = modal.App("neural-compression", image=image)


@app.function(gpu=GPUS, volumes={"/cache": cache}, timeout=3 * 3600)
def run_config(spec: dict, samples: list[dict]) -> dict:
    sys.path.insert(0, "/root")
    import torch
    import transformers

    from neural_compression.benchmark import Sample, run_adaptive, run_fixed
    from neural_compression.decoder import parse_context
    from neural_compression.models import load_model

    t0 = time.perf_counter()
    lm = load_model(spec["model"], seed=spec["seed"])
    cache.commit()  # persist downloaded weights for the other containers / next run
    data = [Sample(**s) for s in samples]
    context = parse_context(spec["context"])
    if spec["kind"] == "fixed":
        rows = run_fixed(lm, data, spec["retention"], context, beam_size=spec["beam"], batch_size=spec["batch_size"])
    else:
        rows = run_adaptive(
            lm, data, context, spec["retention"], spec["target"], beam_size=spec["beam"], batch_size=spec["batch_size"]
        )
    meta = {
        **lm.describe(),
        "device": f"{lm.device} ({torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'})",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
    }
    return {"spec": spec, "rows": rows, "meta": meta, "seconds": time.perf_counter() - t0}


@app.local_entrypoint()
def main(
    dataset: str = "data/benchmark.jsonl",
    retentions: str = "0.25 0.5 0.75",
    contexts: str = "0 2 4 8 full",
    adaptive_targets: str = "1.0 0.9",
    adaptive_start: float = 0.25,
    adaptive_context: str = "full",
    limit_per_category: int = 0,
    model: str = "HuggingFaceTB/SmolLM2-360M",
    beam: int = 4,
    batch_size: int = 64,
    seed: int = 0,
    out: str = "results/gpu",
):
    from neural_compression.benchmark import corpus_baselines, export, load_samples, summarize
    from neural_compression.plots import device_label, plot_all

    samples = load_samples(ROOT / dataset, limit_per_category or None)
    payload = [{"category": s.category, "text": s.text} for s in samples]
    base = {"model": model, "beam": beam, "batch_size": batch_size, "seed": seed}
    specs = [
        {**base, "kind": "fixed", "retention": float(r), "context": c}
        for r in retentions.split()
        for c in contexts.split()
    ]
    specs += [
        {**base, "kind": "adaptive", "retention": adaptive_start, "context": adaptive_context, "target": float(t)}
        for t in adaptive_targets.split()
    ]
    # Each finished configuration is saved immediately, so an interrupted run resumes.
    parts = ROOT / out / "_parts"
    parts.mkdir(parents=True, exist_ok=True)
    done = {}
    for spec in specs:
        f = parts / f"{_name(spec)}.json"
        if f.exists():
            saved = json.loads(f.read_text())
            if saved["spec"] == spec and saved["samples"] == len(samples):
                done[_name(spec)] = saved
    todo = [s for s in specs if _name(s) not in done]
    print(f"{len(samples)} samples, {len(specs)} configurations ({len(done)} cached), GPUs: {' > '.join(GPUS)}")

    t0 = time.perf_counter()
    results = list(done.values())
    for result in run_config.map(todo, [payload] * len(todo), order_outputs=False):
        result["samples"] = len(samples)
        (parts / f"{_name(result['spec'])}.json").write_text(json.dumps(result))
        results.append(result)
        new = result["rows"]
        s = summarize(new)[0]
        spec = result["spec"]
        label = f"{spec['kind']} r={spec['retention']} ctx={spec['context']}" + (f" target={spec['target']}" if spec["kind"] == "adaptive" else "")
        print(
            f"{label:<38} word acc {s['word_accuracy']:.3f}  sent acc {s['sentence_accuracy']:.3f}  "
            f"CER {s['cer']:.3f}  ratio {s['ratio_payload']:.2f}  decode {s['decode_seconds']:.0f}s  "
            f"({result['meta']['device']}, container {result['seconds']:.0f}s)",
            flush=True,
        )
    rows = [row for r in results for row in r["rows"]]
    meta = dict(results[0]["meta"])
    meta.pop("device")
    meta["devices"] = {_name(r["spec"]): r["meta"]["device"] for r in results}
    meta.update(
        {
            "dataset": dataset,
            "samples": len(samples),
            "beam": beam,
            "batch_size": batch_size,
            "seed": seed,
            "runner": "modal",
            "wall_seconds": time.perf_counter() - t0,
            "corpus": corpus_baselines(samples, [float(r) for r in retentions.split()]),
            "note": "Model weights are not included in per-document sizes; see 'size_bytes'.",
        }
    )
    # Stable row order regardless of which container finished first.
    order = {(s["kind"], s["retention"], s["context"], s.get("target")): i for i, s in enumerate(specs)}
    rows.sort(key=lambda r: order.get(_key(r), 0))
    paths = export(rows, ROOT / out, meta)
    charts = plot_all(summarize(rows), ROOT / out, meta["corpus"], device_label(meta))
    for p in [*paths.values(), *charts]:
        print(p)


def _name(spec: dict) -> str:
    name = f"{spec['kind']}_r{spec['retention']}_ctx{spec['context']}"
    return name + (f"_t{spec['target']}" if spec["kind"] == "adaptive" else "")


def _key(row: dict) -> tuple:
    if row["mode"] == "fixed":
        return ("fixed", row["retention"], row["context"], None)
    return ("adaptive", row["retention"], row["context"], float(row["mode"].split("@")[1]))
