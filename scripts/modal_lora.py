"""Run one calibrated LoRA pilot on an L4/A10G, with a $5 compute safety budget.

    uv run --with modal modal run scripts/modal_lora.py

Options: --model/--revision train another base model (a revision must be pinned),
--budget lowers the safety budget (the function timeout shrinks to match),
--gpu A10G pins one GPU type, and --control-run evaluates an earlier run (for
example pilot-20261110-181019) again in the same container, so decode times are
measured on the same hardware.

Only train/validation data is uploaded. Calibration uses the first 30 optimizer
steps of the real run; it does not discard training or start a second GPU job.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import modal

ROOT = Path(__file__).resolve().parent.parent
HOURLY_USD = 1.1 + 2 * 0.0473 + 8 * 0.008  # budget at the more expensive A10G rate
BUDGET_USD = 5.0
RESERVE_USD = 0.75  # startup/build/transfer/storage contingency, not GPU runtime
TIMEOUT_SECONDS = 3 * 3600  # A10G + capped CPU/RAM: $3.78 for three hours


def timeout_for(budget_usd: float) -> int:
    """Function timeout whose worst-case A10G runtime plus reserve fits the budget."""
    return int(min(TIMEOUT_SECONDS, (budget_usd - RESERVE_USD) * 3600 / HOURLY_USD))


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
    .env(
        {
            "HF_HOME": "/cache/hf",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "2",
        }
    )
    .add_local_dir(ROOT / "neural_compression", "/root/neural_compression")
    .add_local_file(ROOT / "data/lora/train.jsonl", "/data/lora/train.jsonl")
    .add_local_file(ROOT / "data/lora/validation.jsonl", "/data/lora/validation.jsonl")
    .add_local_file(ROOT / "data/lora/manifest.json", "/data/lora/manifest.json")
)
cache = modal.Volume.from_name("neural-compression-hf-cache")
app = modal.App("neural-compression-lora-pilot", image=image)


@app.function(
    gpu=["L4", "A10G"],
    cpu=(2, 2),
    memory=(8192, 8192),
    max_containers=1,
    retries=0,
    timeout=TIMEOUT_SECONDS,
    startup_timeout=600,
    volumes={"/cache": cache},
)
def run_pilot(
    run_id: str,
    model: str | None = None,
    revision: str | None = None,
    budget_usd: float = BUDGET_USD,
    control_run: str | None = None,
) -> dict:
    sys.path.insert(0, "/root")
    import torch
    from transformers import TrainerCallback

    from neural_compression.finetune import evaluate, train
    from neural_compression.models import DEFAULT_MODEL, DEFAULT_REVISION

    started = time.monotonic()
    timeout_seconds = timeout_for(budget_usd)
    root = Path("/cache/lora") / run_id
    root.mkdir(parents=True, exist_ok=False)
    data = Path("/data/lora")
    info = json.loads((data / "manifest.json").read_text())
    calibration = {}

    class BudgetGuard(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            elapsed = time.monotonic() - started
            if elapsed > timeout_seconds - 20 * 60:
                raise RuntimeError(
                    "Budget guard: reserve the final 20 minutes for export/evaluation"
                )
            if state.global_step == 5:
                self.warm = time.monotonic()
            if state.global_step == 30:
                seconds_per_step = (time.monotonic() - self.warm) / 25
                # Four complete validation-loss passes, conservatively charged at
                # one training microbatch each; also reserve 15 minutes for decoding
                # (and 15 more when a control run is evaluated again).
                eval_batches = 4 * (
                    (2 * info["splits"]["validation"]["sentences"] + 1) // 2
                )
                remaining_equivalent_steps = (
                    state.max_steps - state.global_step + eval_batches / 8
                )
                projected_seconds = (
                    elapsed
                    + 1.5 * seconds_per_step * remaining_equivalent_steps
                    + 900 * (2 if control_run else 1)
                )
                projected_usd = projected_seconds * HOURLY_USD / 3600 + RESERVE_USD
                calibration.update(
                    {
                        "optimizer_steps": state.max_steps,
                        "measured_steps": 25,
                        "seconds_per_step": seconds_per_step,
                        "elapsed_seconds": elapsed,
                        "projected_seconds_with_margin": projected_seconds,
                        "projected_usd_with_reserve": projected_usd,
                        "hourly_usd": HOURLY_USD,
                        "gpu": torch.cuda.get_device_name(),
                        "budget_usd": budget_usd,
                    }
                )
                (root / "calibration.json").write_text(
                    json.dumps(calibration, indent=2)
                )
                print("CALIBRATION " + json.dumps(calibration), flush=True)
                cache.commit()
                if (
                    projected_usd > budget_usd
                    or projected_seconds > timeout_seconds - 300
                ):
                    raise RuntimeError(
                        "Budget guard: calibration projects beyond the pilot budget/time limit"
                    )

    error = None
    try:
        train(
            SimpleNamespace(
                data=data,
                out=root / "run",
                model=model or DEFAULT_MODEL,
                revision=revision or DEFAULT_REVISION,
                seed=0,
                epochs=3,
                max_steps=-1,
                max_length=1024,
                batch_size=2,
                accumulation=8,
                cpu=False,
            ),
            callbacks=[BudgetGuard()],
        )
        # Training's Trainer must release GPU allocations before real decoding.
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        cache.commit()
        runs = [(root / "run", root / "validation")]
        if control_run:
            runs.append((Path("/cache/lora") / control_run / "run", root / "control"))
        for run, out in runs:
            if time.monotonic() - started > timeout_seconds - 900:
                raise RuntimeError(
                    f"Budget guard: insufficient time remains to evaluate {run}"
                )
            evaluate(
                SimpleNamespace(
                    data=data,
                    run=run,
                    out=out,
                    split="validation",
                    final_test=False,
                    per_book=25,
                    beam=4,
                    batch_size=16,
                    device="cuda",
                    threads=2,
                )
            )
            cache.commit()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logging.getLogger(__name__).exception("LoRA pilot stopped")
    finally:
        elapsed = time.monotonic() - started
        gpu_name = torch.cuda.get_device_name()
        actual_hourly_usd = (
            (0.8 if "L4" == gpu_name.removeprefix("NVIDIA ") else 1.1)
            + 2 * 0.0473
            + 8 * 0.008
        )
        status = {
            "run_id": run_id,
            "status": "failed" if error else "complete",
            "error": error,
            "gpu_seconds": elapsed,
            "estimated_compute_usd": elapsed * actual_hourly_usd / 3600,
            "gpu": gpu_name,
            "budget_usd": budget_usd,
            "reserve_usd": RESERVE_USD,
            "timeout_seconds": timeout_seconds,
            "model": model or DEFAULT_MODEL,
            "revision": revision or DEFAULT_REVISION,
            "control_run": control_run,
            "calibration": calibration,
            "remote_directory": str(root),
            "volume": "neural-compression-hf-cache",
            "note": "Runtime-based estimate, not a finalized invoice; retained storage billed separately.",
        }
        (root / "status.json").write_text(json.dumps(status, indent=2))
        cache.commit()
    # Download small useful results and the reusable adapter, not 1.45 GB of merged
    # weights. The merged decoder model remains on the volume for reproducibility.
    files = {}
    for path in root.rglob("*"):
        if (
            path.is_file()
            and "checkpoints" not in path.parts
            and "merged" not in path.parts
        ):
            files[str(path.relative_to(root))] = path.read_bytes()
    return {"status": status, "files": files}


@app.local_entrypoint()
def main(
    model: str = "",
    revision: str = "",
    budget: float = BUDGET_USD,
    gpu: str = "",
    control_run: str = "",
):
    if model and not revision:
        raise ValueError("pin --revision for a non-default --model")
    if budget > BUDGET_USD:
        raise ValueError(f"--budget cannot exceed ${BUDGET_USD}")
    prefix = model.rsplit("/", 1)[-1].lower() if model else "pilot"
    run_id = prefix + "-" + datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out = ROOT / "results/lora" / run_id
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(
        ROOT / "neural_compression",
        out / "source/neural_compression",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copyfile(__file__, out / "source/modal_lora.py")
    options = {"timeout": timeout_for(budget)}
    if gpu:
        options["gpu"] = gpu
    print(
        f"One {gpu or 'L4/A10G'}, no retries; ${budget} budget "
        f"(timeout {options['timeout']} s); outputs: {out}",
        flush=True,
    )
    result = run_pilot.with_options(**options).remote(
        run_id, model or None, revision or None, budget, control_run or None
    )
    for name, data in result["files"].items():
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    print(json.dumps(result["status"], indent=2))
    if result["status"]["status"] != "complete":
        raise RuntimeError(f"Pilot stopped: {result['status']['error']}; see {out}")
