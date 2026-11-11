"""Dev loop: score the current code on the dev set, log the run, show what changed.

Runs two configurations on ``data/dev_book.txt``:

* fixed 50%, full-sentence context: the quick signal (word accuracy, and which
  words flipped);
* adaptive, target 100%: the real score (lossless bytes, against zstd). With
  ``--adaptive rounds`` (default) whole sentences are decoded again each round;
  with ``--adaptive l2r`` each word is restored from the text to its left
  (``neural_compression/l2r.py``), with ``--lookback`` earlier sentences in the prompt.

Every run is appended to ``results/dev/runs.jsonl`` and compared with the
previous run on the same sentences. Uncommitted code is what gets tested; the
log records the commit and keeps the uncommitted diff next to the rows.

``data/benchmark.jsonl`` is not used here. Keep it as the held-out test and run
it only to confirm a win found on the dev set.

    uv run --with modal python scripts/dev.py --label "baseline"     # Modal GPUs, one container per configuration
    uv run --with modal python scripts/dev.py --adaptive l2r --label "left to right"
    uv run python scripts/dev.py --local --limit 5 --no-adaptive     # CPU smoke check, no Modal needed
    uv run python scripts/dev.py --history                           # all logged runs
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so ``python scripts/dev.py`` finds neural_compression

from neural_compression import devlog  # noqa: E402
from neural_compression.benchmark import Sample, load_samples  # noqa: E402
from neural_compression.models import DEFAULT_MODEL  # noqa: E402

CODE_PATHS = ["neural_compression", "scripts"]


def run_modal(specs: list[dict], samples: list[Sample]) -> list[dict]:
    """One GPU container per configuration, in parallel (the function from modal_app.py)."""
    from modal_app import app, run_config

    payload = [{"category": s.category, "text": s.text} for s in samples]
    with app.run():
        return list(run_config.map(specs, [payload] * len(specs)))


def run_local(specs: list[dict], samples: list[Sample], device: str | None) -> list[dict]:
    from neural_compression.benchmark import run_adaptive, run_fixed, run_left_to_right
    from neural_compression.models import load_model

    lm = load_model(specs[0]["model"], seed=specs[0]["seed"], device=device)
    results = []
    for spec in specs:
        common = {"beam_size": spec["beam"], "batch_size": spec["batch_size"]}
        if spec["kind"] == "fixed":
            rows = run_fixed(lm, samples, spec["retention"], spec["context"], **common)
        elif spec["kind"] == "l2r":
            rows = run_left_to_right(lm, samples, spec["lookback"], spec["retention"], **common)
        else:
            rows = run_adaptive(lm, samples, spec["context"], spec["retention"], spec["target"], **common)
        results.append({"spec": spec, "rows": rows, "meta": {"device": str(lm.device)}})
    return results


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="", help="what this run changes; shown in the log")
    ap.add_argument("--dataset", default="data/dev_book.txt")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N sentences")
    ap.add_argument("--retention", type=float, default=0.5, help="retention of the fixed configuration")
    ap.add_argument("--no-adaptive", action="store_true", help="skip the adaptive configuration (about 5x the decoding)")
    ap.add_argument("--adaptive", choices=["rounds", "l2r"], default="rounds", help="adaptive method: whole-sentence rounds, or left to right")
    ap.add_argument("--lookback", type=int, default=1, help="with --adaptive l2r: earlier sentences shown in each prompt")
    ap.add_argument("--local", action="store_true", help="run in this process instead of on Modal")
    ap.add_argument("--device", default=None, help="with --local: cpu or cuda (default: cuda if available)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=None, help="default: 64 on Modal, 16 with --local")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/dev")
    ap.add_argument("--history", action="store_true", help="print the logged runs and exit")
    args = ap.parse_args()
    out = ROOT / args.out

    if args.history:
        print(devlog.format_history(devlog.load_entries(out)))
        return

    samples = load_samples(ROOT / args.dataset)
    if args.limit:
        samples = samples[: args.limit]
    base = {
        "model": args.model,
        "beam": args.beam,
        "batch_size": args.batch_size or (16 if args.local else 64),
        "seed": args.seed,
        "context": "full",
    }
    specs = [{**base, "kind": "fixed", "retention": args.retention}]
    if not args.no_adaptive and args.adaptive == "l2r":
        specs.append({**base, "kind": "l2r", "retention": 0.25, "lookback": args.lookback})
    elif not args.no_adaptive:
        specs.append({**base, "kind": "adaptive", "retention": 0.25, "target": 1.0})

    where = "locally" if args.local else "on Modal"
    print(f"{len(samples)} sentences from {args.dataset}, {len(specs)} configurations {where} ...", file=sys.stderr, flush=True)
    started = datetime.now()
    t0 = time.perf_counter()
    results = run_local(specs, samples, args.device) if args.local else run_modal(specs, samples)
    wall = time.perf_counter() - t0

    rows = [row for r in results for row in r["rows"]]
    info = {
        "id": started.strftime("%Y%m%d-%H%M%S"),
        "time": started.astimezone().isoformat(timespec="seconds"),
        "label": args.label,
        "commit": git("rev-parse", "--short", "HEAD").strip(),
        "dirty": bool(git("status", "--porcelain", "--", *CODE_PATHS).strip()),
        "dataset": args.dataset,
        "runner": "local" if args.local else "modal",
        "device": " / ".join(sorted({r["meta"]["device"] for r in results})),
        "model": args.model,
        "beam": args.beam,
        "wall_seconds": wall,
    }
    entry = devlog.make_entry(rows, samples, info)
    prev = devlog.previous_entry(out, entry["samples_sha"])
    flips = devlog.flipped_words(devlog.load_rows(out, prev["id"]), rows) if prev else None
    log = devlog.append_run(out, entry, rows, code_diff=git("diff", "HEAD", "--", *CODE_PATHS))
    print(devlog.format_report(entry, prev, flips))
    print(f"\nlogged to {log.relative_to(ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
