"""Command-line interface.

    python -m neural_compression compress input.txt --retention 0.5 [-o out.bin]
    python -m neural_compression compress input.txt --adaptive --target 1.0 --context full
    python -m neural_compression decompress compressed.bin --context 4 [-o out.txt]
    python -m neural_compression benchmark --dataset data/benchmark.jsonl
    python -m neural_compression plot --results results/
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

from .compressor import Compressed, compress
from .decoder import parse_context
from .models import DEFAULT_MODEL, DEFAULT_REVISION


def _model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"HF model id (default {DEFAULT_MODEL})")
    p.add_argument("--revision", default=None, help=f"model commit (default model is pinned to {DEFAULT_REVISION[:10]})")
    p.add_argument("--beam", type=int, default=4)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None, help="cpu (the NumPy backend is CPU-only)")
    p.add_argument("--batch-size", type=int, default=16, help="decoder requests per model batch")


def _load(args):
    from .models import load_model

    t0 = time.perf_counter()
    lm = load_model(args.model, args.revision, threads=args.threads, seed=args.seed, device=args.device)
    print(
        f"loaded {args.model} ({lm.parameters / 1e6:.0f}M params, {lm.size_bytes / 1e6:.0f} MB) on {lm.device} in {time.perf_counter() - t0:.1f}s",
        file=sys.stderr,
    )
    return lm


def cmd_compress(args) -> None:
    text = Path(args.input).read_text(encoding="utf-8")
    out = Path(args.output or Path(args.input).with_suffix(".bin"))
    if args.adaptive:
        from .adaptive import compress_adaptive
        from .decoder import Decoder

        decoder = Decoder(_load(args), parse_context(args.context), beam_size=args.beam, batch_size=args.batch_size)
        result = compress_adaptive(text, decoder, start_retention=args.retention, target_accuracy=args.target, max_rounds=args.max_rounds)
        c = Compressed.from_bytes(result.payload)
        print(f"adaptive: {result.rounds} rounds, verified word accuracy {result.word_accuracy:.3f} (decoder context {args.context})", file=sys.stderr)
    else:
        c = compress(text, args.retention, verbatim_numbers=not args.truncate_numbers)
    payload = c.to_bytes(use_zstd=args.zstd)
    out.write_bytes(payload)
    stats = c.stats(text)
    stats["written_bytes"] = len(payload)
    print(json.dumps(stats, indent=2))
    print(f"wrote {out}", file=sys.stderr)


def cmd_decompress(args) -> None:
    from .decoder import Decoder

    payload = Path(args.input).read_bytes()
    decoder = Decoder(_load(args), parse_context(args.context), beam_size=args.beam, batch_size=args.batch_size)
    result = decoder.decode(Compressed.from_bytes(payload))
    if args.output:
        Path(args.output).write_text(result.text, encoding="utf-8")
    else:
        sys.stdout.write(result.text + ("" if result.text.endswith("\n") else "\n"))
    s = result.stats
    print(
        f"decoded in {result.seconds:.2f}s: {s.forward_calls:.0f} forward passes, {s.sequences:.0f} sequences, {result.failures} failures",
        file=sys.stderr,
    )


def cmd_benchmark(args) -> None:
    import numpy
    import tokenizers

    from .benchmark import corpus_baselines, export, load_samples, run_adaptive, run_fixed, summarize
    from .plots import device_label, plot_all

    samples = load_samples(args.dataset, args.limit_per_category)
    lm = _load(args)
    contexts = [parse_context(c) for c in args.contexts]
    rows: list[dict] = []
    for retention in args.retentions:
        for context in contexts:
            t0 = time.perf_counter()
            new = run_fixed(lm, samples, retention, context, beam_size=args.beam, batch_size=args.batch_size)
            rows += new
            s = summarize(new)[0]
            print(
                f"fixed r={retention:.2f} ctx={context!s:>4}: word acc {s['word_accuracy']:.3f}  sent acc {s['sentence_accuracy']:.3f}  "
                f"CER {s['cer']:.3f}  ratio {s['ratio_payload']:.2f}  {time.perf_counter() - t0:.0f}s",
                file=sys.stderr,
                flush=True,
            )
    for target in args.adaptive_targets:
        t0 = time.perf_counter()
        context = parse_context(args.adaptive_context)
        new = run_adaptive(lm, samples, context, args.adaptive_start, target, beam_size=args.beam, batch_size=args.batch_size)
        rows += new
        s = summarize(new)[0]
        print(
            f"adaptive target={target} ctx={context}: word acc {s['word_accuracy']:.3f}  sent acc {s['sentence_accuracy']:.3f}  "
            f"ratio {s['ratio_payload']:.2f}  rounds {s['mean_rounds']:.1f}  {time.perf_counter() - t0:.0f}s",
            file=sys.stderr,
            flush=True,
        )
    meta = {
        **lm.describe(),
        "device": str(lm.device),
        "batch_size": args.batch_size,
        "dataset": str(args.dataset),
        "samples": len(samples),
        "beam": args.beam,
        "seed": args.seed,
        "numpy": numpy.__version__,
        "tokenizers": tokenizers.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "corpus": corpus_baselines(samples, args.retentions),
        "note": "Model weights are not included in per-document sizes; see 'size_bytes'.",
    }
    paths = export(rows, args.out, meta)
    charts = plot_all(summarize(rows), args.out, meta["corpus"], device_label(meta))
    for p in [*paths.values(), *charts]:
        print(p, file=sys.stderr)


def cmd_plot(args) -> None:
    from .plots import plot_from_results

    for p in plot_from_results(args.results):
        print(p)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="neural_compression", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("compress", help="compress a UTF-8 text file")
    p.add_argument("input")
    p.add_argument("-o", "--output")
    p.add_argument("--retention", type=float, default=0.5, help="fraction of each word kept (start value in adaptive mode)")
    p.add_argument("--zstd", action="store_true", help="entropy-code the payload body with zstd")
    p.add_argument("--truncate-numbers", action="store_true", help="truncate words containing digits too")
    p.add_argument("--adaptive", action="store_true", help="grow prefixes until the decoder reconstructs them")
    p.add_argument("--target", type=float, default=1.0, help="adaptive: target word accuracy")
    p.add_argument("--max-rounds", type=int, default=6)
    p.add_argument("--context", default="full", help="adaptive: decoder context the receiver will use")
    _model_args(p)
    p.set_defaults(func=cmd_compress)

    p = sub.add_parser("decompress", help="reconstruct text from a payload")
    p.add_argument("input")
    p.add_argument("-o", "--output")
    p.add_argument("--context", default="full", help="0, 2, 4, 8, ... neighbouring words, or 'full'")
    _model_args(p)
    p.set_defaults(func=cmd_decompress)

    p = sub.add_parser("benchmark", help="run the evaluation grid and write CSV/JSON/charts")
    p.add_argument("--dataset", default="data/benchmark.jsonl")
    p.add_argument("--retentions", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    p.add_argument("--contexts", nargs="+", default=["0", "2", "4", "8", "full"])
    p.add_argument("--adaptive-targets", type=float, nargs="*", default=[1.0])
    p.add_argument("--adaptive-start", type=float, default=0.25)
    p.add_argument("--adaptive-context", default="full")
    p.add_argument("--limit-per-category", type=int, default=None)
    p.add_argument("--out", default="results")
    _model_args(p)
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("plot", help="re-draw charts from results/results.json")
    p.add_argument("--results", default="results")
    p.set_defaults(func=cmd_plot)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
