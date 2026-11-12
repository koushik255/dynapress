"""Charts for benchmark summaries (static PNGs via matplotlib).

matplotlib is optional (``uv sync --group plots``). Without it, benchmarks still
write their results and skip the charts; ``plot`` redraws them later.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

INSTALL_HINT = "matplotlib is not installed: run `uv sync --group plots`"

# Categorical slots in fixed order (validated palette, light mode).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CONTEXT_ORDER = ["0", "2", "4", "8", "full"]


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)


def _line(ax, xs, ys, i, label, direct_label=True):
    ax.plot(xs, ys, color=SERIES[i], linewidth=2, marker="o", markersize=7, markeredgecolor="white", markeredgewidth=1.5, label=label)
    if direct_label:
        ax.annotate(label, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK)


def _baselines(ax, row, corpus=None):
    """Lossless references at accuracy 1.0, labels staggered so they never collide."""
    points = [("gzip, per sentence", row["ratio_gzip"], "s", INK, 7), ("zstd, per sentence", row["ratio_zstd"], "D", INK, 19)]
    if corpus:
        points += [
            ("gzip, whole corpus", corpus["original_bytes"] / corpus["gzip_bytes"], "s", "white", -14),
            ("zstd, whole corpus", corpus["original_bytes"] / corpus["zstd_bytes"], "D", "white", 8),
        ]
    for label, x, marker, face, dy in points:
        ax.scatter([x], [1.0], marker=marker, s=55, facecolor=face, edgecolor=INK, zorder=4)
        ax.annotate(label, (x, 1.0), xytext=(6, dy), textcoords="offset points", fontsize=7, color=INK)


def _fixed(summary):
    return [r for r in summary if r["mode"] == "fixed"]


def plot_all(summary: list[dict], out_dir: str | Path, corpus: dict | None = None, device: str = "") -> list[Path]:
    if plt is None:
        print(f"{INSTALL_HINT}, then `python -m neural_compression plot --results {out_dir}` to draw the charts", file=sys.stderr)
        return []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fixed = _fixed(summary)
    retentions = sorted({r["retention"] for r in fixed})
    contexts = [c for c in CONTEXT_ORDER if any(r["context"] == c for r in fixed)]
    contexts += sorted({r["context"] for r in fixed} - set(contexts))
    get = {(r["retention"], r["context"]): r for r in fixed}
    paths = []

    def save(fig, name):
        fig.tight_layout()
        path = out_dir / name
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    # 1. retention vs accuracy, one line per context
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, ctx in enumerate(contexts):
        pts = [(r * 100, get[(r, ctx)]["word_accuracy"]) for r in retentions if (r, ctx) in get]
        if pts:
            _line(ax, *zip(*pts), i, f"context {ctx}", direct_label=False)  # 5 series: legend only
    _style(ax, "Word accuracy vs prefix retention", "characters retained per word (%)", "exact word accuracy")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "retention_vs_accuracy.png")

    # 2 & 3. context vs accuracy / decode time, one line per retention
    for metric, ylabel, name, title in [
        ("word_accuracy", "exact word accuracy", "context_vs_accuracy.png", "Word accuracy vs context window"),
        ("decode_ms_per_word", f"decode time per word (ms{', ' + device if device else ''})", "context_vs_time.png", "Decoding cost vs context window"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for i, r in enumerate(retentions):
            pts = [(j, get[(r, c)][metric]) for j, c in enumerate(contexts) if (r, c) in get]
            if pts:
                _line(ax, *zip(*pts), i, f"{r * 100:.0f}% kept", direct_label=metric == "word_accuracy")
        ax.set_xticks(range(len(contexts)), [c if c != "full" else "full\nsentence" for c in contexts])
        _style(ax, title, "context (neighbouring compressed words)", ylabel)
        if metric == "word_accuracy":
            ax.set_ylim(0, 1.02)
        else:
            ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=8)
        save(fig, name)

    # 4. compression ratio vs accuracy (lossy points + lossless baselines)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, r in enumerate(retentions):
        pts = [(get[(r, c)]["ratio_payload"], get[(r, c)]["word_accuracy"], c) for c in contexts if (r, c) in get]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=60, color=SERIES[i], edgecolor="white", linewidth=1.5, label=f"prefix {r * 100:.0f}%", zorder=3)
        for x, y, c in pts:  # label only the extremes: no context and full sentence
            if c in ("0", "full"):
                ax.annotate("no context" if c == "0" else "full sentence", (x, y), xytext=(7, -3), textcoords="offset points", fontsize=7, color=MUTED)
    _baselines(ax, fixed[0], corpus)
    _style(ax, "Compression ratio vs reconstruction accuracy", "compression ratio (original / stored bytes)", "exact word accuracy")
    ax.set_ylim(0, 1.13)
    ax.set_xlim(right=ax.get_xlim()[1] + 0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    save(fig, "ratio_vs_accuracy.png")

    # 5. adaptive vs fixed (same decoder context)
    adaptive = [r for r in summary if r["mode"].startswith("adaptive")]
    if adaptive:
        ctx = adaptive[0]["context"]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        pts = sorted((get[(r, ctx)]["ratio_payload"], get[(r, ctx)]["sentence_accuracy"], r) for r in retentions if (r, ctx) in get)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=SERIES[0], linewidth=2, marker="o", markersize=7, markeredgecolor="white", label="fixed prefix")
            for x, y, r in pts:
                ax.annotate(f"{r * 100:.0f}%", (x, y), xytext=(5, 4), textcoords="offset points", fontsize=7, color=MUTED)
        for a in adaptive:
            ax.scatter([a["ratio_payload"]], [a["sentence_accuracy"]], s=70, color=SERIES[1], edgecolor="white", linewidth=1.5, zorder=3)
            ax.annotate(a["mode"].replace("adaptive@", "adaptive, target "), (a["ratio_payload"], a["sentence_accuracy"]), xytext=(5, -10), textcoords="offset points", fontsize=7, color=INK)
        ax.scatter([], [], s=70, color=SERIES[1], label="adaptive prefix")
        _baselines(ax, fixed[0])
        _style(ax, f"Adaptive vs fixed prefixes (context {ctx})", "compression ratio (original / stored bytes)", "exact sentence accuracy")
        ax.set_ylim(0, 1.13)
        ax.legend(frameon=False, fontsize=8, loc="lower left")
        save(fig, "adaptive_vs_fixed.png")
    return paths


def plot_from_results(results_dir: str | Path) -> list[Path]:
    if plt is None:
        raise SystemExit(INSTALL_HINT)
    data = json.loads((Path(results_dir) / "results.json").read_text())
    meta = data["meta"]
    return plot_all(data["summary"], results_dir, meta.get("corpus"), device_label(meta))


def device_label(meta: dict) -> str:
    """Short device name from benchmark metadata, e.g. 'NVIDIA A10' or 'cpu'."""
    devices = set(meta.get("devices", {}).values()) or {meta.get("device", "")}
    names = sorted({d.split("(")[-1].rstrip(")") for d in devices if d})
    return " / ".join(names)
