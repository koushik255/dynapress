"""Run log for the dev loop: what each run scored, and what changed since the last one.

    <out>/runs.jsonl       one JSON line per run: label, commit, scores, sizes, time
    <out>/runs/<id>.json   that run's per-sentence rows and its uncommitted code diff

A run is compared with the latest earlier run on the same sentences
(``samples_sha``), so a ``--limit`` smoke run is never compared with a full run.

Sizes are whole-corpus bytes (sentences joined by newlines), so per-sentence
headers do not distort the comparison with zstd.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .benchmark import Sample, adaptive_corpus_sizes, corpus_baselines, summarize
from .compressor import WORD_RE


def samples_sha(samples: list[Sample]) -> str:
    return hashlib.sha256("\n".join(s.text for s in samples).encode()).hexdigest()[:12]


def make_entry(rows: list[dict], samples: list[Sample], info: dict) -> dict:
    """The log line for one run. ``info`` holds what the caller knows (id, label, commit, model, ...)."""
    corpus = corpus_baselines(samples, [])
    entry = {
        **info,
        "samples": len(samples),
        "samples_sha": samples_sha(samples),
        "words": sum(len(WORD_RE.findall(s.text)) for s in samples),
        "original_bytes": corpus["original_bytes"],
        "gzip_bytes": corpus["gzip_bytes"],
        "zstd_bytes": corpus["zstd_bytes"],
    }
    sizes = adaptive_corpus_sizes(rows)
    for s in summarize(rows):
        if s["mode"] == "fixed":
            entry.update(
                fixed_retention=s["retention"],
                fixed_word_accuracy=s["word_accuracy"],
                fixed_sentence_accuracy=s["sentence_accuracy"],
                fixed_cer=s["cer"],
                fixed_decode_seconds=s["decode_seconds"],
            )
        else:
            entry.update(
                adaptive_mode=s["mode"],
                adaptive_context=s["context"],
                adaptive_word_accuracy=s["word_accuracy"],
                adaptive_sentence_accuracy=s["sentence_accuracy"],
                adaptive_payload_bytes=sizes[f"{s['mode']}_payload_bytes"],
                adaptive_payload_zstd_bytes=sizes[f"{s['mode']}_payload_zstd_bytes"],
                adaptive_mean_rounds=s["mean_rounds"],
                adaptive_compress_seconds=s["compress_seconds"],
                adaptive_decode_seconds=s["decode_seconds"],
            )
    return entry


def load_entries(out_dir: str | Path) -> list[dict]:
    log = Path(out_dir) / "runs.jsonl"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_rows(out_dir: str | Path, run_id: str) -> list[dict]:
    return json.loads((Path(out_dir) / "runs" / f"{run_id}.json").read_text(encoding="utf-8"))["rows"]


def previous_entry(out_dir: str | Path, sha: str) -> dict | None:
    """The latest logged run on the same sentences, if any."""
    same = [e for e in load_entries(out_dir) if e["samples_sha"] == sha]
    return same[-1] if same else None


def append_run(out_dir: str | Path, entry: dict, rows: list[dict], code_diff: str = "") -> Path:
    out_dir = Path(out_dir)
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    (out_dir / "runs" / f"{entry['id']}.json").write_text(json.dumps({"rows": rows, "code_diff": code_diff}), encoding="utf-8")
    with (out_dir / "runs.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return out_dir / "runs.jsonl"


def flipped_words(old_rows: list[dict], new_rows: list[dict]) -> tuple[list[tuple], list[tuple]]:
    """Words whose fixed-mode result changed between two runs on the same sentences.

    Returns (fixed, broke), each a list of (original word, old output, new output).
    """
    fixed, broke = [], []
    old_fixed = [r for r in old_rows if r["mode"] == "fixed"]
    new_fixed = [r for r in new_rows if r["mode"] == "fixed"]
    for old, new in zip(old_fixed, new_fixed):
        words = zip(WORD_RE.findall(new["original"]), WORD_RE.findall(old["decoded"]), WORD_RE.findall(new["decoded"]))
        for ref, before, after in words:
            if before != ref and after == ref:
                fixed.append((ref, before, after))
            elif before == ref and after != ref:
                broke.append((ref, before, after))
    return fixed, broke


def format_report(entry: dict, prev: dict | None, flips: tuple[list, list] | None, max_words: int = 15) -> str:
    """Scores for this run next to the previous run on the same sentences."""
    state = entry["commit"] + ("+uncommitted" if entry["dirty"] else "")
    lines = [
        f"run {entry['id']}  {entry['label'] or '(no label)'}  [{state}, {entry['device']}]",
        f"{entry['samples']} sentences, {entry['words']} words, {entry['original_bytes']:,} bytes, {entry['wall_seconds']:.0f} s",
    ]
    if prev:
        lines.append(f"previous: run {prev['id']}  {prev['label'] or '(no label)'}")
    lines.append("")

    def row(name: str, key: str, fmt: str, scale: int = 1) -> None:
        if key not in entry:
            return
        text = f"  {name:<34}{format(entry[key] * scale, fmt):>9}"
        if prev and key in prev:
            text += f"   was {format(prev[key] * scale, fmt):>8}   {format((entry[key] - prev[key]) * scale, '+' + fmt)}"
        lines.append(text)

    row(f"fixed {entry.get('fixed_retention', 0):.0%}: word accuracy %", "fixed_word_accuracy", ".1f", 100)
    row("          exact sentences %", "fixed_sentence_accuracy", ".1f", 100)
    row("          char error rate %", "fixed_cer", ".1f", 100)
    if "adaptive_mode" in entry:
        lines.append(f"  {entry['adaptive_mode']}, context {entry.get('adaptive_context', 'full')}")
        row("          bytes (zstd body)", "adaptive_payload_zstd_bytes", ",d")
        row("          bytes (plain body)", "adaptive_payload_bytes", ",d")
        row("          exact sentences %", "adaptive_sentence_accuracy", ".1f", 100)
        row("          decoder rounds", "adaptive_mean_rounds", ".2f")
        row("          compress seconds", "adaptive_compress_seconds", ".1f")
        row("          decode seconds", "adaptive_decode_seconds", ".1f")
        best = min(entry["adaptive_payload_zstd_bytes"], entry["adaptive_payload_bytes"])
        lines.append(f"  zstd -19 of the same text: {entry['zstd_bytes']:,} bytes, so adaptive is {best / entry['zstd_bytes']:.1%} of zstd")
    if flips:
        for title, words in zip(("now right", "now wrong"), flips):
            lines.append("")
            lines.append(f"fixed mode, {title}: {len(words)} words")
            lines += [f"  {ref}: {before} -> {after}" for ref, before, after in words[:max_words]]
            if len(words) > max_words:
                lines.append(f"  ... and {len(words) - max_words} more")
    return "\n".join(lines)


def format_history(entries: list[dict]) -> str:
    """One line per logged run, oldest first."""
    lines = [f"{'run':<16}{'sent.':>6}{'fixed word %':>14}{'adaptive bytes':>16}{'% of zstd':>11}{'seconds':>9}  commit    label"]
    for e in entries:
        accuracy = f"{e['fixed_word_accuracy']:.1%}" if "fixed_word_accuracy" in e else "-"
        size = share = "-"
        if "adaptive_mode" in e:
            best = min(e["adaptive_payload_zstd_bytes"], e["adaptive_payload_bytes"])
            size, share = f"{best:,}", f"{best / e['zstd_bytes']:.1%}"
        state = e["commit"] + ("+" if e["dirty"] else "")
        lines.append(f"{e['id']:<16}{e['samples']:>6}{accuracy:>14}{size:>16}{share:>11}{e['wall_seconds']:>9.0f}  {state:<9} {e['label']}")
    return "\n".join(lines)
