"""Benchmark: run fixed and adaptive configurations, compute metrics, export results.

Every decode goes through ``Compressed.from_bytes(payload)``, so the decoder
only ever sees the serialized payload.

Compression ratio = original UTF-8 bytes / stored bytes (higher is smaller).
gzip/zstd are *lossless*. The prefix scheme is *lossy*, so its ratio only
means something next to its reconstruction accuracy.
"""

from __future__ import annotations

import csv
import gzip
import json
import time
from dataclasses import dataclass
from pathlib import Path

import zstandard

from .adaptive import compress_adaptive_many
from .compressor import WORD_RE, Compressed, compress, compress_with_lengths, split_text
from .decoder import Context, Decoder
from .l2r import LeftToRightDecoder, compress_left_to_right
from .models import LanguageModel

# ---------------------------------------------------------------- metrics


def levenshtein(a, b) -> int:
    """Edit distance between two sequences (strings or lists of words)."""
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def word_accuracy(reference: str, hypothesis: str) -> tuple[int, int]:
    """(exactly correct words, total words), aligned by position.

    The decoder never adds or removes words, so the word sequences line up one-to-one.
    """
    ref, hyp = WORD_RE.findall(reference), WORD_RE.findall(hypothesis)
    return sum(r == h for r, h in zip(ref, hyp)), len(ref)


def char_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    return levenshtein(reference, hypothesis), len(reference)


def word_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    ref = WORD_RE.findall(reference)
    return levenshtein(ref, WORD_RE.findall(hypothesis)), len(ref)


def gzip_size(data: bytes) -> int:
    return len(gzip.compress(data, compresslevel=9, mtime=0))


def zstd_size(data: bytes) -> int:
    return len(zstandard.ZstdCompressor(level=19).compress(data))


# ---------------------------------------------------------------- data


@dataclass
class Sample:
    category: str
    text: str


def load_samples(path: str | Path, limit_per_category: int | None = None) -> list[Sample]:
    """Load ``.jsonl`` ({"category", "text"} per line) or plain text (one sample per line)."""
    path = Path(path)
    samples: list[Sample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if path.suffix == ".jsonl":
            obj = json.loads(line)
            samples.append(Sample(obj.get("category", "default"), obj["text"]))
        else:
            samples.append(Sample("default", line.strip()))
    if limit_per_category:
        seen: dict[str, int] = {}
        kept = []
        for s in samples:
            seen[s.category] = seen.get(s.category, 0) + 1
            if seen[s.category] <= limit_per_category:
                kept.append(s)
        samples = kept
    return samples


# ---------------------------------------------------------------- runs


def _score(sample: Sample, decoded: str) -> dict:
    correct, total = word_accuracy(sample.text, decoded)
    cer_e, cer_n = char_errors(sample.text, decoded)
    wer_e, wer_n = word_errors(sample.text, decoded)
    return {
        "word_correct": correct,
        "word_total": total,
        "sentence_exact": int(decoded == sample.text),
        "char_edits": cer_e,
        "chars": cer_n,
        "word_edits": wer_e,
        "cer": cer_e / max(1, cer_n),
        "wer": wer_e / max(1, wer_n),
    }


def run_fixed(lm: LanguageModel, samples: list[Sample], retention: float, context: Context, beam_size: int = 4, batch_size: int = 16) -> list[dict]:
    rows = []
    payloads, compress_times = [], []
    for s in samples:
        t0 = time.perf_counter()
        payloads.append(compress(s.text, retention).to_bytes())
        compress_times.append(time.perf_counter() - t0)
    decoder = Decoder(lm, context, beam_size=beam_size, batch_size=batch_size)
    results = decoder.decode_many([Compressed.from_bytes(p) for p in payloads])
    for s, payload, ct, res in zip(samples, payloads, compress_times, results):
        c = Compressed.from_bytes(payload)
        rows.append(
            {
                "mode": "fixed",
                "retention": retention,
                "context": str(context),
                "category": s.category,
                "original": s.text,
                "decoded": res.text,
                "original_bytes": len(s.text.encode()),
                "raw_prefix_bytes": sum(len(p.encode()) for p in c.prefixes),
                "payload_bytes": len(payload),
                "payload_zstd_bytes": len(c.to_bytes(use_zstd=True)),
                "gzip_bytes": gzip_size(s.text.encode()),
                "zstd_bytes": zstd_size(s.text.encode()),
                "compress_seconds": ct,
                "decode_seconds": res.seconds,
                "forward_calls": res.stats.forward_calls,
                "sequences": res.stats.sequences,
                "failures": res.failures,
                "rounds": 0,
                **_score(s, res.text),
            }
        )
    return rows


def run_adaptive(
    lm: LanguageModel,
    samples: list[Sample],
    context: Context = "full",
    start_retention: float = 0.25,
    target_accuracy: float = 1.0,
    max_rounds: int = 6,
    beam_size: int = 4,
    batch_size: int = 16,
) -> list[dict]:
    decoder = Decoder(lm, context, beam_size=beam_size, batch_size=batch_size)
    t0 = time.perf_counter()
    results = compress_adaptive_many([s.text for s in samples], decoder, start_retention, target_accuracy, max_rounds=max_rounds)
    total_compress = time.perf_counter() - t0
    # Verify independently: a fresh decode of the final payload bytes.
    verified = decoder.decode_many([Compressed.from_bytes(r.payload) for r in results])
    rows = []
    for s, r, v in zip(samples, results, verified):
        c = Compressed.from_bytes(r.payload)
        rows.append(
            {
                "mode": f"adaptive@{target_accuracy:g}",
                "retention": start_retention,
                "context": str(context),
                "category": s.category,
                "original": s.text,
                "decoded": v.text,
                "original_bytes": len(s.text.encode()),
                "raw_prefix_bytes": sum(len(p.encode()) for p in c.prefixes),
                "payload_bytes": len(r.payload),
                "payload_zstd_bytes": len(c.to_bytes(use_zstd=True)),
                "gzip_bytes": gzip_size(s.text.encode()),
                "zstd_bytes": zstd_size(s.text.encode()),
                "compress_seconds": r.seconds if r.seconds else total_compress / len(samples),
                "decode_seconds": v.seconds,
                "forward_calls": v.stats.forward_calls,
                "sequences": v.stats.sequences,
                "failures": v.failures,
                "rounds": r.rounds,
                "verified_matches_search": int(v.text == r.decoded_text),
                "payload_hex": r.payload.hex(),
                **_score(s, v.text),
            }
        )
    return rows


def run_left_to_right(
    lm: LanguageModel,
    samples: list[Sample],
    lookback: int = 1,
    start_retention: float = 0.25,
    beam_size: int = 4,
    batch_size: int = 16,
) -> list[dict]:
    """Left-to-right adaptive compression of the samples as one document (joined by newlines).

    With ``lookback`` > 0 a sentence is restored with earlier sentences in its prompt, so the
    per-sample payloads in the rows only decode as parts of the merged document
    (``merge_payloads``). That merged payload is the one compressed and verified here.
    """
    decoder = LeftToRightDecoder(lm, lookback, beam_size=beam_size, batch_size=batch_size)
    result = compress_left_to_right("\n".join(s.text for s in samples), decoder, start_retention)
    # Verify independently: a fresh decode of the final payload bytes.
    verified = decoder.decode(Compressed.from_bytes(result.payload))
    pct = Compressed.from_bytes(result.payload).retention_pct
    total_words = max(1, len(result.lengths))
    rows, parts, pos = [], [], 0
    for s in samples:
        separators, words = split_text(s.text)
        span = slice(pos, pos + len(words))
        pos += len(words)
        c = compress_with_lengths(s.text, result.lengths[span], pct)
        parts.append(c)
        decoded = separators[0] + "".join(w + sep for w, sep in zip(verified.words[span], separators[1:]))
        share = len(words) / total_words  # attribute document cost proportionally
        rows.append(
            {
                "mode": "adaptive-l2r@1",
                "retention": start_retention,
                "context": f"l2r-{lookback}",
                "category": s.category,
                "original": s.text,
                "decoded": decoded,
                "original_bytes": len(s.text.encode()),
                "raw_prefix_bytes": sum(len(p.encode()) for p in c.prefixes),
                "payload_bytes": len(c.to_bytes()),
                "payload_zstd_bytes": len(c.to_bytes(use_zstd=True)),
                "gzip_bytes": gzip_size(s.text.encode()),
                "zstd_bytes": zstd_size(s.text.encode()),
                "compress_seconds": result.seconds * share,
                "search_seconds": result.search_seconds * share,
                "decode_seconds": verified.seconds * share,
                "forward_calls": verified.stats.forward_calls * share,
                "sequences": verified.stats.sequences * share,
                "failures": verified.failures * share,
                "rounds": max(result.tries[span], default=0),  # searches spent on the sample's hardest word
                "repairs": sum(span.start <= i < span.stop for i in result.repairs),
                "verify_decodes": result.verify_decodes,
                "verified_matches_search": int(verified.text == result.decoded_text),
                "payload_hex": c.to_bytes().hex(),
                **_score(s, decoded),
            }
        )
    assert merge_payloads(parts).to_bytes() == result.payload, "per-sample payloads must merge into the verified document"
    return rows


# ---------------------------------------------------------------- summaries


def summarize(rows: list[dict], by_category: bool = False) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r["mode"], r["retention"], r["context"]) + ((r["category"],) if by_category else ())
        groups.setdefault(key, []).append(r)
    out = []
    for key, rs in groups.items():
        total = lambda k: sum(r[k] for r in rs)  # noqa: E731
        orig = total("original_bytes")
        out.append(
            {
                "mode": key[0],
                "retention": key[1],
                "context": key[2],
                **({"category": key[3]} if by_category else {}),
                "samples": len(rs),
                "word_accuracy": total("word_correct") / max(1, total("word_total")),
                "sentence_accuracy": total("sentence_exact") / len(rs),
                "cer": total("char_edits") / max(1, total("chars")),
                "wer": total("word_edits") / max(1, total("word_total")),
                "ratio_payload": orig / total("payload_bytes"),
                "ratio_payload_zstd": orig / total("payload_zstd_bytes"),
                "ratio_raw_prefix": orig / max(1, total("raw_prefix_bytes")),
                "ratio_gzip": orig / total("gzip_bytes"),
                "ratio_zstd": orig / total("zstd_bytes"),
                "compress_seconds": total("compress_seconds"),
                "decode_seconds": total("decode_seconds"),
                "decode_ms_per_word": 1000 * total("decode_seconds") / max(1, total("word_total")),
                "forward_calls": total("forward_calls"),
                "sequences": total("sequences"),
                "failures": total("failures"),
                "mean_rounds": total("rounds") / len(rs),
            }
        )
    return out


def corpus_baselines(samples: list[Sample], retentions: list[float]) -> dict:
    """Whole-corpus sizes: removes per-sentence header overhead for every method."""
    corpus = "\n".join(s.text for s in samples)
    data = corpus.encode()
    out = {"original_bytes": len(data), "gzip_bytes": gzip_size(data), "zstd_bytes": zstd_size(data)}
    for r in retentions:
        c = compress(corpus, r)
        out[f"prefix{int(r * 100)}_payload_bytes"] = len(c.to_bytes())
        out[f"prefix{int(r * 100)}_payload_zstd_bytes"] = len(c.to_bytes(use_zstd=True))
    return out


def merge_payloads(payloads: list[Compressed], joiner: str = "\n") -> Compressed:
    """Concatenate per-sentence payloads into one document payload (sentences joined by ``joiner``).

    Sentences are decoded independently (see ``decoder.sentence_units``), so the merged
    payload decodes to the same sentences; only the per-file overhead disappears.
    """
    first = payloads[0]
    separators, prefixes, complete = [first.separators[0]], [], []
    for i, c in enumerate(payloads):
        separators[-1] += c.separators[0] if i else ""
        separators += c.separators[1:]
        prefixes += c.prefixes
        complete += c.complete
        if i + 1 < len(payloads):
            separators[-1] += joiner
    return Compressed(separators, prefixes, first.retention_pct, first.adaptive, first.verbatim_numbers, complete)


def adaptive_corpus_sizes(rows: list[dict]) -> dict:
    """Whole-corpus payload size for each adaptive configuration (from the verified per-sentence payloads)."""
    out = {}
    for mode in sorted({r["mode"] for r in rows if r["mode"].startswith("adaptive") and r.get("payload_hex")}):
        merged = merge_payloads([Compressed.from_bytes(bytes.fromhex(r["payload_hex"])) for r in rows if r["mode"] == mode])
        out[f"{mode}_payload_bytes"] = len(merged.to_bytes())
        out[f"{mode}_payload_zstd_bytes"] = len(merged.to_bytes(use_zstd=True))
    return out


def export(rows: list[dict], out_dir: str | Path, meta: dict) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta.setdefault("corpus", {}).update(adaptive_corpus_sizes(rows))
    paths = {
        "rows": out_dir / "rows.csv",
        "summary": out_dir / "summary.csv",
        "by_category": out_dir / "summary_by_category.csv",
        "json": out_dir / "results.json",
    }
    _write_csv(paths["rows"], rows)
    summary, by_cat = summarize(rows), summarize(rows, by_category=True)
    _write_csv(paths["summary"], summary)
    _write_csv(paths["by_category"], by_cat)
    paths["json"].write_text(json.dumps({"meta": meta, "summary": summary, "by_category": by_cat}, indent=2))
    return paths


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
