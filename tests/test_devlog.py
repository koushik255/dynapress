"""The dev-loop run log: entries, lookup of the previous run, and word flips (no model needed)."""

from neural_compression import devlog
from neural_compression.benchmark import Sample
from neural_compression.compressor import compress

SAMPLES = [Sample("default", "The cat sat on the mat."), Sample("default", "A dog barked loudly.")]
INFO = {"label": "", "commit": "abc1234", "dirty": False, "device": "cpu", "wall_seconds": 1.0}


def _rows(decoded: list[str], adaptive_lengths: list[int] | None = None) -> list[dict]:
    rows = []
    for s, d in zip(SAMPLES, decoded):
        words = s.text.rstrip(".").split()
        fixed = {"mode": "fixed", "retention": 0.5, "context": "full", "category": s.category, "original": s.text, "decoded": d}
        fixed.update(word_correct=sum(a == b for a, b in zip(words, d.rstrip(".").split())), word_total=len(words))
        fixed.update(sentence_exact=int(d == s.text), char_edits=0, chars=len(s.text), word_edits=0)
        sizes = dict(original_bytes=len(s.text), raw_prefix_bytes=1, payload_bytes=1, payload_zstd_bytes=1, gzip_bytes=1, zstd_bytes=1)
        cost = dict(compress_seconds=0.0, decode_seconds=0.0, forward_calls=0, sequences=0, failures=0, rounds=0)
        rows.append({**fixed, **sizes, **cost})
        if adaptive_lengths:
            payload = compress(s.text, 1.0).to_bytes()
            rows.append({**rows[-1], "mode": "adaptive@1", "retention": 0.25, "decoded": s.text, "payload_hex": payload.hex()})
    return rows


def test_run_log_compares_with_previous_run_on_same_sentences(tmp_path):
    first = _rows(["The cap sat on the mat.", "A dog barked loudly."])
    second = _rows(["The cat sat on the mat.", "A dig barked loudly."])
    e1 = devlog.make_entry(first, SAMPLES, {**INFO, "id": "run1"})
    devlog.append_run(tmp_path, e1, first)
    # A run on other sentences (e.g. --limit) must not become the comparison point.
    other = devlog.make_entry(first[:1], SAMPLES[:1], {**INFO, "id": "smoke"})
    devlog.append_run(tmp_path, other, first[:1])

    e2 = devlog.make_entry(second, SAMPLES, {**INFO, "id": "run2"})
    prev = devlog.previous_entry(tmp_path, e2["samples_sha"])
    assert prev["id"] == "run1"
    assert e1["fixed_word_accuracy"] == 9 / 10 and e1["words"] == 10

    fixed, broke = devlog.flipped_words(devlog.load_rows(tmp_path, prev["id"]), second)
    assert fixed == [("cat", "cap", "cat")]
    assert broke == [("dog", "dog", "dig")]
    report = devlog.format_report(e2, prev, (fixed, broke))
    assert "dog: dog -> dig" in report and "+0.0" in report
    devlog.append_run(tmp_path, e2, second)
    assert [e["id"] for e in devlog.load_entries(tmp_path)] == ["run1", "smoke", "run2"]


def test_entry_reports_adaptive_corpus_bytes_next_to_zstd():
    entry = devlog.make_entry(_rows([s.text for s in SAMPLES], adaptive_lengths=[1]), SAMPLES, {**INFO, "id": "run"})
    corpus = "\n".join(s.text for s in SAMPLES)
    # Whole-corpus payload: one 3-byte header plus the text, not one header per sentence.
    assert entry["adaptive_payload_bytes"] == len(compress(corpus, 1.0).to_bytes())
    assert entry["original_bytes"] == len(corpus)
    assert "of zstd" in devlog.format_report(entry, None, None)
    assert "run" in devlog.format_history([entry])
