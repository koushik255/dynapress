"""Single-pass student decoder guarantees, with a random-weight model (no download, fast).

A random model restores little correctly. Whatever it prefers, the output must
keep every stored prefix and separator, respect each slot's length range, and
adaptive compression with it must still reconstruct exactly.
"""

import json
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")  # the student needs the ``training`` dependency group

from neural_compression.adaptive import compress_adaptive_many
from neural_compression.benchmark import merge_payloads
from neural_compression.compressor import WORD_RE, Compressed, compress
from neural_compression.decoder import sentence_units
from neural_compression.student import CharNgram, Config, StudentDecoder, StudentModel, Vocab, unit_features

TEXTS = [
    "The programmer discovered an unexpected problem in the application.",
    "Hello, world! Is THIS (really) working? Yes: 42 times, Zyxwvutsrqponmlkjihgfedcba.",
    "First sentence here. Second one follows!\nA new line, too.",
]


def decoder() -> StudentDecoder:
    torch.manual_seed(0)
    words = sorted({w.lower() for t in TEXTS for w in WORD_RE.findall(t)} | {"probably", "the", "then", "there", "pro"})
    vocab = Vocab(words, list(range(1, len(words) + 1)))
    model = StudentModel(Config(len(vocab), dim=32, layers=1, heads=2, ff=64), vocab.logprior)
    return StudentDecoder(model, vocab, CharNgram(vocab.words, vocab.counts.tolist()))


def test_char_ngram_keeps_predictions_and_ties_across_repeated_calls():
    model = CharNgram(["alpha", "a1", "éclair", "東京", "ßeta"], [91, 2, 7, 3, 1])
    # Captured from the original uncached decoder; a wider beam must not turn into greedy search.
    cases = [("", 1, 4, 1, "a"), ("", 1, 4, 8, "東京"), ("a", 1, 4, 8, "lpha"),
             ("écl", 3, 3, 8, "air"), ("東", 1, 1, 8, "京"), ("İ", 1, 2, 3, "a"), ("a", 0, 0, 8, "")]
    for _ in range(2):
        for prefix, lo, hi, beam, expected in cases:
            assert model.complete(prefix, lo, hi, beam) == expected
    tied = CharNgram(["a", "b"], [1, 1])
    opposite = CharNgram(["a", "b"], [1, 19])
    for _ in range(2):
        assert tied.complete("", 1, 1) == "a"
        assert opposite.complete("", 1, 1) == "b"


def test_output_respects_payload():
    dec = decoder()
    for retention in (0.25, 0.5):
        payloads = [compress(t, retention).to_bytes() for t in TEXTS]
        for payload, res in zip(payloads, dec.decode_many(payloads)):
            c = Compressed.from_bytes(payload)
            assert WORD_RE.findall(res.text) == res.words
            parts = [c.separators[0]]
            for word, sep in zip(res.words, c.separators[1:]):
                parts += [word, sep]
            assert res.text == "".join(parts)
            for word, slot in zip(res.words, c.slots()):
                assert word.startswith(slot.prefix)
                assert slot.min_len <= len(word) <= slot.max_len


def test_sentence_features_do_not_depend_on_neighbours():
    """A sentence must decode the same alone and inside a merged document."""
    vocab = decoder().vocab
    texts = ["“Yes,” she said.", "It was late.", "“Go!”"]
    docs = [compress(t, 0.5) for t in texts]
    merged = merge_payloads(docs)
    units = sentence_units(merged)
    assert len(units) == len(docs)
    for doc, unit in zip(docs, units):
        alone = unit_features(doc, doc.slots(), range(len(doc.prefixes)), vocab)
        inside = unit_features(merged, merged.slots(), unit, vocab)
        assert {k: v for k, v in alone.items() if k != "cand"} == {k: v for k, v in inside.items() if k != "cand"}


@pytest.mark.parametrize("options", [{}, {"step": 2}, {"step": 2, "max_rounds": 3}, {"start_retention": 0.5}])
def test_adaptive_is_exact_and_merges(options):
    dec = decoder()
    results = compress_adaptive_many(TEXTS, dec, **options)
    assert [r.decoded_text for r in results] == TEXTS
    assert [r.text for r in dec.decode_many([r.payload for r in results])] == TEXTS
    merged = merge_payloads([Compressed.from_bytes(r.payload) for r in results])
    assert dec.decode(merged.to_bytes()).text == "\n".join(TEXTS)


def test_evaluation_cli_uses_adaptive_options(tmp_path):
    model, out = tmp_path / "model", tmp_path / "out"
    decoder().save(model)
    text = "Q" + "q" * 99  # Too long for any suffix prediction; forces full-word fallback.
    (tmp_path / "validation.jsonl").write_text(json.dumps({"book_id": "sample", "text": text}) + "\n")
    subprocess.run(
        [sys.executable, "-m", "neural_compression.student_train", "evaluate", "--model", str(model),
         "--data", str(tmp_path), "--out", str(out), "--threads", "1",
         "--start-retention", "0.4", "--step", "2", "--max-rounds", "1"],
        check=True, capture_output=True, text=True,
    )
    summary = json.loads((out / "summary.json").read_text())
    assert summary["adaptive_options"] == {"start_retention": 0.4, "step": 2, "max_rounds": 1}
    assert summary["adaptive100"]["mean_rounds"] == 2  # Default round limit would need seven.
    row = json.loads((out / "adaptive_rows.jsonl").read_text())
    assert StudentDecoder.load(model).decode(bytes.fromhex(row["payload_hex"])).text == text


def test_continuation_keeps_checkpoint_and_vocabulary_on_tie(tmp_path):
    """New text must not rebuild the vocabulary or replace an unimproved checkpoint."""
    source = tmp_path / "source"
    decoder().save(source)
    initial = StudentDecoder.load(source)
    data = tmp_path / "data"
    data.mkdir()
    for split in ("train", "validation"):
        texts = TEXTS if split == "validation" else ["Unlistedwords appear alongside the programmer and the application."]
        (data / f"{split}.jsonl").write_text("".join(json.dumps({"book_id": "sample", "text": t}) + "\n" for t in texts))
    out = tmp_path / "continued"
    subprocess.run(
        [sys.executable, "-m", "neural_compression.student_train", "train", "--data", str(data), "--out", str(out),
         "--init-model", str(source), "--steps", "1", "--lr", "0", "--workers", "0", "--threads", "1", "--batch-size", "4",
         "--dev-start", "0", "--dev-per-book", "3", "--select-by", "adaptive-bytes"],
        check=True, capture_output=True, text=True,
    )
    continued = StudentDecoder.load(out / "model")
    assert continued.model.cfg == initial.model.cfg
    assert (out / "model/vocab.tsv").read_bytes() == (source / "vocab.tsv").read_bytes()
    for k, value in initial.model.state_dict().items():
        assert torch.equal(value, continued.model.state_dict()[k])
    assert json.loads((out / "model/config.json").read_text())["step"] == 0
    assert continued.decode(compress(TEXTS[0], 0.5)).text == initial.decode(compress(TEXTS[0], 0.5)).text
