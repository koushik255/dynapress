"""Single-pass student decoder guarantees, with a random-weight model (no download, fast).

A random model restores little correctly. Whatever it prefers, the output must
keep every stored prefix and separator, respect each slot's length range, and
adaptive compression with it must still reconstruct exactly.
"""

import torch

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


def test_adaptive_is_exact_and_merges():
    dec = decoder()
    results = compress_adaptive_many(TEXTS, dec)
    assert [r.decoded_text for r in results] == TEXTS
    assert [r.text for r in dec.decode_many([r.payload for r in results])] == TEXTS
    merged = merge_payloads([Compressed.from_bytes(r.payload) for r in results])
    assert dec.decode(merged.to_bytes()).text == "\n".join(TEXTS)
