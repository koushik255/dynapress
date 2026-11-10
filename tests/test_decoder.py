"""Decoder guarantees, checked with a tiny random-weight model (fast; needs a one-time download).

A random model reconstructs nothing correctly, which is the point. Whatever the
model prefers, the output must keep every stored prefix and separator and
respect the length range implied by the payload.
"""

import pytest

from neural_compression.compressor import WORD_RE, Compressed, compress, compress_with_lengths
from neural_compression.decoder import Decoder, sentence_units

TINY = "hf-internal-testing/tiny-random-gpt2"

TEXTS = [
    "The programmer discovered an unexpected problem in the application.",
    "Hello, world! Is this (really) working? Yes: 42 times.",
    "First sentence here. Second one follows!\nA new line, too.",
]


@pytest.fixture(scope="module")
def lm():
    from neural_compression.models import load_model

    try:
        return load_model(TINY, threads=2)
    except Exception as exc:  # offline
        pytest.skip(f"cannot load {TINY}: {exc}")


@pytest.mark.parametrize("context", [0, 2, "full"])
@pytest.mark.parametrize("retention", [0.25, 0.5])
def test_output_respects_payload(lm, context, retention):
    payloads = [compress(t, retention).to_bytes() for t in TEXTS]
    results = Decoder(lm, context, beam_size=2).decode_many(payloads)
    for payload, res in zip(payloads, results):
        c = Compressed.from_bytes(payload)
        assert res.failures == 0
        words = WORD_RE.findall(res.text)
        assert len(words) == len(c.prefixes)
        rebuilt = c.separators[0] + "".join(w + s for w, s in zip(words, c.separators[1:]))
        assert rebuilt == res.text  # separators untouched
        for word, slot in zip(words, c.slots()):
            assert word.startswith(slot.prefix)
            assert slot.min_len <= len(word) <= slot.max_len


def test_adaptive_payload_decodes_within_bounds(lm):
    c = compress_with_lengths("the cat sat on the mat", [3, 1, 3, 1, 2, 2], retention_pct=25)
    res = Decoder(lm, "full", beam_size=2).decode(c.to_bytes())
    words = WORD_RE.findall(res.text)
    assert words[0] == "the" and words[2] == "sat"  # complete words are kept verbatim
    assert words[1].startswith("c") and len(words[1]) >= 2


def test_decoding_is_deterministic(lm):
    payload = compress(TEXTS[0], 0.5).to_bytes()
    a = Decoder(lm, 4, beam_size=2).decode(payload).text
    b = Decoder(lm, 4, beam_size=2).decode(payload).text
    assert a == b


def test_batched_equals_single(lm):
    payloads = [compress(t, 0.5).to_bytes() for t in TEXTS]
    decoder = Decoder(lm, "full", beam_size=2)
    batched = [r.text for r in decoder.decode_many(payloads)]
    single = [decoder.decode(p).text for p in payloads]
    assert batched == single


def test_sentence_units_split_on_sentence_ends():
    c = compress(TEXTS[2], 0.5)
    units = sentence_units(c)
    assert [len(u) for u in units] == [3, 3, 4]
