"""Left-to-right compression: the compressor's one-word checks must predict the real decode.

The first tests replace the model search with a small rule-based guesser, so the expected
prefix lengths can be worked out by hand. The last ones run the real search with a tiny
random-weight model (see test_decoder.py).
"""

from types import SimpleNamespace

import pytest

from neural_compression.compressor import WORD_RE, Compressed, compress
from neural_compression.l2r import LeftToRightDecoder, compress_left_to_right
from neural_compression.models import InferenceStats

TEXT = "The cat sat. The cow ran."
COMMON = ["The", "cow", "sat", "ran", "cat"]  # guessed in this order...
AFTER = {"The": ["cat"], "sat": ["Then"]}  # ...unless the previous word suggests something first


class RuleDecoder(LeftToRightDecoder):
    """Guesses from the previous word in the prompt and the stored prefix. No model."""

    def __init__(self, lookback):
        super().__init__(SimpleNamespace(stats=InferenceStats()), lookback)

    def guess(self, request):
        seen = WORD_RE.findall(request.prompt)
        (prefix,) = WORD_RE.findall(request.template[0])
        gap = request.template[1]
        for word in AFTER.get(seen[-1] if seen else None, []) + COMMON:
            if word.startswith(prefix) and gap.min_chars <= len(word) - len(prefix) <= gap.max_chars:
                return word
        return None

    def _search(self, requests):
        return [self.guess(r) for r in requests]


class BatchSensitiveDecoder(RuleDecoder):
    """Gets "cat" wrong, but only when it is searched alone, as in the real decode of a single chain."""

    def _search(self, requests):
        words = super()._search(requests)
        if len(requests) == 1 and requests[0].template[0].endswith(" c"):
            return ["cab"]
        return words


# "cow" needs 2 letters: after "The" the guesser prefers "cat". The second "The" follows "sat" only
# when the previous sentence is in the prompt, and then the guesser insists on "Then".
@pytest.mark.parametrize("lookback, lengths", [(0, [1, 1, 1, 1, 2, 1]), (1, [1, 1, 1, 3, 2, 1])])
def test_shortest_prefixes_round_trip_without_repairs(lookback, lengths):
    result = compress_left_to_right(TEXT, RuleDecoder(lookback))
    assert result.lengths == lengths
    # The decode guesses "cat" only if it is shown the restored "The", not the stored "T".
    assert RuleDecoder(lookback).decode(result.payload).text == TEXT
    assert result.repairs == [] and result.verify_decodes == 1


def test_word_that_decodes_differently_in_the_full_decode_is_lengthened():
    result = compress_left_to_right(TEXT, BatchSensitiveDecoder(1))
    assert result.repairs == [1]
    assert result.lengths == [1, 2, 1, 3, 2, 1]
    assert BatchSensitiveDecoder(1).decode(result.payload).text == TEXT


TINY = "hf-internal-testing/tiny-random-gpt2"

TEXTS = [
    "The programmer discovered an unexpected problem in the application.",
    "Hello, world! Is this (really) working? Yes: 42 times.",
    'First sentence here. "Second one follows!"\nA new line, too.',
]


@pytest.fixture(scope="module")
def lm():
    from neural_compression.models import load_model

    try:
        return load_model(TINY, threads=2)
    except Exception as exc:  # offline
        pytest.skip(f"cannot load {TINY}: {exc}")


@pytest.mark.parametrize("lookback", [0, 1])
def test_output_respects_payload(lm, lookback):
    payloads = [compress(t, 0.5).to_bytes() for t in TEXTS]
    results = LeftToRightDecoder(lm, lookback, beam_size=2).decode_many(payloads)
    for payload, res in zip(payloads, results):
        c = Compressed.from_bytes(payload)
        assert res.failures == 0
        words = WORD_RE.findall(res.text)
        assert len(words) == len(c.prefixes)
        assert c.separators[0] + "".join(w + s for w, s in zip(words, c.separators[1:])) == res.text  # separators untouched
        for word, slot in zip(words, c.slots()):
            assert word.startswith(slot.prefix)
            assert slot.min_len <= len(word) <= slot.max_len


@pytest.mark.parametrize("lookback", [0, 1])
def test_round_trip_with_real_search(lm, lookback):
    text = "\n".join(TEXTS)
    decoder = LeftToRightDecoder(lm, lookback, beam_size=2)
    result = compress_left_to_right(text, decoder)
    assert LeftToRightDecoder(lm, lookback, beam_size=2).decode(result.payload).text == text
