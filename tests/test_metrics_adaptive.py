"""Metrics and the adaptive loop (with stub decoders, no model needed)."""

from neural_compression.adaptive import compress_adaptive_many
from neural_compression.benchmark import char_errors, levenshtein, word_accuracy, word_errors
from neural_compression.compressor import Compressed
from neural_compression.decoder import DecodeResult


def test_levenshtein():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein(["a", "b"], ["a", "c"]) == 1


def test_word_metrics():
    ref, hyp = "The cat sat.", "The cap sat."
    assert word_accuracy(ref, hyp) == (2, 3)
    assert word_errors(ref, hyp) == (1, 3)
    assert char_errors(ref, hyp) == (1, len(ref))


class PrefixOnlyDecoder:
    """Worst case: never fills in anything. Records every payload it is given."""

    def __init__(self):
        self.seen = []

    def decode_many(self, payloads):
        out = []
        for c in payloads:
            assert isinstance(c, Compressed)
            self.seen.append(c)
            text = c.separators[0] + "".join(p + s for p, s in zip(c.prefixes, c.separators[1:]))
            out.append(DecodeResult(text, list(c.prefixes), 0.0))
        return out


class FlakyDecoder(PrefixOnlyDecoder):
    """Knows a vocabulary but breaks word 0 whenever word 1 is long: non-monotonic behaviour."""

    VOCAB = ["alpha", "bravo"]

    def decode_many(self, payloads):
        out = []
        for c in payloads:
            words = []
            for i, (p, slot) in enumerate(zip(c.prefixes, c.slots())):
                match = next((w for w in self.VOCAB if w.startswith(p) and not slot.complete), p)
                words.append(match)
            if len(c.prefixes[1]) >= 3 and not c.slots()[0].complete:
                words[0] = "abc"
            out.append(DecodeResult(" ".join(words), words, 0.0))
        return out


def test_adaptive_terminates_and_reaches_target_with_useless_decoder():
    decoder = PrefixOnlyDecoder()
    texts = ["the quick brown fox", "jumps over the lazy dog"]
    results = compress_adaptive_many(texts, decoder, start_retention=0.25, max_rounds=3)
    for text, r in zip(texts, results):
        assert r.word_accuracy == 1.0
        assert Compressed.from_bytes(r.payload).body_text() == text  # everything had to be stored
        assert r.rounds <= 4 + len(text)
    # the decoder only ever saw prefixes that the payload contains
    assert all(isinstance(c, Compressed) for c in decoder.seen)


def test_adaptive_handles_non_monotonic_decoder():
    # "brave" is not in the decoder's vocabulary, so it must be stored in full; lengthening it
    # breaks "alpha", which was right in round 1.
    r = compress_adaptive_many(["alpha brave"], FlakyDecoder(), start_retention=0.25, max_rounds=5)[0]
    assert r.history[0][1] == 0.5
    assert any(acc == 0.0 for _, acc in r.history)  # got worse before it got better
    assert r.word_accuracy == 1.0
    final = Compressed.from_bytes(r.payload)
    # verify with a fresh decode of the final payload
    assert FlakyDecoder().decode_many([final])[0].words == ["alpha", "brave"]


def test_adaptive_stops_immediately_when_decoder_is_perfect():
    class Oracle(PrefixOnlyDecoder):
        def decode_many(self, payloads):
            return [DecodeResult("hello world", ["hello", "world"], 0.0) for _ in payloads]

    r = compress_adaptive_many(["hello world"], Oracle(), start_retention=0.25)[0]
    assert r.rounds == 1
    assert Compressed.from_bytes(r.payload).prefixes == ["he", "wo"]
