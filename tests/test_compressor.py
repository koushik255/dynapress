import pytest

from neural_compression.compressor import (
    HEADER_BYTES,
    MAX_SUFFIX,
    Compressed,
    compress,
    compress_with_lengths,
    feasible_lengths,
    kept_chars,
    split_text,
)

TEXTS = [
    "The programmer discovered an unexpected problem in the application.",
    "  Leading and trailing whitespace\n\nwith blank lines.  ",
    "Punctuation!? Yes: (parentheses), \"quotes\", and--dashes... done",
    "Numbers like 3.14, 2024 and v2 stay; snake_case_names too.",
    "Unicode: café, naïve, Zürich — and emoji 🙂 work.",
    "",
    "...",
    "a",
]


@pytest.mark.parametrize("text", TEXTS)
def test_split_roundtrip(text):
    seps, words = split_text(text)
    assert len(seps) == len(words) + 1
    rebuilt = seps[0] + "".join(w + s for w, s in zip(words, seps[1:]))
    assert rebuilt == text


def test_kept_chars_rounds_up_with_minimum_one():
    assert kept_chars(10, 50) == 5
    assert kept_chars(3, 50) == 2  # ceil(1.5)
    assert kept_chars(1, 25) == 1
    assert kept_chars(2, 25) == 1  # at least one character
    assert kept_chars(7, 75) == 6  # ceil(5.25)
    assert kept_chars(4, 100) == 4


@pytest.mark.parametrize("pct", [10, 25, 33, 50, 75, 90, 100])
def test_feasible_lengths_are_exact(pct):
    """The decoder's length range contains the true length and nothing that would truncate differently."""
    for n in range(1, 60):
        k = kept_chars(n, pct)
        lengths = feasible_lengths(k, pct)
        assert n in lengths
        assert all(kept_chars(m, pct) == k for m in lengths)
        assert kept_chars(lengths.start - 1, pct) != k or lengths.start - 1 < 1
        assert kept_chars(lengths.stop, pct) != k


@pytest.mark.parametrize("text", TEXTS)
@pytest.mark.parametrize("retention", [0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("use_zstd", [False, True])
def test_serialization_roundtrip(text, retention, use_zstd):
    c = compress(text, retention)
    parsed = Compressed.from_bytes(c.to_bytes(use_zstd=use_zstd))
    assert parsed == c
    assert parsed.slots() == c.slots()


@pytest.mark.parametrize("text", TEXTS)
@pytest.mark.parametrize("retention", [0.25, 0.5, 0.75])
def test_slots_are_consistent_with_original(text, retention):
    """Every original word fits the slot the decoder derives from the payload alone."""
    _, words = split_text(text)
    slots = Compressed.from_bytes(compress(text, retention).to_bytes()).slots()
    for word, slot in zip(words, slots):
        assert word.startswith(slot.prefix)
        assert slot.min_len <= len(word) <= slot.max_len


def test_example_sentence():
    c = compress("The programmer discovered an unexpected problem in the application.", 0.5)
    assert c.body_text() == "Th progr disco a unexp prob i th applic."
    assert c.to_bytes()[:HEADER_BYTES] == bytes([1, 2, 50])


def test_compression_is_deterministic():
    text = TEXTS[0] * 3
    assert compress(text, 0.5).to_bytes() == compress(text, 0.5).to_bytes()
    assert compress(text, 0.5).to_bytes(True) == compress(text, 0.5).to_bytes(True)


def test_capitalization_and_punctuation_preserved():
    c = compress("NASA's Webb Telescope, launched Dec. 25!", 0.5)
    assert c.body_text() == "NA's We Teles, laun De. 25!"


def test_numbers_verbatim_by_default_and_optional():
    assert compress("Room 12345 and abc123", 0.25).prefixes == ["R", "12345", "a", "abc123"]
    assert compress("Room 12345", 0.25, verbatim_numbers=False).prefixes == ["R", "12"]
    slots = Compressed.from_bytes(compress("Room 12345", 0.25).to_bytes()).slots()
    assert slots[1].complete


def test_stats_distinguish_raw_prefix_from_payload():
    text = "The programmer discovered an unexpected problem in the application."
    stats = compress(text, 0.5).stats(text)
    assert stats["raw_prefix_bytes"] == len("Thprogrdiscoaunexpprobithapplic")
    assert stats["payload_bytes"] == HEADER_BYTES + stats["raw_prefix_bytes"] + stats["separator_bytes"]
    assert stats["original_bytes"] == len(text)


def test_adaptive_payload_roundtrip_and_complete_flags():
    text = "the cat sat on the mat"
    c = compress_with_lengths(text, [3, 1, 3, 1, 2, 2], retention_pct=25)
    parsed = Compressed.from_bytes(c.to_bytes())
    assert parsed == c
    slots = parsed.slots()
    assert [s.complete for s in slots] == [True, False, True, False, False, False]
    assert slots[1].min_len == 2 and slots[1].max_len == 1 + MAX_SUFFIX
    # zstd variant too
    assert Compressed.from_bytes(c.to_bytes(use_zstd=True)) == c


def test_adaptive_never_truncates_beyond_max_suffix():
    word = "a" * (MAX_SUFFIX + 10)
    c = compress_with_lengths(word, [1], retention_pct=25)
    assert len(c.prefixes[0]) == 10


def test_bad_inputs():
    with pytest.raises(ValueError):
        compress("x", 0)
    with pytest.raises(ValueError):
        compress("x", 1.5)
    with pytest.raises(ValueError):
        Compressed.from_bytes(b"\x09\x00\x32abc")


def test_merge_payloads_matches_compressing_the_joined_text():
    from neural_compression.benchmark import merge_payloads

    texts = ["The cat sat.", "  Then it slept!", "Done"]
    merged = merge_payloads([compress(t, 0.5) for t in texts])
    assert merged == compress("\n".join(texts), 0.5)
    adaptive = [compress_with_lengths(t, [1] * len(split_text(t)[1]), 50) for t in texts]
    m = merge_payloads(adaptive)
    assert Compressed.from_bytes(m.to_bytes()) == m
    assert m.complete == sum((a.complete for a in adaptive), [])
