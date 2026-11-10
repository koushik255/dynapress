"""Adaptive prefix lengths: keep only as many characters as the decoder needs.

The compressor may read the original text, and it runs the *real* decoder on
the *real* serialized payload to see which words come back wrong:

1. Start every word at ``start_retention`` (fixed-mode rounding).
2. Serialize, parse back, decode with the shared decoder.
3. If word accuracy >= target, stop.
4. Otherwise lengthen every wrong word by ``step`` characters. After
   ``max_rounds`` rounds, store wrong words in full instead.
5. Go to 2.

Adding characters does not always help. With joint decoding, lengthening one
word can break a neighbour that was right before. The loop does not assume
monotonic progress: each round re-checks every word, and a word that becomes
wrong is lengthened like any other. Termination is guaranteed because every
round that does not stop strictly increases the total number of stored
characters, and a fully stored word is always reconstructed exactly (its slot
has no gap). The accuracy returned is the one measured on the final payload.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .compressor import MAX_SUFFIX, Compressed, compress_with_lengths, has_digit, kept_chars, retention_to_pct, split_text
from .decoder import Decoder


@dataclass
class AdaptiveResult:
    payload: bytes
    word_accuracy: float  # measured by decoding ``payload``
    rounds: int
    decoded_text: str
    seconds: float = 0.0
    forward_calls: float = 0.0
    history: list[tuple[int, float]] = field(default_factory=list)  # (payload bytes, word accuracy) per round


def compress_adaptive_many(
    texts: list[str],
    decoder: Decoder,
    start_retention: float = 0.25,
    target_accuracy: float = 1.0,
    step: int = 1,
    max_rounds: int = 6,
) -> list[AdaptiveResult]:
    """Adaptively compress several texts, batching decoder calls across texts."""
    pct = retention_to_pct(start_retention)
    words = [split_text(t)[1] for t in texts]
    # Numbers start in full; long words start no shorter than the decoder's MAX_SUFFIX allows.
    lengths = [[len(w) if has_digit(w) else max(kept_chars(len(w), pct), len(w) - MAX_SUFFIX) for w in ws] for ws in words]
    results: list[AdaptiveResult | None] = [None] * len(texts)
    history: list[list[tuple[int, float]]] = [[] for _ in texts]
    seconds = [0.0] * len(texts)
    calls = [0.0] * len(texts)
    pending = list(range(len(texts)))
    round_no = 0
    while pending:
        round_no += 1
        payloads = [compress_with_lengths(texts[i], lengths[i], pct).to_bytes() for i in pending]
        # The decoder sees only what a real receiver would: the parsed payload.
        t0 = time.perf_counter()
        decoded = decoder.decode_many([Compressed.from_bytes(p) for p in payloads])
        elapsed = time.perf_counter() - t0
        still = []
        for i, payload, res in zip(pending, payloads, decoded):
            seconds[i] += res.seconds or elapsed / len(pending)
            calls[i] += res.stats.forward_calls
            wrong = [j for j, (got, ref) in enumerate(zip(res.words, words[i])) if got != ref]
            accuracy = 1 - len(wrong) / max(1, len(words[i]))
            history[i].append((len(payload), accuracy))
            if accuracy >= target_accuracy:
                results[i] = AdaptiveResult(payload, accuracy, round_no, res.text, seconds[i], calls[i], history[i])
                continue
            for j in wrong:
                full = len(words[i][j])
                lengths[i][j] = full if round_no >= max_rounds else min(full, lengths[i][j] + step)
            still.append(i)
        pending = still
    return results


def compress_adaptive(text: str, decoder: Decoder, **kwargs) -> AdaptiveResult:
    return compress_adaptive_many([text], decoder, **kwargs)[0]
