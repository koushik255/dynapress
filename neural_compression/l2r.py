"""Left-to-right adaptive compression: each word is restored from the text before it.

The sentence decoder (``decoder.py``) restores a whole sentence jointly, so the
adaptive loop (``adaptive.py``) must decode whole sentences again every round,
and one hard word keeps its sentence in the loop. Here every truncated word is
its own small request::

    prompt     "\\n" + the restored text to the left: this sentence so far,
               after the ``lookback`` sentences before it
    template   separator + stored prefix, the missing suffix, the separator that follows

The prompt holds restored words only, and the restored text must equal the
original. The compressor therefore knows every prompt before anything is
decoded: it tests each word alone with the true text on its left, all words in
one batch, and lengthens only the words that come back wrong.

Decoding has to go in order, because each word needs the words before it.
With ``lookback=0`` sentences are independent and are decoded side by side.
With ``lookback>0`` a sentence needs the one before it, so a document is one
long chain of single-word model calls.

The one-word tests and the real decode run in different batches, and batching
changes the model's floating-point results slightly. A near-tie can therefore
come out differently in the real decode. ``compress_left_to_right`` always
decodes its final payload in full and lengthens any word that disagrees, so the
payload it returns is one the decoder restored exactly.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from .compressor import WORD_RE, Compressed, Slot, compress_with_lengths, retention_to_pct, split_text, start_lengths
from .decoder import DecodeResult, _gap, sentence_units
from .models import InferenceStats, LanguageModel, Request, constrained_beam_search

_SPACE = re.compile(r"\s")


def _lead(separator: str) -> str:
    """What follows the last whitespace: the punctuation that opens the next word, such as a quote."""
    return _SPACE.split(separator)[-1]


class LeftToRightDecoder:
    def __init__(self, lm: LanguageModel, lookback: int = 1, beam_size: int = 4, batch_size: int = 16):
        if lookback < 0:
            raise ValueError("lookback must be >= 0")
        self.lm = lm
        self.lookback = lookback
        self.beam_size = beam_size
        self.batch_size = batch_size

    def window_starts(self, doc: Compressed) -> list[int]:
        """For each word, the first word its prompt shows: the start of the sentence ``lookback`` sentences back."""
        units = sentence_units(doc)
        starts: list[int] = []
        for u, unit in enumerate(units):
            starts += [units[max(0, u - self.lookback)].start] * len(unit)
        return starts

    def _request(self, doc: Compressed, slot: Slot, words: list[str], i: int, start: int) -> Request:
        """Restore word ``i`` given ``words[start:i]``. Only the payload and those words are used."""
        seps = doc.separators
        prompt = ["\n"]  # a prompt may not be empty
        for j in range(start, i):
            prompt += [seps[j] if j > start else _lead(seps[j]), words[j]]
        before = seps[i] if i > start else _lead(seps[i])
        # The word ends where the next separator starts. Show it up to its first whitespace.
        after = seps[i + 1] + ("" if i + 1 < len(doc.prefixes) else "\n")
        space = _SPACE.search(after)
        return Request("".join(prompt), [before + slot.prefix, _gap(slot), after[: space.end()] if space else after])

    def _search(self, requests: list[Request]) -> list[str | None]:
        """The restored word for each request, or None if the search found nothing."""
        order = sorted(range(len(requests)), key=lambda r: len(requests[r].prompt))  # similar lengths: less padding
        found = constrained_beam_search(self.lm, [requests[r] for r in order], beam_size=self.beam_size, batch_size=self.batch_size)
        words: list[str | None] = [None] * len(requests)
        for r, text in zip(order, found):
            if text is not None:
                (words[r],) = WORD_RE.findall(text)  # the template holds exactly one word
        return words

    def restore_alone(self, doc: Compressed, words: list[str], indices: list[int]) -> list[str | None]:
        """Restore each word in ``indices`` on its own, with ``words`` as the text to its left.

        The compressor calls this with the true words. The result for one word does not
        depend on how long any other word's prefix is.
        """
        slots, starts = doc.slots(), self.window_starts(doc)
        return self._search([self._request(doc, slots[i], words, i, starts[i]) for i in indices])

    # ---- decoding -------------------------------------------------------
    def decode(self, payload: Compressed | bytes) -> DecodeResult:
        return self.decode_many([payload])[0]

    def decode_many(self, payloads: list[Compressed | bytes]) -> list[DecodeResult]:
        """Decode several documents, batching model calls across words that do not depend on each other."""
        t0 = time.perf_counter()
        before = InferenceStats(**vars(self.lm.stats))
        docs = [Compressed.from_bytes(p) if isinstance(p, bytes) else p for p in payloads]
        slots = [doc.slots() for doc in docs]
        starts = [self.window_starts(doc) for doc in docs]
        words_out = [list(doc.prefixes) for doc in docs]
        failures = [0] * len(docs)
        # A chain is a run of truncated words that must be restored in order.
        chains: list[tuple[int, list[int]]] = []
        for d, doc in enumerate(docs):
            spans = sentence_units(doc) if self.lookback == 0 else [range(len(doc.prefixes))]
            chains += [(d, [i for i in span if not slots[d][i].complete]) for span in spans]
        for step in range(max((len(chain) for _, chain in chains), default=0)):
            live = [(d, chain[step]) for d, chain in chains if step < len(chain)]
            requests = [self._request(docs[d], slots[d][i], words_out[d], i, starts[d][i]) for d, i in live]
            for (d, i), word in zip(live, self._search(requests)):
                if word is None:
                    failures[d] += 1
                else:
                    words_out[d][i] = word

        elapsed = time.perf_counter() - t0
        delta = InferenceStats(**{k: v - getattr(before, k) for k, v in vars(self.lm.stats).items()})
        total_words = sum(len(doc.prefixes) for doc in docs) or 1
        results = []
        for d, doc in enumerate(docs):
            share = len(doc.prefixes) / total_words  # attribute batched cost proportionally
            parts = [doc.separators[0]]
            for word, sep in zip(words_out[d], doc.separators[1:]):
                parts += [word, sep]
            stats = InferenceStats(**{k: v * share for k, v in vars(delta).items()})
            results.append(DecodeResult("".join(parts), words_out[d], elapsed * share, failures[d], stats))
        return results


@dataclass
class LeftToRightResult:
    payload: bytes
    lengths: list[int]  # characters stored per word
    tries: list[int]  # one-word searches spent on each word
    repairs: list[int]  # words lengthened because the full decode disagreed with the one-word search
    decoded_text: str  # the full decode of ``payload``; equals the input text
    verify_decodes: int  # full decodes run (1 if nothing needed repair)
    seconds: float  # total, of which...
    search_seconds: float  # ...finding the lengths; the rest is the full decodes
    forward_calls: float


def compress_left_to_right(text: str, decoder: LeftToRightDecoder, start_retention: float = 0.25) -> LeftToRightResult:
    """Shortest prefix per word (from ``start_retention`` up) that ``decoder`` restores, verified by a full decode."""
    t0 = time.perf_counter()
    calls = decoder.lm.stats.forward_calls
    pct = retention_to_pct(start_retention)
    words = split_text(text)[1]
    lengths = start_lengths(words, pct)
    tries = [0] * len(words)

    def payload() -> bytes:
        return compress_with_lengths(text, lengths, pct).to_bytes()

    pending = [i for i, w in enumerate(words) if lengths[i] < len(w)]
    while pending:
        # Templates come from the parsed payload, exactly as the decoder will derive them.
        restored = decoder.restore_alone(Compressed.from_bytes(payload()), words, pending)
        still = []
        for i, got in zip(pending, restored):
            tries[i] += 1
            if got != words[i]:
                lengths[i] += 1
                if lengths[i] < len(words[i]):  # a word stored in full needs no model
                    still.append(i)
        pending = still
    search_seconds = time.perf_counter() - t0

    repairs: list[int] = []
    decodes = 0
    while True:
        final = payload()
        doc = Compressed.from_bytes(final)
        res = decoder.decode(doc)
        decodes += 1
        wrong = [i for i, (got, ref) in enumerate(zip(res.words, words)) if got != ref]
        if not wrong:
            break
        # A wrong word whose prompt was right is a real disagreement. The others may only be
        # wrong because of it, so they are left alone and checked again by the next decode.
        starts = decoder.window_starts(doc)
        for i in wrong:
            if all(res.words[j] == words[j] for j in range(starts[i], i)):
                lengths[i] += 1
                repairs.append(i)
    return LeftToRightResult(
        final,
        lengths,
        tries,
        repairs,
        res.text,
        decodes,
        time.perf_counter() - t0,
        search_seconds,
        decoder.lm.stats.forward_calls - calls,
    )
