"""Model-based reconstruction of a compressed payload.

Every strategy uses the same model, prompt format, few-shot sentences, and
constrained beam search. Only the *window* of compressed words the model sees
changes:

* ``context=0`` (Strategy A): each truncated word is restored from its own
  prefix only. Window = 1 word.
* ``context=k`` (Strategy B): each truncated word is restored from a window
  of itself plus ``k`` surrounding compressed words (k/2 on each side, fewer
  at sentence edges). Context is bidirectional because every prefix is known.
  The whole window is generated jointly, and only the centre word is kept.
* ``context="full"`` (Strategy C): the whole sentence is the window and is
  generated in one pass. Every word is kept.

Prompt format (few-shot examples are cut to the same window size)::

    Compressed: Th progr disco a unexp prob
    Original: The programmer discovered an unexpected problem

The decoder only uses the payload (``Compressed``), the model, and the constants
in this file. It never sees the original text.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .compressor import WORD_RE, Compressed, Slot, compress, split_text
from .models import Gap, InferenceStats, LanguageModel, Request, constrained_beam_search

MAX_UNIT_WORDS = 48
SHOTS_PER_SENTENCE = 2
SENTENCE_END = re.compile(r"[.!?][\"')\]]*\s|\n")

FEWSHOT = [
    "We stayed inside all afternoon because the weather was terrible.",
    "Install the package with pip, then import it at the top of your script.",
    "The study found that regular exercise improves memory in older adults.",
]
INSTRUCTION = "Every word in the compressed text was cut short. Restore the original text.\n\n"

Context = int | str  # int window size, or "full"


@dataclass
class DecodeResult:
    text: str
    words: list[str]
    seconds: float
    failures: int = 0  # requests where the search found no valid output
    stats: InferenceStats = field(default_factory=InferenceStats)


def parse_context(value: str | int) -> Context:
    if value == "full":
        return "full"
    k = int(value)
    if k < 0:
        raise ValueError("context must be >= 0 or 'full'")
    return k


def sentence_units(c: Compressed) -> list[range]:
    """Word index ranges decoded together (sentences, capped at MAX_UNIT_WORDS)."""
    units, start = [], 0
    for i in range(len(c.prefixes)):
        end_here = SENTENCE_END.search(c.separators[i + 1]) or i + 1 == len(c.prefixes)
        if end_here or i + 1 - start >= MAX_UNIT_WORDS:
            units.append(range(start, i + 1))
            start = i + 1
    return units


def _span(separators: list[str], words: list[str], span: range) -> str:
    """Words in ``span`` with the separators between them, plus trailing punctuation."""
    parts = []
    for i in span:
        if i > span.start:
            parts.append(separators[i])
        parts.append(words[i])
    return "".join(parts) + separators[span.stop].rstrip()


def _gap(slot: Slot) -> Gap:
    k = len(slot.prefix)
    return Gap(max(0, slot.min_len - k), slot.max_len - k)


class Decoder:
    def __init__(self, lm: LanguageModel, context: Context = "full", beam_size: int = 4, batch_size: int = 16):
        self.lm = lm
        self.context = context
        self.beam_size = beam_size
        self.batch_size = batch_size

    def _window(self, unit: range, i: int) -> range:
        if self.context == "full":
            return unit
        left = self.context // 2
        return range(max(unit.start, i - left), min(unit.stop, i + (self.context - left) + 1))

    def _fewshot_prefix(self, pct: int) -> str:
        """Instruction plus example sentences, cut into chunks of the current window size."""
        shots = []
        for sentence in FEWSHOT:
            c = compress(sentence, pct / 100)
            seps, words = split_text(sentence)
            size = len(words) if self.context == "full" else self.context + 1
            starts = list(range(0, len(words), size))
            # At most SHOTS_PER_SENTENCE chunks, evenly spread: the prompt prefix is copied into
            # every beam hypothesis, so its length directly drives decoding cost.
            if len(starts) > SHOTS_PER_SENTENCE:
                step = (len(starts) - 1) / (SHOTS_PER_SENTENCE - 1)
                starts = [starts[round(j * step)] for j in range(SHOTS_PER_SENTENCE)]
            for start in starts:
                span = range(start, min(len(words), start + size))
                shots.append(f"Compressed: {_span(c.separators, c.prefixes, span)}\nOriginal: {_span(seps, words, span)}\n\n")
        return INSTRUCTION + "".join(shots)

    def _request(self, doc: Compressed, slots: list[Slot], span: range) -> Request:
        """Prompt + template for jointly restoring the words in ``span``."""
        template: list = [" "]
        for i in span:
            if i > span.start:
                template[-1] += doc.separators[i]
            template[-1] += doc.prefixes[i]
            if not slots[i].complete:
                template += [_gap(slots[i]), ""]
        template[-1] += doc.separators[span.stop].rstrip() + "\n"
        return Request(f"Compressed: {_span(doc.separators, doc.prefixes, span)}\nOriginal:", template)

    # ---- decoding -------------------------------------------------------
    def decode(self, payload: Compressed | bytes) -> DecodeResult:
        return self.decode_many([payload])[0]

    def decode_many(self, payloads: list[Compressed | bytes]) -> list[DecodeResult]:
        """Decode several documents, batching model calls across all of them."""
        docs = [Compressed.from_bytes(p) if isinstance(p, bytes) else p for p in payloads]
        by_pct: dict[int, list[int]] = {}
        for d, doc in enumerate(docs):
            by_pct.setdefault(doc.retention_pct, []).append(d)
        results: list[DecodeResult | None] = [None] * len(docs)
        for pct, indices in by_pct.items():  # few-shot examples depend on the retention ratio
            for d, res in zip(indices, self._decode_group([docs[d] for d in indices], pct)):
                results[d] = res
        return results

    def _decode_group(self, docs: list[Compressed], pct: int) -> list[DecodeResult]:
        t0 = time.perf_counter()
        before = InferenceStats(**vars(self.lm.stats))
        requests: list[Request] = []
        owners: list[tuple[int, range, list[int]]] = []  # (doc, window, word indices to keep)
        words_out = [list(doc.prefixes) for doc in docs]
        for d, doc in enumerate(docs):
            slots = doc.slots()
            for unit in sentence_units(doc):
                todo = [i for i in unit if not slots[i].complete]
                if not todo:
                    continue
                if self.context == "full":
                    requests.append(self._request(doc, slots, unit))
                    owners.append((d, unit, todo))
                else:
                    for i in todo:
                        window = self._window(unit, i)
                        requests.append(self._request(doc, slots, window))
                        owners.append((d, window, [i]))

        outputs = constrained_beam_search(
            self.lm, requests, shared_prefix=self._fewshot_prefix(pct), beam_size=self.beam_size, batch_size=self.batch_size
        )
        failures = [0] * len(docs)
        for (d, window, keep), text in zip(owners, outputs):
            if text is None:
                failures[d] += 1
                continue
            restored = WORD_RE.findall(text)
            assert len(restored) == len(window), "template guarantees one word per slot"
            for i in keep:
                words_out[d][i] = restored[i - window.start]

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
