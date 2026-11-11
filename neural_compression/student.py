"""Single-pass student decoder: a small bidirectional encoder that picks every missing suffix at once.

The LM decoder (``decoder.py``) writes suffixes token by token with constrained
beam search. This decoder reads the same payload and restores each sentence
with **one encoder forward pass**:

1. Every word becomes one input vector built from what the payload says about
   it: hashed position-aware characters of its stored prefix, its case pattern,
   the separators around it, its gap length range, and a *candidate bag*, the
   prior-weighted mean embedding of every vocabulary word that fits the prefix
   and the range.
2. A pre-LN transformer encoder (bidirectional) contextualises the words.
3. For each truncated word, the candidates are the vocabulary words (lower
   case, sorted) that start with the lower-cased prefix and have a length in
   the slot's range. Each scores ``proj(h) . E[w] + bias[w]``; one extra
   ``OOV`` score says "the word is not in the vocabulary". The argmax wins
   (ties go to the earlier vocabulary entry, so the choice is deterministic).
4. When ``OOV`` wins or no candidate fits, a character n-gram model built from
   the vocabulary spells a suffix of an allowed length.

The suffix keeps the prefix's case pattern: upper case after an all-caps prefix
of two or more letters, otherwise the vocabulary's lower case. A word whose real
spelling the decoder cannot produce simply needs a longer prefix in adaptive
mode, as with the LM decoder.

The output respects the payload by construction: a candidate starts with the
stored prefix and has an allowed length, separators are copied, and stored
complete words are never touched. ``decode`` uses only the payload and the
saved model directory.

Each sentence unit runs through the encoder on its own (batch of one, no
padding), so a unit's result never depends on which other units or payloads are
decoded with it. The merged corpus payload and the per-sentence payloads
therefore decode to the same words.
"""

from __future__ import annotations

import bisect
import json
import math
import re
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .compressor import MAX_SUFFIX, Compressed, Slot
from .decoder import DecodeResult, sentence_units
from .models import InferenceStats

CHAR_BUCKETS = 8192
SEP_BUCKETS = 2048
MAX_WORDS = 64  # positions; decoder units have at most 48 words
_SPACES = re.compile(r"\s+")
_LETTER = re.compile(r"[^\W\d_]")


def _h(s: str, buckets: int) -> int:
    return zlib.crc32(s.encode("utf-8")) % buckets


def case_id(prefix: str) -> int:
    if any(ch.isdigit() for ch in prefix):
        return 4
    if prefix.islower() or not _LETTER.search(prefix):
        return 0
    if len(prefix) >= 2 and prefix.isupper():
        return 2
    if prefix[0].isupper() and (len(prefix) == 1 or prefix[1:].islower()):
        return 1
    return 3


def suffix_case(prefix: str, suffix: str) -> str:
    return suffix.upper() if case_id(prefix) == 2 else suffix


class Vocab:
    """Sorted lower-case word list with counts; prefix ranges by bisection."""

    def __init__(self, words: list[str], counts: list[int]):
        order = sorted(range(len(words)), key=lambda i: words[i])
        self.words = [words[i] for i in order]
        self.counts = np.array([counts[i] for i in order], dtype=np.int64)
        self.lens = np.array([len(w) for w in self.words], dtype=np.int64)
        self.index = {w: i for i, w in enumerate(self.words)}
        self.logprior = np.log(self.counts / self.counts.sum()).astype(np.float32)

    def __len__(self) -> int:
        return len(self.words)

    def prefix_range(self, prefix: str) -> tuple[int, int]:
        if not prefix:
            return 0, len(self.words)
        a = bisect.bisect_left(self.words, prefix)
        b = bisect.bisect_left(self.words, prefix[:-1] + chr(ord(prefix[-1]) + 1), a)
        return a, b

    def candidates(self, slot: Slot) -> np.ndarray:
        """Vocabulary indices that can fill ``slot`` (empty if the prefix changes length when lower-cased)."""
        low = slot.prefix.lower()
        if len(low) != len(slot.prefix):
            return np.zeros(0, dtype=np.int64)
        a, b = self.prefix_range(low)
        lens = self.lens[a:b]
        return a + np.nonzero((lens >= slot.min_len) & (lens <= slot.max_len))[0]


class CharNgram:
    """Witten-Bell interpolated character n-gram over vocabulary words (lower case), for OOV suffixes."""

    END = "\x03"

    def __init__(self, words: list[str], counts: list[int], order: int = 5):
        self.order = order
        self.table: dict[str, dict[str, int]] = {}
        alphabet: set[str] = set()
        for w, c in zip(words, counts):
            s = "\x02" * (order - 1) + w + self.END
            alphabet.update(w)
            for i in range(order - 1, len(s)):
                for n in range(order):
                    ctx = s[i - n : i]
                    d = self.table.setdefault(ctx, {})
                    d[s[i]] = d.get(s[i], 0) + c
        self.alphabet = sorted(alphabet)
        self.totals = {ctx: (sum(d.values()), len(d)) for ctx, d in self.table.items()}

    def prob(self, ctx: str, ch: str) -> float:
        p = 1.0 / (len(self.alphabet) + 1)
        for n in range(self.order):  # shortest context first
            c = ctx[len(ctx) - n :] if n else ""
            if c not in self.table:
                break
            total, types = self.totals[c]
            p = (self.table[c].get(ch, 0) + types * p) / (total + types)
        return p

    def complete(self, prefix: str, min_chars: int, max_chars: int, beam: int = 8) -> str:
        """Most likely suffix (``min_chars``..``max_chars`` letters, then end of word)."""
        start = "\x02" * (self.order - 1) + prefix.lower()
        beams = [(0.0, "")]
        best = (-math.inf, "")
        for n in range(max_chars + 1):
            if n >= min_chars:
                for score, suf in beams:
                    total = score + math.log(self.prob((start + suf)[-(self.order - 1) :], self.END))
                    if total > best[0]:
                        best = (total, suf)
            if n == max_chars:
                break
            nxt = []
            for score, suf in beams:
                ctx = (start + suf)[-(self.order - 1) :]
                nxt += [(score + math.log(self.prob(ctx, ch)), suf + ch) for ch in self.alphabet]
            nxt.sort(key=lambda x: (-x[0], x[1]))
            beams = nxt[:beam]
        return best[1]


@dataclass
class Config:
    vocab_size: int
    dim: int = 256
    layers: int = 6
    heads: int = 4
    ff: int = 1024
    dropout: float = 0.1


class StudentModel(nn.Module):
    def __init__(self, cfg: Config, logprior: np.ndarray):
        super().__init__()
        d = cfg.dim
        self.cfg = cfg
        self.word = nn.Embedding(cfg.vocab_size, d)  # candidate bag input and output scores (tied)
        nn.init.normal_(self.word.weight, std=0.02)
        self.bias = nn.Parameter(torch.tensor(logprior))
        self.register_buffer("prior", torch.tensor(logprior), persistent=False)
        self.chars = nn.EmbeddingBag(CHAR_BUCKETS, d, mode="sum")
        self.sep_before = nn.Embedding(SEP_BUCKETS, d)
        self.sep_after = nn.Embedding(SEP_BUCKETS, d)
        self.case = nn.Embedding(5, d)
        self.gap_lo = nn.Embedding(MAX_SUFFIX + 1, d)
        self.gap_hi = nn.Embedding(MAX_SUFFIX + 1, d)
        self.kind = nn.Embedding(4, d)  # complete in vocab, complete OOV, truncated, truncated without candidates
        self.ncand = nn.Embedding(16, d)
        self.pos = nn.Embedding(MAX_WORDS, d)
        for emb in (self.chars, self.sep_before, self.sep_after, self.case, self.gap_lo, self.gap_hi, self.kind, self.ncand, self.pos):
            nn.init.normal_(emb.weight, std=0.02)
        layer = nn.TransformerEncoderLayer(d, cfg.heads, cfg.ff, cfg.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, d)
        self.oov = nn.Linear(d, 1)
        nn.init.constant_(self.oov.bias, -10.0)  # start below every in-vocabulary candidate (log prior)

    def encode(self, f: dict, bag: torch.Tensor) -> torch.Tensor:
        """Contextual vectors ``(B, L, d)``. ``bag`` is the candidate-bag vector per word ``(B, L, d)``."""
        B, L = f["case"].shape
        x = (
            self.chars(f["char_ids"], f["char_offsets"]).view(B, L, -1)
            + self.sep_before(f["sep_before"])
            + self.sep_after(f["sep_after"])
            + self.case(f["case"])
            + self.gap_lo(f["gap_lo"])
            + self.gap_hi(f["gap_hi"])
            + self.kind(f["kind"])
            + self.ncand(f["ncand"])
            + self.pos(torch.arange(L, device=bag.device))[None]
            + bag
        )
        pad = f.get("pad")
        return self.norm(self.encoder(x, src_key_padding_mask=pad))

    def bag_from_candidates(self, cand: torch.Tensor) -> torch.Tensor:
        """Prior-weighted mean embedding of the candidate indices ``cand`` (inference, one slot)."""
        w = torch.softmax(self.prior[cand], 0)
        return w @ self.word.weight[cand]

    def scores(self, h: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        """Scores of ``cand`` followed by the OOV score, for one contextual vector ``h``."""
        q = self.out(h)
        return torch.cat([self.word.weight[cand] @ q + self.bias[cand], self.oov(h)])


# ---------------------------------------------------------------- features


def unit_features(doc: Compressed, slots: list[Slot], unit: range, vocab: Vocab) -> dict:
    """Per-word encoder inputs and candidate sets for one decoder unit. Uses only the payload."""
    char_ids, offsets, rows, spans = [], [], [], []
    for i in unit:
        slot = slots[i]
        p = slot.prefix
        low = p.lower()
        offsets.append(len(char_ids))
        char_ids += [_h(f"{j}|{ch}", CHAR_BUCKETS) for j, ch in enumerate(low[:12])]
        char_ids += [_h(f"-{r}|{low[-r]}", CHAR_BUCKETS) for r in range(1, min(3, len(low)) + 1)]
        char_ids.append(_h(f"len|{min(len(low), 20)}", CHAR_BUCKETS))
        if slot.complete:
            idx = vocab.index.get(low, -1)
            cand = np.array([idx] if idx >= 0 else [], dtype=np.int64)
            kind = 0 if idx >= 0 else 1
            lo = hi = 0
            span = (idx, idx + 1, 0, 1 << 30) if idx >= 0 else (0, 0, 0, 0)
        else:
            cand = vocab.candidates(slot)
            kind = 2 if len(cand) else 3
            lo, hi = min(MAX_SUFFIX, max(0, slot.min_len - len(p))), min(MAX_SUFFIX, slot.max_len - len(p))
            span = (*vocab.prefix_range(low), slot.min_len, slot.max_len) if len(cand) else (0, 0, 0, 0)
        spans.append(span)
        # Only the unit's own text: what opens its first word after any whitespace, and what closes
        # its last word before any whitespace, so a sentence decodes the same alone or inside a document.
        before = _SPACES.split(doc.separators[i])[-1] if i == unit.start else doc.separators[i]
        after = _SPACES.split(doc.separators[i + 1])[0] if i + 1 == unit.stop else doc.separators[i + 1]
        rows.append(
            (
                _h("S|" + _SPACES.sub(" ", before), SEP_BUCKETS),
                _h("S|" + _SPACES.sub(" ", after), SEP_BUCKETS),
                case_id(p),
                lo,
                hi,
                kind,
                min(15, len(cand).bit_length()),
                cand,
            )
        )
    cols = list(zip(*rows))
    return {
        "char_ids": char_ids,
        "char_offsets": offsets,
        "sep_before": list(cols[0]),
        "sep_after": list(cols[1]),
        "case": list(cols[2]),
        "gap_lo": list(cols[3]),
        "gap_hi": list(cols[4]),
        "kind": list(cols[5]),
        "ncand": list(cols[6]),
        "cand": list(cols[7]),
        "spans": spans,  # (a, b, min_len, max_len): candidates are vocab[a:b] with a length in range
    }


INT_KEYS = ("sep_before", "sep_after", "case", "gap_lo", "gap_hi", "kind", "ncand")


def to_tensors(f: dict, device) -> dict:
    t = {k: torch.tensor([f[k]], device=device) for k in INT_KEYS}
    t["char_ids"] = torch.tensor(f["char_ids"], device=device)
    t["char_offsets"] = torch.tensor(f["char_offsets"], device=device)
    return t


# ---------------------------------------------------------------- decoder


class StudentDecoder:
    """Drop-in for ``Decoder`` in ``adaptive.compress_adaptive_many``: ``decode_many(payloads)``."""

    def __init__(self, model: StudentModel, vocab: Vocab, ngram: CharNgram, device: str = "cpu"):
        self.model = model.eval().to(device)
        self.vocab = vocab
        self.ngram = ngram
        self.device = torch.device(device)
        self.stats = InferenceStats()
        self.fallbacks = 0

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "StudentDecoder":
        path = Path(path)
        meta = json.loads((path / "config.json").read_text())
        vocab_rows = [line.split("\t") for line in (path / "vocab.tsv").read_text(encoding="utf-8").splitlines()]
        vocab = Vocab([w for w, _ in vocab_rows], [int(c) for _, c in vocab_rows])
        model = StudentModel(Config(**meta["config"]), vocab.logprior)
        state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict({k: v.float() for k, v in state.items()})
        ngram = CharNgram(vocab.words, vocab.counts.tolist(), order=meta.get("ngram_order", 5))
        return cls(model, vocab, ngram, device)

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.half() for k, v in self.model.state_dict().items()}, path / "model.pt")
        (path / "vocab.tsv").write_text("".join(f"{w}\t{c}\n" for w, c in zip(self.vocab.words, self.vocab.counts.tolist())), encoding="utf-8")
        (path / "config.json").write_text(json.dumps({"config": vars(self.model.cfg), "ngram_order": self.ngram.order, **(extra or {})}, indent=2))

    @torch.no_grad()
    def restore_unit(self, doc: Compressed, slots: list[Slot], unit: range) -> list[str]:
        f = unit_features(doc, slots, unit, self.vocab)
        t = to_tensors(f, self.device)
        m = self.model
        d = m.cfg.dim
        cands = [torch.from_numpy(c).to(self.device) for c in f["cand"]]
        bag = torch.stack([m.bag_from_candidates(c) if len(c) else torch.zeros(d, device=self.device) for c in cands])
        h = m.encode(t, bag[None])[0]
        self.stats.forward_calls += 1
        self.stats.sequences += 1
        words = []
        for j, i in enumerate(unit):
            slot = slots[i]
            if slot.complete:
                words.append(slot.prefix)
                continue
            k = len(slot.prefix)
            best = -1
            if len(cands[j]):
                best = int(torch.argmax(m.scores(h[j], cands[j])))
            if 0 <= best < len(cands[j]):
                suffix = self.vocab.words[int(f["cand"][j][best])][k:]
            else:
                self.fallbacks += 1
                suffix = self.ngram.complete(slot.prefix, max(0, slot.min_len - k), slot.max_len - k)
            words.append(slot.prefix + suffix_case(slot.prefix, suffix))
        return words

    def decode(self, payload: Compressed | bytes) -> DecodeResult:
        return self.decode_many([payload])[0]

    def decode_many(self, payloads: list[Compressed | bytes]) -> list[DecodeResult]:
        results = []
        for p in payloads:
            t0 = time.perf_counter()
            calls = self.stats.forward_calls
            doc = Compressed.from_bytes(p) if isinstance(p, bytes) else p
            slots = doc.slots()
            words = list(doc.prefixes)
            for unit in sentence_units(doc):
                if all(slots[i].complete for i in unit):
                    continue
                words[unit.start : unit.stop] = self.restore_unit(doc, slots, unit)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            parts = [doc.separators[0]]
            for word, sep in zip(words, doc.separators[1:]):
                parts += [word, sep]
            stats = InferenceStats(forward_calls=self.stats.forward_calls - calls)
            results.append(DecodeResult("".join(parts), words, time.perf_counter() - t0, 0, stats))
        return results
