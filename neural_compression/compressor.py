"""Prefix-based lossy compression and the binary payload format.

Text is split into *words* (maximal runs of ``\\w`` characters) and *separators*
(everything in between: whitespace, punctuation, symbols). Separators are
stored verbatim. Each word is replaced by a prefix of itself.

Because prefixes only contain word characters and separators only contain
non-word characters, the body ``sep0 prefix0 sep1 prefix1 ... sepN`` can be
split back into words and separators with the same regex: no extra delimiters
are needed.

What the decoder can know about each word (a :class:`Slot`) comes only from the
payload:

* **fixed mode**: every word keeps ``ceil(n * pct / 100)`` characters (at least
  1). The header stores ``pct``, so for a prefix of length ``k`` the decoder can
  work out which original lengths ``n`` are possible. At 50%, a 3-character
  prefix means the word had 5 or 6 characters.
* **adaptive mode**: prefix lengths vary per word, so the payload also carries
  a bitmap saying which words are stored complete. A truncated word may be up
  to ``MAX_SUFFIX`` characters longer than its prefix.

Payload layout (all integers unsigned)::

    byte 0      FORMAT_VERSION
    byte 1      flags: bit0 adaptive, bit1 verbatim_numbers, bit2 zstd body
    byte 2      retention percent (1..100)
    rest        body section, zstd-compressed if bit2 is set:
                  adaptive only: varint word count, then complete-bitmap
                  UTF-8 text: separators and prefixes
"""

from __future__ import annotations

import re
from compression import zstd
from dataclasses import dataclass, field

FORMAT_VERSION = 1
FLAG_ADAPTIVE = 1
FLAG_VERBATIM_NUMBERS = 2
FLAG_ZSTD = 4
HEADER_BYTES = 3
MAX_SUFFIX = 20  # adaptive mode: longest suffix the decoder may generate
ZSTD_LEVEL = 19

TOKEN_RE = re.compile(r"\w+|\W+")
WORD_RE = re.compile(r"\w+")


def is_word_char(ch: str) -> bool:
    return WORD_RE.fullmatch(ch) is not None


def split_text(text: str) -> tuple[list[str], list[str]]:
    """Split text into ``(separators, words)``; ``len(separators) == len(words) + 1``.

    ``separators[i]`` precedes ``words[i]``; the last separator follows the last
    word. Any separator may be empty only at the start or end of the text.
    """
    separators, words = [""], []
    for token in TOKEN_RE.findall(text):
        if WORD_RE.fullmatch(token):
            words.append(token)
            separators.append("")
        else:
            separators[-1] += token
    return separators, words


def retention_to_pct(retention: float) -> int:
    pct = round(retention * 100)
    if not 1 <= pct <= 100:
        raise ValueError(f"retention must be in (0, 1], got {retention}")
    return pct


def kept_chars(length: int, pct: int) -> int:
    """Number of characters fixed mode keeps from a word: ceil(length*pct/100), min 1."""
    return max(1, -(-length * pct // 100))


def feasible_lengths(prefix_len: int, pct: int) -> range:
    """Original word lengths that fixed mode would truncate to ``prefix_len``."""
    candidates = [n for n in range(prefix_len, prefix_len * 100 // pct + 2) if kept_chars(n, pct) == prefix_len]
    return range(candidates[0], candidates[-1] + 1)


def has_digit(word: str) -> bool:
    return any(ch.isdigit() for ch in word)


@dataclass(frozen=True)
class Slot:
    """Decoder-side description of one word: its prefix and possible total lengths."""

    prefix: str
    min_len: int
    max_len: int

    @property
    def complete(self) -> bool:
        return self.max_len == len(self.prefix)


@dataclass
class Compressed:
    """Everything stored for one document. Holds no information beyond the payload."""

    separators: list[str]
    prefixes: list[str]
    retention_pct: int
    adaptive: bool = False
    verbatim_numbers: bool = True
    complete: list[bool] = field(default_factory=list)  # adaptive mode only

    # ---- decoder view -------------------------------------------------
    def slots(self) -> list[Slot]:
        slots = []
        for i, prefix in enumerate(self.prefixes):
            k = len(prefix)
            if self.adaptive:
                lo, hi = (k, k) if self.complete[i] else (k + 1, k + MAX_SUFFIX)
            elif self.verbatim_numbers and has_digit(prefix):
                lo, hi = k, k
            else:
                lengths = feasible_lengths(k, self.retention_pct)
                lo, hi = lengths.start, lengths.stop - 1
            slots.append(Slot(prefix, lo, hi))
        return slots

    def body_text(self) -> str:
        parts = [self.separators[0]]
        for prefix, sep in zip(self.prefixes, self.separators[1:]):
            parts += [prefix, sep]
        return "".join(parts)

    # ---- serialization ------------------------------------------------
    def to_bytes(self, use_zstd: bool = False) -> bytes:
        flags = (FLAG_ADAPTIVE if self.adaptive else 0) | (FLAG_VERBATIM_NUMBERS if self.verbatim_numbers else 0)
        section = b""
        if self.adaptive:
            section += _varint(len(self.prefixes)) + _pack_bits(self.complete)
        section += self.body_text().encode("utf-8")
        if use_zstd:
            flags |= FLAG_ZSTD
            section = zstd.compress(section, level=ZSTD_LEVEL)
        return bytes([FORMAT_VERSION, flags, self.retention_pct]) + section

    @classmethod
    def from_bytes(cls, payload: bytes) -> "Compressed":
        if len(payload) < HEADER_BYTES or payload[0] != FORMAT_VERSION:
            raise ValueError("not a neural_compression payload (bad header)")
        flags, pct = payload[1], payload[2]
        section = payload[HEADER_BYTES:]
        if flags & FLAG_ZSTD:
            section = zstd.decompress(section)
        complete: list[bool] = []
        if flags & FLAG_ADAPTIVE:
            count, pos = _read_varint(section, 0)
            nbytes = (count + 7) // 8
            complete = _unpack_bits(section[pos : pos + nbytes], count)
            section = section[pos + nbytes :]
        separators, prefixes = split_text(section.decode("utf-8"))
        if flags & FLAG_ADAPTIVE and len(complete) != len(prefixes):
            raise ValueError("corrupt payload: bitmap size does not match word count")
        return cls(separators, prefixes, pct, bool(flags & FLAG_ADAPTIVE), bool(flags & FLAG_VERBATIM_NUMBERS), complete)

    # ---- size accounting ----------------------------------------------
    def stats(self, original: str | None = None) -> dict[str, int]:
        """Byte sizes. ``raw_prefix_bytes`` counts only stored word characters."""
        out = {
            "raw_prefix_bytes": sum(len(p.encode()) for p in self.prefixes),
            "separator_bytes": sum(len(s.encode()) for s in self.separators),
            "payload_bytes": len(self.to_bytes()),
            "payload_zstd_bytes": len(self.to_bytes(use_zstd=True)),
            "words": len(self.prefixes),
        }
        if original is not None:
            out["original_bytes"] = len(original.encode())
        return out


def compress(text: str, retention: float = 0.5, verbatim_numbers: bool = True) -> Compressed:
    """Fixed-ratio compression: keep ceil(retention * len) characters of each word."""
    pct = retention_to_pct(retention)
    separators, words = split_text(text)
    prefixes = [w if verbatim_numbers and has_digit(w) else w[: kept_chars(len(w), pct)] for w in words]
    return Compressed(separators, prefixes, pct, adaptive=False, verbatim_numbers=verbatim_numbers)


def compress_with_lengths(text: str, prefix_lengths: list[int], retention_pct: int) -> Compressed:
    """Adaptive-mode payload with explicit per-word prefix lengths (see adaptive.py)."""
    separators, words = split_text(text)
    if len(prefix_lengths) != len(words):
        raise ValueError("need one prefix length per word")
    lengths = [min(len(w), max(1, k)) for w, k in zip(words, prefix_lengths)]
    # A word whose suffix exceeds what the decoder may generate cannot be truncated that far.
    lengths = [max(k, len(w) - MAX_SUFFIX) for w, k in zip(words, lengths)]
    prefixes = [w[:k] for w, k in zip(words, lengths)]
    complete = [k == len(w) for w, k in zip(words, lengths)]
    return Compressed(separators, prefixes, retention_pct, adaptive=True, verbatim_numbers=False, complete=complete)


def start_lengths(words: list[str], retention_pct: int) -> list[int]:
    """Where an adaptive search starts: numbers in full, long words no shorter than MAX_SUFFIX allows."""
    return [len(w) if has_digit(w) else max(kept_chars(len(w), retention_pct), len(w) - MAX_SUFFIX) for w in words]


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        byte, n = n & 0x7F, n >> 7
        out.append(byte | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        shift += 7
        if not byte & 0x80:
            return value, pos


def _pack_bits(bits: list[bool]) -> bytes:
    out = bytearray((len(bits) + 7) // 8)
    for i, bit in enumerate(bits):
        if bit:
            out[i // 8] |= 1 << (i % 8)
    return bytes(out)


def _unpack_bits(data: bytes, count: int) -> list[bool]:
    return [bool(data[i // 8] >> (i % 8) & 1) for i in range(count)]
