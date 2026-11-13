"""Model loading and character-constrained beam search for causal LMs.

The decoder describes the text it wants as a *template*: a list of fixed
strings (separators and stored prefixes, which must be reproduced exactly) and
:class:`Gap` objects (missing suffixes: word characters only, within a length
range). Beam search runs over the model's own tokens, but a token is only
allowed if its characters keep the output consistent with the template. The
model can therefore never change a stored prefix or separator; it only chooses
the gap characters and where each gap ends.

Matching is deterministic. Gap characters are word characters and the fixed text
after a gap always starts with a non-word character or the terminator, so
there is exactly one way to align any string against the template.
"""

from __future__ import annotations

import bisect
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .compressor import is_word_char

_LEADING_WORD = re.compile(r"\w*")
DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M"
DEFAULT_REVISION = "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"  # pinned for reproducibility


@dataclass(frozen=True)
class Gap:
    min_chars: int
    max_chars: int


Template = list  # list[str | Gap]; must end with a non-empty fixed string


def advance(template: Template, state: tuple[int, int], text: str) -> tuple[int, int] | None:
    """Consume ``text`` from ``state`` = (segment index, chars used in segment).

    Returns the new state, or None if the text cannot follow the template.
    """
    seg, pos = state
    for ch in text:
        while True:
            if seg == len(template):
                return None
            part = template[seg]
            if isinstance(part, str):
                if ch != part[pos]:
                    return None
                pos += 1
                if pos == len(part):
                    seg, pos = seg + 1, 0
                break
            if pos < part.max_chars and is_word_char(ch):
                pos += 1
                break
            if pos >= part.min_chars:  # close the gap; retry this char on the next segment
                seg, pos = seg + 1, 0
                continue
            return None
    return seg, pos


@dataclass
class Request:
    prompt: str
    template: Template


@dataclass
class _Hyp:
    request: int
    row: int  # row of the parent in the current cache batch
    state: tuple[int, int]
    text: str
    score: float


@dataclass
class InferenceStats:
    forward_calls: int = 0  # model forward passes (each may cover many sequences)
    sequences: int = 0  # requests decoded
    prompt_tokens: int = 0
    generated_tokens: int = 0  # tokens fed through the model during beam search
    seconds: float = 0.0

    def add(self, other: "InferenceStats") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))


@dataclass
class LanguageModel:
    name: str
    revision: str
    backend: object  # OnnxLlama (default) or TorchBackend: prefix / start / step
    tokenizer: object  # tokenizers.Tokenizer
    token_strings: list[str]
    ids_by_string: dict[str, list[int]]  # exact string -> token ids
    sorted_strings: list[str]  # all token strings, sorted (for prefix range lookups)
    sorted_ids: np.ndarray  # token id for each entry of sorted_strings
    word_ids: np.ndarray  # tokens starting with a word character...
    word_lead: np.ndarray  # ...and how many word characters each starts with
    parameters: int
    size_bytes: int  # weights in memory
    device: str = "cpu"
    weights: str = "fp32"  # or "int8"
    stats: InferenceStats = field(default_factory=InferenceStats)
    _prefix_cache: dict = field(default_factory=dict)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def describe(self) -> dict:
        return {
            "model": self.name, "revision": self.revision,
            "parameters": self.parameters, "size_bytes": self.size_bytes, "weights": self.weights,
            "attention_backend": self.backend.attention_backend,
        }


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if "torch" in sys.modules:
        sys.modules["torch"].manual_seed(seed)


def _model_file(name: str, revision: str, filename: str) -> str:
    if Path(name).is_dir():
        return str(Path(name) / filename)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(name, filename, revision=revision)


def _int8_checkpoint(name: str, revision: str) -> Path:
    """An int8 copy of ``model.safetensors``, quantized on first use and cached locally.

    Local model directories get it next to their checkpoint (redone when that is newer);
    Hub models in ``~/.cache/neural_compression``, keyed by name and revision.
    """
    from .onnx_llama import quantize_checkpoint

    if Path(name).is_dir():
        src, path = Path(name) / "model.safetensors", Path(name) / "model.int8.safetensors"
        stale = not path.exists() or path.stat().st_mtime < src.stat().st_mtime
    else:
        cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "neural_compression"
        path = cache / f"{name.replace('/', '--')}@{revision}" / "model.int8.safetensors"
        stale = not path.exists()
    if stale:
        print(f"quantizing {name} to int8 -> {path}", file=sys.stderr)
        quantize_checkpoint(_model_file(name, revision, "model.safetensors"), path)
    return path


def load_model(
    name: str = DEFAULT_MODEL,
    revision: str | None = None,
    threads: int | None = None,
    seed: int = 0,
    device: str | None = None,
    backend: str | None = None,
    weights: str = "fp32",
) -> LanguageModel:
    """Load a causal LM (``name`` is a Hub id or a local directory).

    ``backend`` is "onnx" (ONNX Runtime, CPU, Llama-architecture models; the default on
    CPU) or "torch" (needs the ``torch`` dependency group; the default for CUDA).
    ``weights`` is "fp32" or experimental CPU ONNX "int8" (see ``onnx_llama``).
    Int8 outputs depend on padding; adaptive payloads are not reliably lossless.
    ``device`` defaults to CUDA if torch is installed and sees a GPU, else CPU.
    ``revision`` defaults to the pinned commit for the default model.
    """
    from tokenizers import Tokenizer

    revision = revision or (DEFAULT_REVISION if name == DEFAULT_MODEL else "main")
    if device is None:
        try:
            import torch
        except ImportError:
            device = "cpu"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
    backend = backend or ("torch" if device != "cpu" else "onnx")
    if weights not in ("fp32", "int8") or (weights == "int8" and backend != "onnx"):
        raise ValueError(f"weights={weights!r} is not supported on the {backend} backend")
    if backend == "torch":
        from .torch_backend import TorchBackend  # imports torch, so set_seed seeds it too
    set_seed(seed)
    threads = threads or max(1, (os.cpu_count() or 2) // 2)
    tokenizer = Tokenizer.from_file(_model_file(name, revision, "tokenizer.json"))
    if backend == "onnx":
        if device != "cpu":
            raise ValueError("the onnx backend runs on CPU only; use backend='torch' for CUDA")
        from .onnx_llama import OnnxLlama, read_safetensors

        config = json.loads(Path(_model_file(name, revision, "config.json")).read_text())
        if config.get("model_type") != "llama":
            raise ValueError(f"the onnx backend supports Llama-architecture models, not {config.get('model_type')}; use backend='torch'")
        path = _int8_checkpoint(name, revision) if weights == "int8" else _model_file(name, revision, "model.safetensors")
        tensors = read_safetensors(path)
        model = OnnxLlama(config, tensors, threads)
        heads = ("lm_head.weight", "lm_head.weight.q8", "model.embed_tokens.weight", "model.embed_tokens.weight.q8")
        vocab_rows = next(tensors[k].shape[0] for k in heads if k in tensors)
        parameters = sum(w.size for k, w in tensors.items() if not k.endswith(".scale"))
        size_bytes = sum(w.nbytes for w in tensors.values())
    else:
        model = TorchBackend(name, revision, device, threads)
        vocab_rows = model.vocab_rows
        parameters = model.parameters
        size_bytes = parameters * 4
    special = {i for i, t in tokenizer.get_added_tokens_decoder().items() if t.special}
    vocab = min(tokenizer.get_vocab_size(with_added_tokens=True), vocab_rows)
    strings = ["" if i in special else tokenizer.decode([i], skip_special_tokens=False) for i in range(vocab)]
    # Tokens that are partial UTF-8 sequences decode to U+FFFD; never allow them.
    strings = ["" if "�" in s else s for s in strings]
    by_string: dict[str, list[int]] = {}
    for i, s in enumerate(strings):
        if s:
            by_string.setdefault(s, []).append(i)
    order = sorted((s, i) for i, s in enumerate(strings) if s)
    word_ids, word_lead = [], []
    for i, s in enumerate(strings):
        lead = _LEADING_WORD.match(s).end()
        if lead:
            word_ids.append(i)
            word_lead.append(lead)
    return LanguageModel(
        name,
        revision,
        model,
        tokenizer,
        strings,
        by_string,
        [s for s, _ in order],
        np.array([i for _, i in order], dtype=np.int64),
        np.array(word_ids, dtype=np.int64),
        np.array(word_lead, dtype=np.int64),
        parameters,
        size_bytes,
        device,
        weights,
    )


def _shared_prefix_cache(lm: LanguageModel, prefix: str):
    """KV cache for a prompt prefix shared by every request (instructions + examples)."""
    if prefix not in lm._prefix_cache:
        ids = lm.encode(prefix)
        state = lm.backend.prefix(ids)
        lm.stats.forward_calls += 1
        lm.stats.prompt_tokens += len(ids)
        lm._prefix_cache = {prefix: (len(ids), state)}
    return lm._prefix_cache[prefix]


def constrained_beam_search(
    lm: LanguageModel,
    requests: list[Request],
    shared_prefix: str = "",
    beam_size: int = 4,
    batch_size: int = 16,
    scan_top: int = 128,
    max_groups: int = 2,
) -> list[str | None]:
    """Return, per request, the highest-scoring text that matches its template.

    ``shared_prefix`` is prepended to every prompt and its KV cache is computed
    only once. Returns None for a request if no valid continuation was found.

    Hypotheses are pruned per template segment (``beam_size`` each), keeping the
    ``max_groups`` most advanced segments, so at most beam_size * max_groups
    hypotheses per request are alive.
    """
    results: list[str | None] = []
    for start in range(0, len(requests), batch_size):
        batch = requests[start : start + batch_size]
        results += _search_batch(lm, batch, shared_prefix, beam_size, scan_top, max_groups)
    return results


def _search_batch(lm, requests, shared_prefix, beam_size, scan_top, max_groups):
    t0 = time.perf_counter()
    n = len(requests)
    stats = lm.stats

    # ---- prompt pass: [shared prefix][left-padded request prompt] ----------
    encoded = [lm.encode(r.prompt) for r in requests]
    width = max(len(e) for e in encoded)
    input_ids = np.array([[0] * (width - len(e)) + e for e in encoded], dtype=np.int64)  # padding is masked
    attn = np.array([[0] * (width - len(e)) + [1] * len(e) for e in encoded], dtype=np.int64)
    prefix_len, prefix_state = _shared_prefix_cache(lm, shared_prefix) if shared_prefix else (0, None)
    positions = np.clip(prefix_len + attn.cumsum(1) - 1, 0, None)
    cache, logprobs = lm.backend.start(prefix_state, input_ids, attn, positions)
    stats.forward_calls += 1
    stats.prompt_tokens += sum(len(e) for e in encoded)
    stats.sequences += n
    last_pos = positions[:, -1]

    active = [_Hyp(i, i, (0, 0), "", 0.0) for i in range(n)]
    best: list[tuple[float, str] | None] = [None] * n
    max_steps = max(_template_chars(r.template) for r in requests) + 2

    for _ in range(max_steps):
        if not active:
            break
        candidates: dict[int, dict[str, tuple]] = {}
        for hyp in active:
            found = _best_valid_tokens(lm, hyp, requests[hyp.request].template, logprobs[hyp.row], beam_size, scan_top)
            bucket = candidates.setdefault(hyp.request, {})
            for token_id, state, text, score in found:
                # Different tokenizations of the same text: keep the best one.
                if text not in bucket or bucket[text][3] < score:
                    bucket[text] = (hyp.row, token_id, state, score)

        new_active, rows, next_tokens = [], [], []
        for req, bucket in candidates.items():
            template_len = len(requests[req].template)
            ranked = sorted(bucket.items(), key=lambda kv: -kv[1][3])
            groups: dict[int, list] = {}
            for text, (row, token_id, state, score) in ranked:
                if state[0] == template_len:
                    if best[req] is None or score > best[req][0]:
                        best[req] = (score, text)
                    continue
                # Hypotheses only compete with others at the same template segment. Tokens cover
                # different amounts of text, so a hypothesis that has paid for a whole word must
                # not be pruned by one that has only emitted a cheap fragment of it.
                group = groups.setdefault(state[0], [])
                if len(group) < beam_size:
                    group.append((text, row, token_id, state, score))
            # Keep the most advanced segments; laggards are mostly paths that emit fragments.
            chosen = [groups[seg] for seg in sorted(groups, reverse=True)[:max_groups]]
            # Scores only decrease, so stop once no open hypothesis can beat the best finished one.
            keep = [k for g in chosen for k in g if best[req] is None or k[4] > best[req][0]]
            for text, row, token_id, state, score in keep:
                new_active.append(_Hyp(req, len(rows), state, text, score))
                rows.append(row)
                next_tokens.append(token_id)
        if not new_active:
            break
        # Forced tokens often keep every beam in place: don't copy the entire
        # KV cache just to select its existing rows in their existing order.
        index = None
        if rows != list(range(len(active))):
            index = np.array(rows, dtype=np.int64)
            last_pos = last_pos[index]
        last_pos = last_pos + 1
        logprobs = lm.backend.step(cache, index, np.array(next_tokens, dtype=np.int64), last_pos)
        stats.forward_calls += 1
        stats.generated_tokens += len(rows)
        active = new_active

    stats.seconds += time.perf_counter() - t0
    return [b[1] if b else None for b in best]


def _forced_candidates(lm: LanguageModel, text: str, template_ends: bool) -> np.ndarray:
    """Tokens that can start ``text``: its own prefixes, or longer tokens beginning with all of it.

    If the template ends with ``text``, a longer token always overruns it. Leaving those out
    matters: when ``text`` is a single space they are every token that starts with a space,
    and they are exactly the tokens the model ranks highest after a word.
    """
    ids = [i for j in range(1, len(text) + 1) for i in lm.ids_by_string.get(text[:j], ())]
    if template_ends:
        return np.array(ids, dtype=np.int64)
    lo = bisect.bisect_right(lm.sorted_strings, text)  # strictly longer than ``text``
    hi = bisect.bisect_left(lm.sorted_strings, text + "\U0010ffff")
    longer = lm.sorted_ids[lo:hi]
    return np.concatenate([np.array(ids, dtype=np.int64), longer]) if ids else longer


def _candidates(lm: LanguageModel, template: Template, state: tuple[int, int]) -> np.ndarray:
    """A superset of the valid next tokens. ``advance`` makes the final decision."""
    seg, pos = state
    part = template[seg]
    if isinstance(part, str):
        return _forced_candidates(lm, part[pos:], seg + 1 == len(template))
    groups = []
    if pos < part.max_chars:  # keep filling the gap: tokens starting with few enough word chars
        groups.append(lm.word_ids[lm.word_lead <= part.max_chars - pos])
    if pos >= part.min_chars:  # or close it: the next segment is always fixed text
        groups.append(_forced_candidates(lm, template[seg + 1], seg + 2 == len(template)))
    return np.concatenate(groups) if groups else np.empty(0, dtype=np.int64)


def _best_valid_tokens(lm, hyp, template, logprobs, beam_size, scan_top):
    """The ``beam_size`` most likely tokens that keep ``hyp`` consistent with its template.

    Exact: candidates are checked in descending probability order until enough fit, so a
    valid token is never missed just because the model ranks it low.
    """
    ids = _candidates(lm, template, hyp.state)
    if len(ids) == 0:
        return []
    lps = logprobs[ids]
    k = min(scan_top, len(ids))
    top_idx = np.argpartition(-lps, k - 1)[:k] if k < len(ids) else np.arange(k)
    top_idx = top_idx[np.argsort(-lps[top_idx], kind="stable")]
    found = _expand(lm, hyp, template, ids[top_idx].tolist(), lps[top_idx].tolist(), beam_size)
    if len(found) < beam_size and k < len(ids):
        order = np.argsort(-lps, kind="stable")[k:]
        found += _expand(lm, hyp, template, ids[order].tolist(), lps[order].tolist(), beam_size - len(found))
    return found


def _expand(lm, hyp, template, ids, lps, limit):
    found = []
    for token_id, lp in zip(ids, lps):
        piece = lm.token_strings[token_id]
        if not piece:
            continue
        state = advance(template, hyp.state, piece)
        if state is not None:
            found.append((token_id, state, hyp.text + piece, hyp.score + lp))
            if len(found) >= limit:
                break
    return found


def _template_chars(template: Template) -> int:
    return sum(len(p) if isinstance(p, str) else p.max_chars for p in template)
