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
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

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
    model: LlamaModel
    tokenizer: Tokenizer
    token_strings: list[str]
    ids_by_string: dict[str, list[int]]  # exact string -> token ids
    sorted_strings: list[str]  # all token strings, sorted (for prefix range lookups)
    sorted_ids: np.ndarray  # token id for each entry of sorted_strings
    word_ids: np.ndarray  # tokens starting with a word character...
    word_lead: np.ndarray  # ...and how many word characters each starts with
    device: str = "cpu"
    stats: InferenceStats = field(default_factory=InferenceStats)
    _prefix_cache: dict = field(default_factory=dict)

    @property
    def parameters(self) -> int:
        return self.model.parameters

    @property
    def size_bytes(self) -> int:
        return self.model.parameters * 4  # fp32

    def describe(self) -> dict:
        return {
            "model": self.name, "revision": self.revision,
            "parameters": self.parameters, "size_bytes": self.size_bytes,
            "attention_backend": "numpy",
        }


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)


_SAFETENSORS_DTYPES = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16}


def _read_safetensors(path: Path):
    """Yield (name, fp32 array) for every tensor in a safetensors file.

    The file is memory-mapped and parsed directly: the ``safetensors`` NumPy loader
    rejects BF16, the dtype SmolLM2 ships in. BF16 widens to fp32 exactly (it is the
    top half of an fp32), as Transformers does when loading in float32.
    """
    with open(path, "rb") as f:
        size = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(size))
    data = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + size)
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        raw = data[start:end].view(_SAFETENSORS_DTYPES[info["dtype"]]).reshape(info["shape"])
        if info["dtype"] == "BF16":
            yield name, (raw.astype(np.uint32) << 16).view(np.float32)
        else:
            yield name, raw.astype(np.float32)


def _rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * weight


def _rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Llama rotary embedding (rotate-half layout); cos/sin cover one half of head_dim."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


class _KVCache:
    """Keys/values of every row, plus an optional prefix that all rows share.

    Per-row buffers are (kv_heads, rows, capacity, head_dim) and grow by whole chunks, so
    a step writes only its new positions. The shared prefix (kv_heads, P, head_dim) per
    layer is never copied into the rows: every row attends to the same arrays, which
    also turns prefix attention into one large matrix product per head. A beam reorder
    gathers only the per-row part. ``valid`` marks real (non-padding) row positions.
    """

    def __init__(self, layers: int, kv_heads: int, head_dim: int, rows: int, prefix: list | None = None):
        self.prefix = prefix
        self.length = 0
        self.k = [np.empty((kv_heads, rows, 0, head_dim), np.float32) for _ in range(layers)]
        self.v = [np.empty_like(a) for a in self.k]
        self.valid = np.zeros((rows, 0), bool)

    def _realloc(self, index: np.ndarray | None, capacity: int) -> None:
        kv, rows, _, hd = self.k[0].shape
        n = rows if index is None else len(index)
        length = self.length
        for buffers in (self.k, self.v):
            for i, old in enumerate(buffers):
                new = np.empty((kv, n, capacity, hd), np.float32)
                if index is None:
                    new[:, :, :length] = old[:, :, :length]
                else:
                    np.take(old[:, :, :length], index, axis=1, out=new[:, :, :length], mode="clip")
                buffers[i] = new
        valid = np.zeros((n, capacity), bool)
        valid[:, :length] = self.valid[:, :length] if index is None else self.valid[index, :length]
        self.valid = valid

    def reserve(self, new: int) -> None:
        if self.length + new > self.valid.shape[1]:
            self._realloc(None, self.length + new + 32)

    def reorder(self, index: np.ndarray) -> None:
        self._realloc(index, self.valid.shape[1])


class LlamaModel:
    """Llama-architecture causal LM (SmolLM2, etc.) as a NumPy fp32 forward pass.

    Q/K/V and gate/up projections are fused into one matrix each, so every layer does
    four BLAS matrix products over all rows and positions; attention uses grouped KV
    heads without repeating them.
    """

    def __init__(self, config: dict, tensors: dict[str, np.ndarray]):
        if config.get("model_type") != "llama":
            raise NotImplementedError(f"the NumPy backend runs Llama models, not {config.get('model_type')!r}")
        if config.get("rope_scaling") or config.get("attention_bias") or config.get("mlp_bias") or config.get("hidden_act", "silu") != "silu":
            raise NotImplementedError("rope scaling, attention/MLP biases and non-SiLU activations are not supported")
        self.parameters = sum(t.size for t in tensors.values())
        dim = config["hidden_size"]
        self.heads = config["num_attention_heads"]
        self.kv_heads = config.get("num_key_value_heads") or self.heads
        self.head_dim = config.get("head_dim") or dim // self.heads
        self.eps = float(config["rms_norm_eps"])
        theta = float(config.get("rope_theta", 10000.0))
        self.inv_freq = (1.0 / theta ** (np.arange(0, self.head_dim, 2) / self.head_dim)).astype(np.float32)
        self.embed = tensors.pop("model.embed_tokens.weight")
        self.norm = tensors.pop("model.norm.weight")
        self.lm_head = tensors.pop("lm_head.weight", self.embed)  # tied by default (SmolLM2)
        self.layers = []
        for i in range(config["num_hidden_layers"]):
            p = lambda n: tensors.pop(f"model.layers.{i}.{n}.weight")  # noqa: E731
            self.layers.append((
                p("input_layernorm"),
                np.concatenate([p("self_attn.q_proj"), p("self_attn.k_proj"), p("self_attn.v_proj")]),
                p("self_attn.o_proj"),
                p("post_attention_layernorm"),
                np.concatenate([p("mlp.gate_proj"), p("mlp.up_proj")]),
                p("mlp.down_proj"),
            ))

    @property
    def vocab_size(self) -> int:
        return self.lm_head.shape[0]

    def new_cache(self, rows: int, prefix: list | None = None) -> _KVCache:
        return _KVCache(len(self.layers), self.kv_heads, self.head_dim, rows, prefix)

    def prefix_states(self, cache: _KVCache) -> list:
        """A batch-1 cache's states as a prefix that other caches share."""
        n = cache.length
        return [(k[:, 0, :n].copy(), v[:, 0, :n].copy()) for k, v in zip(cache.k, cache.v)]

    def forward(self, ids: np.ndarray, positions: np.ndarray, cache: _KVCache, valid: np.ndarray | None = None) -> np.ndarray:
        """Run (rows, T) tokens after ``cache`` and append them; return the last position's logits.

        ``positions`` are RoPE positions; ``valid`` marks real tokens (left padding is False).
        A token attends to the shared prefix, its row's earlier valid tokens, and itself.
        """
        rows, t = ids.shape
        kv, hd, groups = self.kv_heads, self.head_dim, self.heads // self.kv_heads
        cache.reserve(t)
        start, stop = cache.length, cache.length + t
        cache.valid[:, start:stop] = True if valid is None else valid
        key_pos, query_pos = np.arange(stop), np.arange(start, stop)[:, None]
        visible = (cache.valid[:, None, :stop] & (key_pos <= query_pos)) | (key_pos == query_pos)
        bias = np.where(visible, np.float32(0), np.float32(-np.inf))[None, :, None]  # (1, rows, 1, T, keys)
        freqs = positions[..., None].astype(np.float32) * self.inv_freq
        cos, sin = np.cos(freqs)[:, :, None], np.sin(freqs)[:, :, None]
        scale = np.float32(hd ** -0.5)
        x = self.embed[ids.reshape(-1)]  # (rows * T, dim): 2-D, so each projection is one GEMM
        for layer, (ln1, wqkv, wo, ln2, wgu, wd) in enumerate(self.layers):
            h = (_rms_norm(x, ln1, self.eps) @ wqkv.T).reshape(rows, t, -1, hd)
            q = _rope(h[:, :, : self.heads], cos, sin) * scale
            cache.k[layer][:, :, start:stop] = _rope(h[:, :, self.heads : self.heads + kv], cos, sin).transpose(2, 0, 1, 3)
            cache.v[layer][:, :, start:stop] = h[:, :, self.heads + kv :].transpose(2, 0, 1, 3)
            q = q.reshape(rows, t, kv, groups, hd).transpose(2, 0, 3, 1, 4).reshape(kv, rows, groups * t, hd)
            prefix = cache.prefix[layer] if cache.prefix else None
            a = _attention(q, cache.k[layer][:, :, :stop], cache.v[layer][:, :, :stop], bias, prefix, groups)
            x = x + a.reshape(kv, rows, groups, t, hd).transpose(1, 3, 0, 2, 4).reshape(rows * t, -1) @ wo.T
            gate, up = np.split(_rms_norm(x, ln2, self.eps) @ wgu.T, 2, axis=-1)
            x = x + (gate * (0.5 + 0.5 * np.tanh(0.5 * gate)) * up) @ wd.T  # silu = x * sigmoid(x)
        cache.length = stop
        return _rms_norm(x.reshape(rows, t, -1)[:, -1], self.norm, self.eps) @ self.lm_head.T


def _attention(q, k, v, bias, prefix, groups):
    """Softmax attention of q (kv, rows, groups*T, d) over the shared prefix and per-row keys."""
    kv, rows, gt, d = q.shape
    s = q @ k.swapaxes(-1, -2)  # (kv, rows, groups*T, keys)
    s = (s.reshape(kv, rows, groups, gt // groups, -1) + bias).reshape(kv, rows, gt, -1)
    m = s.max(-1, keepdims=True)
    if prefix is not None:
        pk, pv = prefix
        sp = (q.reshape(kv, rows * gt, d) @ pk.swapaxes(-1, -2)).reshape(kv, rows, gt, -1)
        m = np.maximum(m, sp.max(-1, keepdims=True))
        np.exp(sp - m, out=sp)
    np.exp(s - m, out=s)
    total = s.sum(-1, keepdims=True)
    out = s @ v
    if prefix is not None:
        total += sp.sum(-1, keepdims=True)
        out += (sp.reshape(kv, rows * gt, -1) @ pv).reshape(kv, rows, gt, d)
    return out / total


def _model_dir(name: str, revision: str) -> Path:
    if Path(name).is_dir():
        return Path(name)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(name, revision=revision, allow_patterns=["config.json", "tokenizer.json", "*.safetensors"]))


def load_model(
    name: str = DEFAULT_MODEL,
    revision: str | None = None,
    threads: int | None = None,
    seed: int = 0,
    device: str | None = None,
) -> LanguageModel:
    """Load a Llama-architecture causal LM in fp32 for NumPy inference on the CPU.

    ``revision`` defaults to the pinned commit for the default model. ``threads`` sets
    the BLAS thread count (default: half the CPUs).
    """
    if device not in (None, "cpu"):
        raise ValueError(f"the NumPy backend runs on the CPU only, not {device!r}")
    from threadpoolctl import threadpool_limits

    revision = revision or (DEFAULT_REVISION if name == DEFAULT_MODEL else "main")
    set_seed(seed)
    threadpool_limits(threads or max(1, (os.cpu_count() or 2) // 2), user_api="blas")
    path = _model_dir(name, revision)
    tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
    tensors = {k: v for f in sorted(path.glob("*.safetensors")) for k, v in _read_safetensors(f)}
    model = LlamaModel(json.loads((path / "config.json").read_text()), tensors)
    special = {i for i, t in tokenizer.get_added_tokens_decoder().items() if t.special}
    vocab = min(tokenizer.get_vocab_size(with_added_tokens=True), model.vocab_size)
    strings = tokenizer.decode_batch([[i] for i in range(vocab)], skip_special_tokens=False)
    # Tokens that are partial UTF-8 sequences decode to U+FFFD; never allow them.
    strings = ["" if i in special or "\ufffd" in s else s for i, s in enumerate(strings)]
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
    )


def _shared_prefix_cache(lm: LanguageModel, prefix: str):
    """KV states for a prompt prefix shared by every request (instructions + examples)."""
    if prefix not in lm._prefix_cache:
        ids = np.array([lm.tokenizer.encode(prefix, add_special_tokens=False).ids])
        cache = lm.model.new_cache(1)
        lm.model.forward(ids, np.arange(ids.shape[1])[None], cache)
        lm.stats.forward_calls += 1
        lm.stats.prompt_tokens += ids.shape[1]
        lm._prefix_cache = {prefix: (ids.shape[1], lm.model.prefix_states(cache))}
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
    model = lm.model
    n = len(requests)
    stats = lm.stats

    # ---- prompt pass: [shared prefix][left-padded request prompt] ----------
    encoded = [e.ids for e in lm.tokenizer.encode_batch([r.prompt for r in requests], add_special_tokens=False)]
    width = max(len(e) for e in encoded)
    input_ids = np.zeros((n, width), dtype=np.int64)  # padding is masked out, so its id is irrelevant
    attn = np.zeros((n, width), dtype=bool)
    for row, e in enumerate(encoded):
        input_ids[row, width - len(e) :] = e
        attn[row, width - len(e) :] = True
    prefix_len, prefix = _shared_prefix_cache(lm, shared_prefix) if shared_prefix else (0, None)
    cache = model.new_cache(n, prefix)
    positions = np.maximum(prefix_len + attn.cumsum(1) - 1, 0)
    logits = model.forward(input_ids, positions, cache, attn)
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
        top = logits.max(-1, keepdims=True)
        logprobs = logits - (top + np.log(np.exp(logits - top).sum(-1, keepdims=True)))
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
        if rows != list(range(len(active))):
            index = np.array(rows)
            cache.reorder(index)
            last_pos = last_pos[index]
        last_pos = last_pos + 1
        logits = model.forward(np.array(next_tokens)[:, None], last_pos[:, None], cache)
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
    ids = np.array([i for j in range(1, len(text) + 1) for i in lm.ids_by_string.get(text[:j], ())], dtype=np.int64)
    if template_ends:
        return ids
    lo = bisect.bisect_right(lm.sorted_strings, text)  # strictly longer than ``text``
    hi = bisect.bisect_left(lm.sorted_strings, text + "\U0010ffff")
    return np.concatenate([ids, lm.sorted_ids[lo:hi]])


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
    top = np.argpartition(-lps, k - 1)[:k] if k < len(ids) else np.arange(k)
    top = top[np.argsort(-lps[top], kind="stable")]
    found = _expand(lm, hyp, template, ids[top].tolist(), lps[top].tolist(), beam_size)
    if len(found) < beam_size and k < len(ids):
        rest = np.ones(len(ids), dtype=bool)
        rest[top] = False
        rest = np.flatnonzero(rest)
        rest = rest[np.argsort(-lps[rest], kind="stable")]
        found += _expand(lm, hyp, template, ids[rest].tolist(), lps[rest].tolist(), beam_size - len(found))
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
