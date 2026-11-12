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

The model runs in llama.cpp (a GGUF file) by default. All beams of a batch
share one unified KV cache: a beam is a llama.cpp sequence, and forking a beam
only tags the parent's cache cells with another sequence id, so nothing is
copied. ``backend="torch"`` (``models_torch.py``, needs the ``training`` group)
runs Hugging Face checkpoints with transformers instead, for CUDA and for
checkpoints that have no GGUF.
"""

from __future__ import annotations

import bisect
import ctypes
import os
import random
import re
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .compressor import is_word_char

_LEADING_WORD = re.compile(r"\w*")
# Hugging Face checkpoint (torch backend, training) and its GGUF conversions (llama.cpp backend).
DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M"
DEFAULT_REVISION = "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"  # pinned for reproducibility
GGUF_REPO = "mradermacher/SmolLM2-360M-GGUF"  # converted from DEFAULT_MODEL's weights
GGUF_REVISION = "630c4866d716e28f45a516ed00e2882149726c13"
GGUF_QUANTS = ("f16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M")
DEFAULT_QUANT = "f16"  # Q8_0 and K-quants change a sequence's scores with the batch size (see README)


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
    row: int  # row of the parent in the current batch of sequences
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
    backend: object  # _LlamaBackend or models_torch.TorchBackend
    token_strings: list[str]
    ids_by_string: dict[str, list[int]]  # exact string -> token ids
    sorted_strings: list[str]  # all token strings, sorted (for prefix range lookups)
    sorted_ids: np.ndarray  # token id for each entry of sorted_strings
    word_ids: np.ndarray  # tokens starting with a word character...
    word_lead: np.ndarray  # ...and how many word characters each starts with
    device: str = "cpu"
    stats: InferenceStats = field(default_factory=InferenceStats)

    @property
    def parameters(self) -> int:
        return self.backend.parameters

    @property
    def size_bytes(self) -> int:
        return self.backend.size_bytes

    def describe(self) -> dict:
        return {"model": self.name, "revision": self.revision, "parameters": self.parameters,
                "size_bytes": self.size_bytes, **self.backend.describe()}


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_model(
    name: str = DEFAULT_MODEL,
    revision: str | None = None,
    threads: int | None = None,
    seed: int = 0,
    device: str | None = None,
    backend: str | None = None,
    quant: str = DEFAULT_QUANT,
    n_ctx: int = 8192,
) -> LanguageModel:
    """Load a causal LM.

    ``backend`` "llama" (llama.cpp) takes the default model, a local ``.gguf`` file, or
    ``owner/repo/file.gguf`` on the Hugging Face Hub; ``quant`` picks the default model's
    GGUF file. ``backend`` "torch" loads a Hugging Face checkpoint in fp32. By default,
    GGUF names and the default model use llama.cpp, other names and ``device="cuda"`` torch.
    ``revision`` defaults to the pinned commit for the default model.
    """
    threads = threads or max(1, (os.cpu_count() or 2) // 2)
    if backend is None:
        backend = "llama" if name.endswith(".gguf") or (name == DEFAULT_MODEL and device != "cuda") else "torch"
    set_seed(seed)
    if backend == "torch":
        from .models_torch import TorchBackend

        revision = revision or (DEFAULT_REVISION if name == DEFAULT_MODEL else "main")
        impl = TorchBackend(name, revision, threads, seed, device)
    elif backend == "llama":
        if name == DEFAULT_MODEL:
            if quant not in GGUF_QUANTS:
                raise ValueError(f"quant must be one of {GGUF_QUANTS}")
            name, revision = f"{GGUF_REPO}/SmolLM2-360M.{quant}.gguf", GGUF_REVISION
        if Path(name).is_file():
            path, revision = Path(name), "local"
        else:
            revision = revision or "main"
            path = _hub_file(name, revision)
        impl = _LlamaBackend(path, threads, device or "cpu", n_ctx)
    else:
        raise ValueError(f"unknown backend {backend!r}")
    strings = impl.token_strings()
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
        impl,
        strings,
        by_string,
        [s for s, _ in order],
        np.array([i for _, i in order], dtype=np.int64),
        np.array(word_ids, dtype=np.int64),
        np.array(word_lead, dtype=np.int64),
        impl.device,
    )


def _hub_file(name: str, revision: str) -> Path:
    """Download ``owner/repo/file`` at ``revision`` into the Hugging Face cache layout (once)."""
    repo, filename = name.rsplit("/", 1)
    hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub")
    path = hub / f"models--{repo.replace('/', '--')}" / "snapshots" / revision / filename
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
        with urllib.request.urlopen(url) as response, open(part, "wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
        part.rename(path)
    return path


@ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
def _llama_log(level, text, user_data):
    if level == 4:  # GGML_LOG_LEVEL_ERROR; llama.cpp logs every tensor at load otherwise
        os.write(2, text)


class _LlamaBackend:
    """A llama.cpp model and one context whose unified KV cache holds every sequence.

    Sequence 0 keeps the shared prompt prefix; a session's beams use the other ids.
    """

    def __init__(self, path: Path, threads: int, device: str, n_ctx: int):
        import llama_cpp

        self.llama = llama = llama_cpp
        llama.llama_log_set(_llama_log, None)
        llama.llama_backend_init()
        mp = llama.llama_model_default_params()
        if device != "cpu" and not llama.llama_supports_gpu_offload():
            raise ValueError(f"device {device!r} needs a GPU build of llama-cpp-python; use --backend torch for CUDA")
        mp.n_gpu_layers = 0 if device == "cpu" else -1
        self.model = llama.llama_model_load_from_file(str(path).encode(), mp)
        if not self.model:
            raise RuntimeError(f"llama.cpp could not load {path}")
        cp = llama.llama_context_default_params()
        cp.n_ctx = cp.n_batch = n_ctx
        cp.n_seq_max = llama.llama_max_parallel_sequences()
        cp.n_threads = cp.n_threads_batch = threads
        cp.kv_unified = True  # forked beams share their parent's cells instead of copying them
        cp.flash_attn_type = llama.LLAMA_FLASH_ATTN_TYPE_ENABLED  # skips other sequences' cells
        self.ctx = llama.llama_init_from_model(self.model, cp)
        if not self.ctx:
            raise RuntimeError("llama.cpp could not create a context")
        self.mem = llama.llama_get_memory(self.ctx)
        self.vocab = llama.llama_model_get_vocab(self.model)
        self.n_vocab = llama.llama_vocab_n_tokens(self.vocab)
        self.n_ctx, self.n_seq = n_ctx, cp.n_seq_max
        self.batch = llama.llama_batch_init(n_ctx, 0, 1)
        self.path, self.device = path, device
        self.prefix: tuple[str, int] | None = None  # prompt prefix held by sequence 0, and its length
        self.parameters = llama.llama_model_n_params(self.model)
        self.size_bytes = llama.llama_model_size(self.model)

    def describe(self) -> dict:
        desc = ctypes.create_string_buffer(128)
        self.llama.llama_model_desc(self.model, desc, len(desc))
        return {"backend": "llama.cpp", "llama_cpp_python": self.llama.__version__,
                "gguf": self.path.name, "gguf_type": desc.value.decode()}

    def token_strings(self) -> list[str]:
        buf = ctypes.create_string_buffer(256)
        out = []
        for i in range(self.n_vocab):  # control tokens (special=False) render as ""
            n = self.llama.llama_token_to_piece(self.vocab, i, buf, len(buf), 0, False)
            out.append(buf.raw[:n].decode("utf-8", errors="replace"))
        return out

    def tokenize(self, text: str) -> list[int]:
        data = text.encode()
        buf = (self.llama.llama_token * (len(data) + 1))()
        n = self.llama.llama_tokenize(self.vocab, data, len(data), buf, len(buf), False, False)
        return list(buf[:n])

    def decode(self, tokens: list[int], positions: list[int], seqs: list[int], outputs: list[bool]) -> np.ndarray:
        """Run one batch; return log-probabilities for the tokens marked in ``outputs``, in order."""
        if len(tokens) > self.n_ctx:
            raise RuntimeError(f"batch of {len(tokens)} tokens exceeds n_ctx={self.n_ctx}")
        b = self.batch
        b.n_tokens = len(tokens)
        for i, (t, p, s, o) in enumerate(zip(tokens, positions, seqs, outputs)):
            b.token[i], b.pos[i], b.n_seq_id[i], b.logits[i] = t, p, 1, o
            b.seq_id[i][0] = s
        if (rc := self.llama.llama_decode(self.ctx, b)) != 0:
            raise RuntimeError(f"llama_decode failed ({rc}); the KV cache may be full, try a larger n_ctx")
        n_out = sum(outputs)
        if not n_out:
            return np.empty((0, self.n_vocab), dtype=np.float32)
        logits = np.ctypeslib.as_array(self.llama.llama_get_logits(self.ctx), shape=(n_out, self.n_vocab))
        shifted = logits - logits.max(axis=1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))

    def start(self, prefix: str, prompts: list[list[int]], stats: InferenceStats) -> tuple["_LlamaSession", np.ndarray]:
        if self.prefix is None or self.prefix[0] != prefix:
            self.llama.llama_memory_clear(self.mem, True)
            ids = self.tokenize(prefix)
            if ids:
                self.decode(ids, list(range(len(ids))), [0] * len(ids), [False] * len(ids))
                stats.forward_calls += 1
                stats.prompt_tokens += len(ids)
            self.prefix = (prefix, len(ids))
        session = _LlamaSession(self, self.prefix[1], len(prompts))
        return session, session.prompt(prompts)


class _LlamaSession:
    """The beams of one batch of requests; row ``r`` of the last logits belongs to sequence ``seq[r]``."""

    def __init__(self, backend: _LlamaBackend, prefix_len: int, n: int):
        if n >= backend.n_seq:
            raise ValueError(f"at most {backend.n_seq - 1} requests per batch")
        self.b, self.prefix_len = backend, prefix_len
        self.seq = list(range(1, n + 1))
        self.free = list(range(backend.n_seq - 1, n, -1))
        self.pos: list[int] = []  # next position of each row

    def prompt(self, prompts: list[list[int]]) -> np.ndarray:
        tokens, positions, seqs, outputs = [], [], [], []
        for s, ids in zip(self.seq, prompts):
            if self.prefix_len:  # share the prefix cells; no copy
                self.b.llama.llama_memory_seq_cp(self.b.mem, 0, s, -1, -1)
            tokens += ids
            positions += range(self.prefix_len, self.prefix_len + len(ids))
            seqs += [s] * len(ids)
            outputs += [False] * (len(ids) - 1) + [True]
            self.pos.append(self.prefix_len + len(ids))
        return self.b.decode(tokens, positions, seqs, outputs)

    def step(self, rows: list[int], tokens: list[int]) -> np.ndarray:
        """Append ``tokens[i]`` to a fork of row ``rows[i]``; rows no one continues are dropped."""
        llama, mem = self.b.llama, self.b.mem
        kept = set(rows)
        for r, s in enumerate(self.seq):
            if r not in kept:
                llama.llama_memory_seq_rm(mem, s, -1, -1)
                self.free.append(s)
        seq, pos, taken = [], [], set()
        for r in rows:
            if r in taken:  # another child of the same parent: fork it
                s = self.free.pop()
                llama.llama_memory_seq_cp(mem, self.seq[r], s, -1, -1)
            else:  # the first child takes over its parent's sequence
                taken.add(r)
                s = self.seq[r]
            seq.append(s)
            pos.append(self.pos[r])
        logprobs = self.b.decode(tokens, pos, seq, [True] * len(tokens))
        self.seq, self.pos = seq, [p + 1 for p in pos]
        return logprobs

    def close(self) -> None:
        for s in self.seq:
            self.b.llama.llama_memory_seq_rm(self.b.mem, s, -1, -1)


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

    # ---- prompt pass: [shared prefix][request prompt] ----------------------
    encoded = [lm.backend.tokenize(r.prompt) for r in requests]
    session, logprobs = lm.backend.start(shared_prefix, encoded, stats)
    stats.forward_calls += 1
    stats.prompt_tokens += sum(len(e) for e in encoded)
    stats.sequences += n

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
        logprobs = session.step(rows, next_tokens)
        stats.forward_calls += 1
        stats.generated_tokens += len(rows)
        active = new_active

    session.close()
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
    top = np.argpartition(-lps, k - 1)[:k] if k < len(ids) else np.arange(len(ids))
    top = top[np.argsort(-lps[top], kind="stable")]
    found = _expand(lm, hyp, template, ids[top].tolist(), lps[top].tolist(), beam_size)
    if len(found) < beam_size and k < len(ids):
        order = np.argsort(-lps, kind="stable")
        order = order[~np.isin(order, top)]
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
