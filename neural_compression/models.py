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
import os
import random
import re
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    model: torch.nn.Module
    tokenizer: object
    token_strings: list[str]
    ids_by_string: dict[str, list[int]]  # exact string -> token ids
    sorted_strings: list[str]  # all token strings, sorted (for prefix range lookups)
    sorted_ids: torch.Tensor  # token id for each entry of sorted_strings
    word_ids: torch.Tensor  # tokens starting with a word character...
    word_lead: torch.Tensor  # ...and how many word characters each starts with
    device: torch.device = torch.device("cpu")
    stats: InferenceStats = field(default_factory=InferenceStats)
    _prefix_cache: dict = field(default_factory=dict)

    @property
    def parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @property
    def size_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.model.parameters())

    def describe(self) -> dict:
        return {"model": self.name, "revision": self.revision, "parameters": self.parameters, "size_bytes": self.size_bytes}


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model(
    name: str = DEFAULT_MODEL,
    revision: str | None = None,
    threads: int | None = None,
    seed: int = 0,
    device: str | None = None,
) -> LanguageModel:
    """Load a causal LM in fp32 on ``device`` (default: CUDA if available, else CPU).

    ``revision`` defaults to the pinned commit for the default model.
    """
    revision = revision or (DEFAULT_REVISION if name == DEFAULT_MODEL else "main")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_seed(seed)
    torch.set_num_threads(threads or max(1, (os.cpu_count() or 2) // 2))
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=torch.float32).eval().to(dev)
    special = set(tokenizer.all_special_ids)
    vocab = min(len(tokenizer), model.get_output_embeddings().weight.shape[0])
    strings = ["" if i in special else tokenizer.decode([i]) for i in range(vocab)]
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
        torch.tensor([i for _, i in order]),
        torch.tensor(word_ids),
        torch.tensor(word_lead),
        dev,
    )


def _shared_prefix_cache(lm: LanguageModel, prefix: str):
    """KV cache for a prompt prefix shared by every request (instructions + examples)."""
    if prefix not in lm._prefix_cache:
        ids = lm.tokenizer(prefix, add_special_tokens=False, return_tensors="pt").input_ids
        with torch.no_grad():
            out = lm.model(input_ids=ids.to(lm.device), use_cache=True)
        lm.stats.forward_calls += 1
        lm.stats.prompt_tokens += ids.shape[1]
        lm._prefix_cache = {prefix: (ids.shape[1], out.past_key_values)}
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


@torch.no_grad()
def _search_batch(lm, requests, shared_prefix, beam_size, scan_top, max_groups):
    t0 = time.perf_counter()
    tok, model = lm.tokenizer, lm.model
    n = len(requests)
    stats = lm.stats

    # ---- prompt pass: [shared prefix][left-padded request prompt] ----------
    encoded = [tok(r.prompt, add_special_tokens=False).input_ids for r in requests]
    width = max(len(e) for e in encoded)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    dev = lm.device
    input_ids = torch.tensor([[pad_id] * (width - len(e)) + e for e in encoded], device=dev)
    attn = torch.tensor([[0] * (width - len(e)) + [1] * len(e) for e in encoded], device=dev)
    cache = None
    if shared_prefix:
        prefix_len, prefix_cache = _shared_prefix_cache(lm, shared_prefix)
        cache = _copy_cache(prefix_cache, n)
        attn = torch.cat([torch.ones(n, prefix_len, dtype=attn.dtype, device=dev), attn], dim=1)
    positions = (attn.cumsum(1) - 1).clamp(min=0)
    out = model(input_ids=input_ids, attention_mask=attn, position_ids=positions[:, -width:], past_key_values=cache, use_cache=True)
    stats.forward_calls += 1
    stats.prompt_tokens += sum(len(e) for e in encoded)
    stats.sequences += n
    cache, logits = out.past_key_values, out.logits[:, -1]
    last_pos = positions[:, -1]

    active = [_Hyp(i, i, (0, 0), "", 0.0) for i in range(n)]
    best: list[tuple[float, str] | None] = [None] * n
    max_steps = max(_template_chars(r.template) for r in requests) + 2

    for _ in range(max_steps):
        if not active:
            break
        # Token bookkeeping happens on the CPU; only the model runs on ``dev``.
        logprobs = torch.log_softmax(logits.float(), dim=-1).cpu()
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
        index = torch.tensor(rows, device=dev)
        cache.reorder_cache(index)
        attn = torch.cat([attn[index], torch.ones(len(rows), 1, dtype=attn.dtype, device=dev)], dim=1)
        last_pos = last_pos[index] + 1
        out = model(
            input_ids=torch.tensor(next_tokens, device=dev)[:, None],
            attention_mask=attn,
            position_ids=last_pos[:, None],
            past_key_values=cache,
            use_cache=True,
        )
        stats.forward_calls += 1
        stats.generated_tokens += len(rows)
        cache, logits = out.past_key_values, out.logits[:, -1]
        active = new_active

    stats.seconds += time.perf_counter() - t0
    return [b[1] if b else None for b in best]


def _forced_candidates(lm: LanguageModel, text: str) -> torch.Tensor:
    """Tokens that can start ``text``: its own prefixes, or longer tokens beginning with all of it."""
    ids = [i for j in range(1, len(text) + 1) for i in lm.ids_by_string.get(text[:j], ())]
    lo = bisect.bisect_right(lm.sorted_strings, text)  # strictly longer than ``text``
    hi = bisect.bisect_left(lm.sorted_strings, text + "\U0010ffff")
    longer = lm.sorted_ids[lo:hi]
    return torch.cat([torch.tensor(ids, dtype=torch.long), longer]) if ids else longer


def _candidates(lm: LanguageModel, template: Template, state: tuple[int, int]) -> torch.Tensor:
    """A superset of the valid next tokens. ``advance`` makes the final decision."""
    seg, pos = state
    part = template[seg]
    if isinstance(part, str):
        return _forced_candidates(lm, part[pos:])
    groups = []
    if pos < part.max_chars:  # keep filling the gap: tokens starting with few enough word chars
        groups.append(lm.word_ids[lm.word_lead <= part.max_chars - pos])
    if pos >= part.min_chars:  # or close it: the next segment is always fixed text
        groups.append(_forced_candidates(lm, template[seg + 1]))
    return torch.cat(groups) if groups else torch.empty(0, dtype=torch.long)


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
    top_lp, top_idx = lps.topk(k)
    found = _expand(lm, hyp, template, ids[top_idx].tolist(), top_lp.tolist(), beam_size)
    if len(found) < beam_size and k < len(ids):
        order = lps.argsort(descending=True)[k:]
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


def _copy_cache(cache, n: int):
    """Repeat a batch-1 cache n times (deep copy so the shared cache stays intact)."""
    import copy

    clone = copy.deepcopy(cache)
    clone.batch_repeat_interleave(n)
    return clone
