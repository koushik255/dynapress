"""PyTorch + transformers backend for :mod:`models` (needs the ``training`` dependency group).

Runs any Hugging Face causal LM checkpoint in fp32, on CUDA or the CPU. The
default backend is llama.cpp; this one serves CUDA runs, LoRA-merged
checkpoints, and other models that have no GGUF file.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache, DynamicLayer


def _cpu_gqa_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs):
    """Use native CPU grouped attention without materializing repeated KV heads.

    The standard masked SDPA path repeats KV heads before calling the kernel.
    PyTorch >= 2.15's CPU flash kernel supports masked GQA directly instead.
    This backend is only selected for CPU Llama inference, not training/CUDA.
    """
    causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
    causal = query.shape[2] > 1 and attention_mask is None and causal
    output = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=dropout,
        scale=scaling, is_causal=causal, enable_gqa=True,
    )
    return output.transpose(1, 2).contiguous(), None


class TorchBackend:
    def __init__(self, name: str, revision: str, threads: int, seed: int, device: str | None):
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(seed)
        torch.set_num_threads(threads)
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=torch.float32).eval().to(dev)
        if dev.type == "cpu" and model.config.model_type == "llama" and tuple(map(int, torch.__version__.split(".")[:2])) >= (2, 15):
            # Older Transformers versions lack the public custom-attention registry.
            try:
                from transformers import AttentionInterface, AttentionMaskInterface
            except ImportError:
                pass
            else:
                AttentionInterface.register("cpu_gqa", _cpu_gqa_attention)
                AttentionMaskInterface.register("cpu_gqa", AttentionMaskInterface()["sdpa"])
                model.set_attn_implementation("cpu_gqa")
        self.model, self.dev, self.device = model, dev, str(dev)
        self._prefix_cache: dict = {}

    @property
    def parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @property
    def size_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.model.parameters())

    def describe(self) -> dict:
        return {"backend": "torch", "torch": torch.__version__, "attention_backend": self.model.config._attn_implementation}

    def token_strings(self) -> list[str]:
        tok = self.tokenizer
        special = set(tok.all_special_ids)
        vocab = min(len(tok), self.model.get_output_embeddings().weight.shape[0])
        return ["" if i in special else tok.decode([i]) for i in range(vocab)]

    def tokenize(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False).input_ids

    @torch.no_grad()
    def _shared_prefix_cache(self, prefix: str, stats):
        """KV cache for a prompt prefix shared by every request (instructions + examples)."""
        if prefix not in self._prefix_cache:
            ids = self.tokenizer(prefix, add_special_tokens=False, return_tensors="pt").input_ids
            out = self.model(input_ids=ids.to(self.dev), use_cache=True)
            stats.forward_calls += 1
            stats.prompt_tokens += ids.shape[1]
            self._prefix_cache = {prefix: (ids.shape[1], out.past_key_values)}
        return self._prefix_cache[prefix]

    @torch.no_grad()
    def start(self, prefix: str, encoded: list[list[int]], stats) -> tuple["_TorchSession", np.ndarray]:
        """Prompt pass: [shared prefix][left-padded request prompt]."""
        n, dev = len(encoded), self.dev
        width = max(len(e) for e in encoded)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        input_ids = torch.tensor([[pad_id] * (width - len(e)) + e for e in encoded], device=dev)
        attn = torch.tensor([[0] * (width - len(e)) + [1] * len(e) for e in encoded], device=dev)
        cache = _beam_cache()
        if prefix:
            prefix_len, prefix_cache = self._shared_prefix_cache(prefix, stats)
            cache = _beam_cache(prefix_cache, n)
            attn = torch.cat([torch.ones(n, prefix_len, dtype=attn.dtype, device=dev), attn], dim=1)
        positions = (attn.cumsum(1) - 1).clamp(min=0)
        out = self.model(input_ids=input_ids, attention_mask=attn, position_ids=positions[:, -width:], past_key_values=cache, use_cache=True)
        session = _TorchSession(self, out.past_key_values, attn, positions[:, -1])
        return session, _logprobs(out.logits[:, -1])


class _TorchSession:
    def __init__(self, backend: TorchBackend, cache, attn, last_pos):
        self.b, self.cache, self.attn, self.last_pos = backend, cache, attn, last_pos
        self.rows = len(attn)

    @torch.no_grad()
    def step(self, rows: list[int], tokens: list[int]) -> np.ndarray:
        dev = self.b.dev
        # Forced tokens often keep every beam in place: don't copy the entire
        # KV cache just to select its existing rows in their existing order.
        if rows != list(range(self.rows)):
            index = torch.tensor(rows, device=dev)
            self.cache.reorder_cache(index)
            self.attn = self.attn[index]
            self.last_pos = self.last_pos[index]
        self.attn = torch.cat([self.attn, torch.ones(len(rows), 1, dtype=self.attn.dtype, device=dev)], dim=1)
        self.last_pos = self.last_pos + 1
        out = self.b.model(
            input_ids=torch.tensor(tokens, device=dev)[:, None],
            attention_mask=self.attn,
            position_ids=self.last_pos[:, None],
            past_key_values=self.cache,
            use_cache=True,
        )
        self.cache, self.rows = out.past_key_values, len(rows)
        return _logprobs(out.logits[:, -1])

    def close(self) -> None:
        self.cache = None


def _logprobs(logits: torch.Tensor) -> np.ndarray:
    # Token bookkeeping happens on the CPU; only the model runs on the device.
    return torch.log_softmax(logits.float(), dim=-1).cpu().numpy()


class _BeamLayer(DynamicLayer):
    """KV cache layer for beam search that appends in place.

    DynamicLayer appends with torch.cat, which copies the whole cache of every layer on
    every step. Here ``keys``/``values`` are views into buffers with spare positions, so
    an append writes only the new token. A beam reorder still gathers every row into new
    buffers sized to the beam, one copy as before, so memory follows the beam width.
    """

    def __init__(self, keys: torch.Tensor | None = None, values: torch.Tensor | None = None, rows: int = 1):
        super().__init__()
        if keys is not None:  # batch-1 shared-prefix states, repeated for ``rows`` rows
            self.lazy_initialization(keys, values)
            self._store(keys.expand(rows, -1, -1, -1), values.expand(rows, -1, -1, -1))

    def _store(self, keys: torch.Tensor, values: torch.Tensor, capacity: int = 0, index: torch.Tensor | None = None) -> None:
        """Copy ``keys``/``values`` (rows ``index`` only, if given) into new buffers."""
        rows = len(index) if index is not None else keys.shape[0]
        _, heads, length, dim = keys.shape
        shape = (rows, heads, max(capacity, length + 16), dim)
        self._k, self._v = (torch.empty(shape, dtype=self.dtype, device=self.device) for _ in range(2))
        self.keys, self.values = self._k[:, :, :length], self._v[:, :, :length]
        for src, dst in ((keys, self.keys), (values, self.values)):
            if index is None:
                dst.copy_(src)
            else:
                torch.index_select(src, 0, index, out=dst)

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
            self._store(key_states[..., :0, :], value_states[..., :0, :])
        length, new = self.keys.shape[-2], key_states.shape[-2]
        if length + new > self._k.shape[-2]:
            self._store(self.keys, self.values, length + new + 16)
        self._k[:, :, length : length + new] = key_states
        self._v[:, :, length : length + new] = value_states
        self.keys, self.values = self._k[:, :, : length + new], self._v[:, :, : length + new]
        return self.keys, self.values

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        self._store(self.keys, self.values, self._k.shape[-2], beam_idx)


def _beam_cache(prefix_cache=None, n: int = 1) -> Cache:
    """Empty beam cache, or one starting from a batch-1 prefix cache repeated n times (left intact)."""
    if prefix_cache is None:
        return Cache(layer_class_to_replicate=_BeamLayer)
    return Cache(layers=[_BeamLayer(layer.keys, layer.values, n) for layer in prefix_cache.layers])
