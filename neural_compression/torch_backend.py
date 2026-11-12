"""PyTorch + Transformers backend for constrained beam search (CUDA, and any HF causal LM).

Needs the ``torch`` dependency group (``uv sync --group torch``). The default CPU
backend is ONNX Runtime (``onnx_llama.py``); this one is used for CUDA and for models
that backend does not support.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM
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
    def __init__(self, name: str, revision: str, device: str, threads: int):
        self.device = torch.device(device)
        torch.set_num_threads(threads)
        model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=torch.float32).eval().to(self.device)
        if self.device.type == "cpu" and model.config.model_type == "llama" and tuple(map(int, torch.__version__.split(".")[:2])) >= (2, 15):
            # Older Transformers versions lack the public custom-attention registry.
            try:
                from transformers import AttentionInterface, AttentionMaskInterface
            except ImportError:
                pass
            else:
                AttentionInterface.register("cpu_gqa", _cpu_gqa_attention)
                AttentionMaskInterface.register("cpu_gqa", AttentionMaskInterface()["sdpa"])
                model.set_attn_implementation("cpu_gqa")
        self.model = model
        self.vocab_rows = model.get_output_embeddings().weight.shape[0]
        self.parameters = sum(p.numel() for p in model.parameters())

    @property
    def attention_backend(self) -> str:
        return self.model.config._attn_implementation

    def _logprobs(self, out) -> np.ndarray:
        return torch.log_softmax(out.logits[:, -1].float(), dim=-1).cpu().numpy()

    @torch.no_grad()
    def prefix(self, ids: list[int]):
        return self.model(input_ids=torch.tensor([ids], device=self.device), use_cache=True).past_key_values

    @torch.no_grad()
    def start(self, prefix, input_ids: np.ndarray, attn: np.ndarray, positions: np.ndarray):
        n, dev = len(input_ids), self.device
        attn = torch.from_numpy(attn).to(dev)
        cache = _beam_cache()
        if prefix is not None:
            cache = _beam_cache(prefix, n)
            attn = torch.cat([torch.ones(n, prefix.get_seq_length(), dtype=attn.dtype, device=dev), attn], dim=1)
        out = self.model(input_ids=torch.from_numpy(input_ids).to(dev), attention_mask=attn,
                         position_ids=torch.from_numpy(positions).to(dev), past_key_values=cache, use_cache=True)
        return {"cache": out.past_key_values, "attn": attn}, self._logprobs(out)

    @torch.no_grad()
    def step(self, state: dict, rows: np.ndarray | None, tokens: np.ndarray, positions: np.ndarray) -> np.ndarray:
        dev, attn = self.device, state["attn"]
        if rows is not None:
            index = torch.from_numpy(rows).to(dev)
            state["cache"].reorder_cache(index)
            attn = attn[index]
        attn = torch.cat([attn, torch.ones(len(tokens), 1, dtype=attn.dtype, device=dev)], dim=1)
        out = self.model(input_ids=torch.from_numpy(tokens).to(dev)[:, None], attention_mask=attn,
                         position_ids=torch.from_numpy(positions).to(dev)[:, None], past_key_values=state["cache"], use_cache=True)
        state["cache"], state["attn"] = out.past_key_values, attn
        return self._logprobs(out)


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
