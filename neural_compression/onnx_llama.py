"""Llama-architecture inference on ONNX Runtime, without torch or the onnx package.

The graph is written straight to ONNX protobuf from the model's ``config.json``. Weights
are not part of it: they are graph inputs read from the checkpoint's ``model.safetensors``
(bf16/fp16 upcast to fp32 exactly as ``from_pretrained(dtype=float32)`` does).

One forward pass attends over three key/value blocks instead of one concatenated cache:

* ``prefix``: the shared prompt prefix, batch 1, broadcast to every row (never copied),
* ``cache``: each row's earlier tokens, a buffer with spare positions that the caller
  appends to in place (masked positions contribute exactly zero),
* the step's own new tokens.

It returns log-probabilities for each row's last position and the new keys/values, which
the caller writes into ``cache``. Keys are stored transposed (``[B, kv_heads, head_dim,
positions]``) so the attention matmuls read them without a transpose copy.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

NEG = -1e9  # additive mask value; finite so a fully masked (padding) row stays NaN-free
_FLOAT, _INT64 = 1, 7


# ---- minimal protobuf writer for the ONNX messages used here ------------------------

def _varint(n: int) -> bytes:
    n &= (1 << 64) - 1  # int64 two's complement
    out = bytearray()
    while n > 0x7F:
        out.append(n & 0x7F | 0x80)
        n >>= 7
    out.append(n)
    return bytes(out)


def _msg(*fields) -> bytes:
    """Encode (field number, value) pairs; a list value is a repeated field."""
    out = bytearray()
    for num, value in fields:
        for v in value if isinstance(value, list) else [value]:
            if isinstance(v, int):
                out += _varint(num << 3) + _varint(v)
            elif isinstance(v, float):
                out += _varint(num << 3 | 5) + struct.pack("<f", v)
            else:
                v = v.encode() if isinstance(v, str) else v
                out += _varint(num << 3 | 2) + _varint(len(v)) + v
    return bytes(out)


def _value_info(name: str, elem_type: int, dims: list) -> bytes:
    dim = [_msg((2, d) if isinstance(d, str) else (1, d)) for d in dims]
    tensor_type = _msg((1, elem_type), (2, _msg((1, dim))))
    return _msg((1, name), (2, _msg((1, tensor_type))))


def _attribute(name: str, value) -> bytes:
    if isinstance(value, float):
        return _msg((1, name), (20, 1), (2, value))
    if isinstance(value, int):
        return _msg((1, name), (20, 2), (3, value))
    return _msg((1, name), (20, 7), (8, list(value)))


class _Graph:
    def __init__(self):
        self.nodes: list[bytes] = []
        self.initializers: list[bytes] = []
        self.count = 0

    def op(self, op_type: str, *inputs: str, outputs: int = 1, name: str | None = None, **attrs):
        self.count += 1
        outs = [name] if name else [f"t{self.count}_{i}" for i in range(outputs)]
        attributes = [_attribute(k, v) for k, v in attrs.items()]
        self.nodes.append(_msg((1, list(inputs)), (2, outs), (4, op_type), (5, attributes)))
        return outs[0] if outputs == 1 else outs

    def const(self, values, dtype=np.int64) -> str:
        a = np.asarray(values, dtype=dtype)
        self.count += 1
        name = f"c{self.count}"
        elem = _INT64 if a.dtype == np.int64 else _FLOAT
        self.initializers.append(_msg((1, list(a.shape)), (2, elem), (8, name), (9, a.tobytes())))
        return name


def build_graph(config: dict, weight_shapes: dict[str, tuple]) -> bytes:
    """ONNX model bytes for a Llama-style config. Weight names match the HF checkpoint."""
    E, layers = config["hidden_size"], config["num_hidden_layers"]
    H, Hkv = config["num_attention_heads"], config.get("num_key_value_heads") or config["num_attention_heads"]
    D = config.get("head_dim") or E // H
    eps = float(config["rms_norm_eps"])
    g = _Graph()

    def norm(x, weight):
        return g.op("RMSNormalization", x, weight, axis=-1, epsilon=eps)

    def linear(x, weight):
        return g.op("Gemm", x, weight, transB=1)

    ids, pos, bias = "input_ids", "position_ids", "bias"
    x = g.op("Reshape", g.op("Gather", "model.embed_tokens.weight", ids), g.const([-1, E]))
    rope_cfg = config.get("rope_parameters") or config.get("rope_scaling") or {}  # Transformers 5 moved rope_theta here
    if rope_cfg.get("rope_type", rope_cfg.get("type", "default")) != "default":
        raise ValueError(f"unsupported RoPE scaling {rope_cfg}")
    theta = config.get("rope_theta") or rope_cfg.get("rope_theta", 10000.0)
    inv_freq = 1.0 / (np.float32(theta) ** (np.arange(0, D, 2, dtype=np.float32) / np.float32(D)))
    freqs = g.op("Mul", g.op("Unsqueeze", g.op("Cast", pos, to=_FLOAT), g.const([-1])), g.const(inv_freq, np.float32))
    angles = g.op("Concat", freqs, freqs, axis=-1)
    cos, sin = (g.op("Unsqueeze", g.op(f, angles), g.const([1])) for f in ("Cos", "Sin"))

    def rope(t):  # HF rotate_half convention
        lo, hi = g.op("Split", t, axis=-1, num_outputs=2, outputs=2)
        return g.op("Add", g.op("Mul", t, cos), g.op("Mul", g.op("Concat", g.op("Neg", hi), lo, axis=-1), sin))

    batch_time = g.op("Shape", ids)
    to_heads = {n: g.op("Concat", batch_time, g.const([n, D]), axis=0) for n in {H, Hkv}}

    def heads(t, n):  # [B*T, n*D] -> [B, n, T, D]
        return g.op("Transpose", g.op("Reshape", t, to_heads[n]), perm=[0, 2, 1, 3])

    sizes = g.op("Concat", g.op("Shape", "prefix_kT.0", start=3), g.op("Shape", "cache_kT.0", start=3),
                 g.op("Shape", ids, start=1), axis=0)
    outputs = []
    for i in range(layers):
        p = f"model.layers.{i}."
        h = norm(x, p + "input_layernorm.weight")
        q = rope(heads(linear(h, p + "self_attn.q_proj.weight"), H))
        k = rope(heads(linear(h, p + "self_attn.k_proj.weight"), Hkv))
        v = heads(linear(h, p + "self_attn.v_proj.weight"), Hkv)
        # Query heads sharing a kv head become extra rows: [B, Hkv, G*T, D].
        q = g.op("Mul", g.op("Reshape", q, g.const([0, Hkv, -1, D])), g.const(D ** -0.5, np.float32))
        kT = g.op("Transpose", k, perm=[0, 1, 3, 2], name=f"new_kT.{i}")
        v = g.op("Identity", v, name=f"new_v.{i}")
        outputs += [kT, v]
        scores = g.op("Concat", *(g.op("MatMul", q, key) for key in (f"prefix_kT.{i}", f"cache_kT.{i}", kT)), axis=-1)
        probs = g.op("Split", g.op("Softmax", g.op("Add", scores, bias), axis=-1), sizes, axis=-1, outputs=3)
        values = [g.op("MatMul", pr, val) for pr, val in zip(probs, (f"prefix_v.{i}", f"cache_v.{i}", v))]
        o = g.op("Add", g.op("Add", values[0], values[1]), values[2])
        o = g.op("Transpose", g.op("Reshape", o, g.const([0, H, -1, D])), perm=[0, 2, 1, 3])
        x = g.op("Add", x, linear(g.op("Reshape", o, g.const([-1, H * D])), p + "self_attn.o_proj.weight"))
        h = norm(x, p + "post_attention_layernorm.weight")
        gate = linear(h, p + "mlp.gate_proj.weight")
        act = g.op("Mul", g.op("Mul", gate, g.op("Sigmoid", gate)), linear(h, p + "mlp.up_proj.weight"))
        x = g.op("Add", x, linear(act, p + "mlp.down_proj.weight"))
    last = g.op("Slice", g.op("Reshape", x, g.op("Concat", batch_time, g.const([E]), axis=0)),
                g.const([-1]), g.const([2**62]), g.const([1]))
    last = norm(g.op("Reshape", last, g.const([-1, E])), "model.norm.weight")
    head = "lm_head.weight" if "lm_head.weight" in weight_shapes else "model.embed_tokens.weight"
    logprobs = g.op("LogSoftmax", linear(last, head), axis=-1, name="logprobs")

    inputs = [_value_info(ids, _INT64, ["B", "T"]), _value_info(pos, _INT64, ["B", "T"]),
              _value_info(bias, _FLOAT, ["B", 1, "GT", "S"])]
    for i in range(layers):
        inputs += [_value_info(f"prefix_kT.{i}", _FLOAT, [1, Hkv, D, "P"]), _value_info(f"prefix_v.{i}", _FLOAT, [1, Hkv, "P", D]),
                   _value_info(f"cache_kT.{i}", _FLOAT, ["B", Hkv, D, "C"]), _value_info(f"cache_v.{i}", _FLOAT, ["B", Hkv, "C", D])]
    inputs += [_value_info(name, _FLOAT, list(shape)) for name, shape in weight_shapes.items()]
    outs = [_value_info(logprobs, _FLOAT, ["B", weight_shapes[head][0]])]
    for i in range(layers):
        outs += [_value_info(f"new_kT.{i}", _FLOAT, ["B", Hkv, D, "T"]), _value_info(f"new_v.{i}", _FLOAT, ["B", Hkv, "T", D])]
    graph = _msg((1, g.nodes), (2, "llama"), (5, g.initializers), (11, inputs), (12, outs))
    return _msg((1, 11), (2, "neural_compression"), (7, graph), (8, _msg((1, ""), (2, 23))))


# ---- weights --------------------------------------------------------------------------

def read_safetensors(path: str | Path) -> dict[str, np.ndarray]:
    """All tensors of a .safetensors file as contiguous fp32 arrays (bf16/fp16 upcast exactly)."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    data = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + n)
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        raw = data[start:end]
        if info["dtype"] == "BF16":
            a = raw.view(np.uint16).astype(np.uint32)
            a <<= 16
            a = a.view(np.float32)
        else:
            a = raw.view({"F32": np.float32, "F16": np.float16}[info["dtype"]]).astype(np.float32)
        out[name] = a.reshape(info["shape"])
    del data
    return out


# ---- runtime --------------------------------------------------------------------------

class OnnxLlama:
    """ONNX Runtime session plus the KV cache bookkeeping for constrained beam search."""

    attention_backend = "onnxruntime"

    def __init__(self, config: dict, weights: dict[str, np.ndarray], threads: int):
        import onnxruntime as ort

        self.layers = config["num_hidden_layers"]
        H = config["num_attention_heads"]
        self.kv_heads = config.get("num_key_value_heads") or H
        self.head_dim = config.get("head_dim") or config["hidden_size"] // H
        self.groups = H // self.kv_heads
        self.weights = weights
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        graph = build_graph(config, {k: v.shape for k, v in weights.items()})
        self.session = ort.InferenceSession(graph, options, providers=["CPUExecutionProvider"])
        self._weight_values = {k: ort.OrtValue.ortvalue_from_numpy(v) for k, v in weights.items()}
        self.output_names = [o.name for o in self.session.get_outputs()]

    def _empty(self, rows: int) -> list[np.ndarray]:
        kv, d = self.kv_heads, self.head_dim
        return [np.zeros((rows, kv, d, 0), np.float32), np.zeros((rows, kv, 0, d), np.float32)] * self.layers

    def forward(self, ids, positions, bias, prefix: list[np.ndarray], cache: list[np.ndarray]):
        """Returns (logprobs [B, V], [new_kT, new_v] * layers)."""
        feeds = {"input_ids": ids, "position_ids": positions, "bias": bias, **self._weight_values}
        for i in range(self.layers):
            feeds[f"prefix_kT.{i}"], feeds[f"prefix_v.{i}"] = prefix[2 * i], prefix[2 * i + 1]
            feeds[f"cache_kT.{i}"], feeds[f"cache_v.{i}"] = cache[2 * i], cache[2 * i + 1]
        out = self.session.run(self.output_names, feeds)
        return out[0], out[1:]

    def _bias(self, prefix_len: int, cache_mask: np.ndarray, new_mask: np.ndarray) -> np.ndarray:
        """Additive mask [B, 1, G*T, P+C+T]: prefix always visible, cache by mask, new tokens causal."""
        B, T = new_mask.shape
        causal = np.tril(np.ones((T, T), dtype=bool))[None] & new_mask[:, None, :].astype(bool)
        visible = np.concatenate([
            np.ones((B, T, prefix_len), dtype=bool),
            np.broadcast_to(cache_mask[:, None, :].astype(bool), (B, T, cache_mask.shape[1])),
            causal,
        ], axis=2)
        bias = np.where(visible, np.float32(0), np.float32(NEG))
        return np.ascontiguousarray(np.broadcast_to(bias[:, None, None], (B, 1, self.groups, T, bias.shape[2])).reshape(B, 1, self.groups * T, -1))

    # Backend interface used by models._search_batch -----------------------------------

    def prefix(self, ids: list[int]):
        a = np.array([ids], dtype=np.int64)
        _, new = self.forward(a, np.arange(len(ids), dtype=np.int64)[None], self._bias(0, np.zeros((1, 0)), np.ones_like(a)),
                              self._empty(1), self._empty(1))
        return new

    def start(self, prefix, input_ids: np.ndarray, attn: np.ndarray, positions: np.ndarray):
        n = len(input_ids)
        prefix = prefix if prefix is not None else self._empty(1)
        logprobs, new = self.forward(input_ids, positions, self._bias(prefix[0].shape[-1], np.zeros((n, 0)), attn), prefix, self._empty(n))
        state = _CacheState(prefix, self._empty(n), np.zeros((n, 0), np.float32))
        state.append(new, attn.astype(np.float32))
        return state, logprobs

    def step(self, state: "_CacheState", rows: np.ndarray | None, tokens: np.ndarray, positions: np.ndarray):
        if rows is not None:
            state.select(rows)
        bias = self._bias(state.prefix[0].shape[-1], state.padded_mask(), np.ones((len(tokens), 1)))
        logprobs, new = self.forward(tokens[:, None], positions[:, None], bias, state.prefix, state.buffers)
        state.append(new, np.ones((len(tokens), 1), np.float32))
        return logprobs


class _CacheState:
    """Per-row KV buffers with spare positions; ``mask`` marks the filled, non-padding ones."""

    def __init__(self, prefix, buffers, mask):
        self.prefix, self.buffers, self.mask = prefix, buffers, mask

    @property
    def length(self) -> int:
        return self.mask.shape[1]

    def padded_mask(self) -> np.ndarray:
        cap = self.buffers[0].shape[-1]
        return np.pad(self.mask, ((0, 0), (0, cap - self.length)))

    def select(self, rows: np.ndarray) -> None:
        self.buffers = [b[rows] for b in self.buffers]  # one gather copy, keeps the capacity
        self.mask = self.mask[rows]

    def append(self, new: list[np.ndarray], mask: np.ndarray) -> None:
        L, T = self.length, mask.shape[1]
        cap = self.buffers[0].shape[-1]
        if L + T > cap:  # grow with spare room; unused positions stay zero
            grown = []
            for j, b in enumerate(self.buffers):
                axis = 3 if j % 2 == 0 else 2
                pad = [(0, 0)] * 4
                pad[axis] = (0, L + T + 16 - cap)
                grown.append(np.pad(b, pad))
            self.buffers = grown
        for j, (b, a) in enumerate(zip(self.buffers, new)):
            if j % 2 == 0:
                b[..., L : L + T] = a
            else:
                b[:, :, L : L + T] = a
        self.mask = np.concatenate([self.mask, mask], axis=1)
