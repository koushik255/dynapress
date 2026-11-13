"""Llama-architecture inference on ONNX Runtime, without torch or the onnx package.

The graph is written straight to ONNX protobuf from the model's ``config.json``. Weights
are not part of it: they are graph inputs read from the checkpoint's ``model.safetensors``
(bf16/fp16 upcast to fp32 exactly as ``from_pretrained(dtype=float32)`` does).

With int8 weights (``quantize_checkpoint``), every matrix is stored as 8-bit codes with one
scale per block of 32 input columns and multiplied by ``MatMulNBits`` with int8 compute.
These tensors are external initializers, which lets ONNX Runtime prepack them for its
int8 kernels. This is experimental: row-local activation scales do NOT guarantee batch
invariance. The full model's log-probabilities change substantially with left padding,
so adaptive payloads must not be assumed lossless outside their compression batch.

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
import mmap
import struct
from pathlib import Path

import numpy as np

NEG = -1e9  # additive mask value; finite so a fully masked (padding) row stays NaN-free
INT8_BLOCK = 32  # input columns per int8 scale (as in llama.cpp's Q8_0)
_FLOAT, _UINT8, _INT64 = 1, 2, 7


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

    def op(self, op_type: str, *inputs: str, outputs: int = 1, name: str | None = None, domain: str = "", **attrs):
        self.count += 1
        outs = [name] if name else [f"t{self.count}_{i}" for i in range(outputs)]
        attributes = [_attribute(k, v) for k, v in attrs.items()]
        self.nodes.append(_msg((1, list(inputs)), (2, outs), (4, op_type), (5, attributes), *([(7, domain)] if domain else [])))
        return outs[0] if outputs == 1 else outs

    def const(self, values, dtype=np.int64) -> str:
        a = np.asarray(values, dtype=dtype)
        self.count += 1
        name = f"c{self.count}"
        elem = _INT64 if a.dtype == np.int64 else _FLOAT
        self.initializers.append(_msg((1, list(a.shape)), (2, elem), (8, name), (9, a.tobytes())))
        return name


def build_graph(config: dict, weight_shapes: dict[str, tuple], int8_shapes: dict[str, tuple] | None = None) -> bytes:
    """ONNX model bytes for a Llama-style config. Weight names match the HF checkpoint.

    ``weight_shapes`` become fp32 graph inputs. ``int8_shapes`` lists quantized matrices as
    ``<weight>.q8`` / ``<weight>.scale`` pairs, which become external initializers.
    """
    int8_shapes = int8_shapes or {}
    E, layers = config["hidden_size"], config["num_hidden_layers"]
    H, Hkv = config["num_attention_heads"], config.get("num_key_value_heads") or config["num_attention_heads"]
    D = config.get("head_dim") or E // H
    eps = float(config["rms_norm_eps"])
    g = _Graph()

    def norm(x, weight):
        return g.op("RMSNormalization", x, weight, axis=-1, epsilon=eps)

    def linear(x, weight):
        if weight + ".q8" not in int8_shapes:
            return g.op("Gemm", x, weight, transB=1)
        rows, blocks, block = int8_shapes[weight + ".q8"]
        return g.op("MatMulNBits", x, weight + ".q8", weight + ".scale", domain="com.microsoft",
                    K=blocks * block, N=rows, bits=8, block_size=block, accuracy_level=4)

    ids, pos, bias = "input_ids", "position_ids", "bias"
    emb = "model.embed_tokens.weight"
    if emb + ".q8" in int8_shapes:  # dequantize just the looked-up rows
        vocab, blocks, _ = int8_shapes[emb + ".q8"]
        codes = g.op("Sub", g.op("Cast", g.op("Gather", emb + ".q8", ids), to=_FLOAT), g.const(128.0, np.float32))
        scales = g.op("Gather", g.op("Reshape", emb + ".scale", g.const([vocab, blocks, 1])), ids)
        x = g.op("Reshape", g.op("Mul", codes, scales), g.const([-1, E]))
    else:
        x = g.op("Reshape", g.op("Gather", emb, ids), g.const([-1, E]))
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
    head = "lm_head.weight" if {"lm_head.weight", "lm_head.weight.q8"} & (weight_shapes.keys() | int8_shapes.keys()) else emb
    vocab = (weight_shapes.get(head) or int8_shapes[head + ".q8"])[0]
    logprobs = g.op("LogSoftmax", linear(last, head), axis=-1, name="logprobs")

    inputs = [_value_info(ids, _INT64, ["B", "T"]), _value_info(pos, _INT64, ["B", "T"]),
              _value_info(bias, _FLOAT, ["B", 1, "GT", "S"])]
    for i in range(layers):
        inputs += [_value_info(f"prefix_kT.{i}", _FLOAT, [1, Hkv, D, "P"]), _value_info(f"prefix_v.{i}", _FLOAT, [1, Hkv, "P", D]),
                   _value_info(f"cache_kT.{i}", _FLOAT, ["B", Hkv, D, "C"]), _value_info(f"cache_v.{i}", _FLOAT, ["B", Hkv, "C", D])]
    inputs += [_value_info(name, _FLOAT, list(shape)) for name, shape in weight_shapes.items()]
    outs = [_value_info(logprobs, _FLOAT, ["B", vocab])]
    for i in range(layers):
        outs += [_value_info(f"new_kT.{i}", _FLOAT, ["B", Hkv, D, "T"]), _value_info(f"new_v.{i}", _FLOAT, ["B", Hkv, "T", D])]
    external = [_msg((1, list(shape)), (2, _UINT8 if name.endswith(".q8") else _FLOAT), (8, name),
                     (13, [_msg((1, "location"), (2, "supplied-at-session-creation"))]), (14, 1))
                for name, shape in int8_shapes.items()]
    graph = _msg((1, g.nodes), (2, "llama"), (5, g.initializers + external), (11, inputs), (12, outs))
    opsets = [_msg((1, ""), (2, 23)), _msg((1, "com.microsoft"), (2, 1))]
    return _msg((1, 11), (2, "neural_compression"), (7, graph), (8, opsets))


# ---- weights --------------------------------------------------------------------------

def iter_safetensors(path: str | Path):
    """(name, array) for each tensor of a .safetensors file: floats as contiguous fp32 arrays
    (bf16/fp16 upcast exactly), U8 as read-only views of the memory-mapped file."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    data = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + n)
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        raw = data[start:end]
        if info["dtype"] == "U8":
            a = raw
        elif info["dtype"] == "BF16":
            a = raw.view(np.uint16).astype(np.uint32)
            a <<= 16
            a = a.view(np.float32)
        else:
            a = raw.view({"F32": np.float32, "F16": np.float16}[info["dtype"]]).astype(np.float32)
        yield name, a.reshape(info["shape"])


def read_safetensors(path: str | Path) -> dict[str, np.ndarray]:
    return dict(iter_safetensors(path))


def write_safetensors(path: str | Path, tensors: dict[str, np.ndarray], metadata: dict[str, str] | None = None) -> None:
    """Write ``tensors`` atomically (F32, F16 and U8 only)."""
    header, offset = {"__metadata__": metadata or {}}, 0
    for name, a in tensors.items():
        dtype = {np.dtype(np.float32): "F32", np.dtype(np.float16): "F16", np.dtype(np.uint8): "U8"}[a.dtype]
        header[name] = {"dtype": dtype, "shape": list(a.shape), "data_offsets": [offset, offset + a.nbytes]}
        offset += a.nbytes
    head = json.dumps(header).encode()
    head += b" " * (-len(head) % 8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(len(head).to_bytes(8, "little") + head)
        for a in tensors.values():
            f.write(np.ascontiguousarray(a).data)
    tmp.replace(path)


def quantize_int8(w: np.ndarray, block: int = INT8_BLOCK) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric 8-bit codes for a [rows, cols] matrix, one fp16 scale per ``block`` columns of a row.

    Returns codes [rows, cols / block, block] as uint8 with zero point 128 (the MatMulNBits
    layout) and scales [rows * cols / block]. A weight is (code - 128) * scale.
    """
    rows, cols = w.shape
    if cols % block:
        raise ValueError(f"{cols} columns are not a multiple of the int8 block size {block}")
    blocks = w.reshape(rows, cols // block, block)
    scale = (np.abs(blocks).max(axis=-1) / 127).astype(np.float16)  # codes use the stored (rounded) scale
    s = scale.astype(np.float32)[..., None]
    codes = np.rint(np.divide(blocks, s, out=np.zeros_like(blocks), where=s > 0))
    return (np.clip(codes, -127, 127) + 128).astype(np.uint8), scale.reshape(-1)


def quantize_checkpoint(src: str | Path, dst: str | Path, block: int = INT8_BLOCK) -> None:
    """Write an int8 copy of a checkpoint: every matrix becomes ``<name>.q8`` and ``<name>.scale``
    (see ``quantize_int8``); vectors (the norms) stay fp32."""
    out = {}
    for name, w in iter_safetensors(src):
        if w.ndim == 2:
            out[name + ".q8"], out[name + ".scale"] = quantize_int8(w, block)
        else:
            out[name] = w
    write_safetensors(dst, out, {"quantization": f"int8, symmetric, block {block}", "source": str(Path(src).name)})


# ---- runtime --------------------------------------------------------------------------

def _drop_file_pages(arrays) -> None:
    """ONNX Runtime keeps its own prepacked copy of the int8 tensors, so release the pages of the
    memory-mapped file they came from (about 370 MB of RSS). The mapping stays valid: a later read
    would just fault the pages back in from the file."""
    if not hasattr(mmap, "MADV_DONTNEED"):  # Windows
        return
    maps = {}
    for a in arrays:
        while a is not None and not isinstance(a, mmap.mmap):
            a = a.base
        if a is not None:
            maps[id(a)] = a
    for m in maps.values():
        m.madvise(mmap.MADV_DONTNEED)


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
        int8 = {k: v for k, v in weights.items() if k.endswith((".q8", ".scale"))}
        fp32 = {k: v for k, v in weights.items() if k not in int8}
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self._int8_values = [ort.OrtValue.ortvalue_from_numpy(v) for v in int8.values()]  # must outlive the session
        if int8:
            options.add_external_initializers(list(int8), self._int8_values)
        graph = build_graph(config, {k: v.shape for k, v in fp32.items()}, {k: v.shape for k, v in int8.items()})
        self.session = ort.InferenceSession(graph, options, providers=["CPUExecutionProvider"])
        self._weight_values = {k: ort.OrtValue.ortvalue_from_numpy(v) for k, v in fp32.items()}
        _drop_file_pages(int8.values())
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
