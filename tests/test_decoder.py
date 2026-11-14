"""Decoder guarantees, checked with a tiny random-weight Llama (fast; needs a one-time tokenizer download).

A random model reconstructs nothing correctly, which is the point. Whatever the
model prefers, the output must keep every stored prefix and separator and
respect the length range implied by the payload.
"""

import json

import numpy as np
import pytest

from neural_compression.compressor import WORD_RE, Compressed, compress, compress_with_lengths
from neural_compression.decoder import Decoder, sentence_units

TINY_TOKENIZER = "hf-internal-testing/tiny-random-gpt2"

TEXTS = [
    "The programmer discovered an unexpected problem in the application.",
    "Hello, world! Is this (really) working? Yes: 42 times.",
    "First sentence here. Second one follows!\nA new line, too.",
]


@pytest.fixture(scope="module")
def tiny_llama(tmp_path_factory):
    """A real grouped-attention Llama with random bf16 weights, as an HF model directory."""
    from huggingface_hub import hf_hub_download

    try:
        tokenizer = hf_hub_download(TINY_TOKENIZER, "tokenizer.json")
    except Exception as exc:  # offline
        pytest.skip(f"cannot download {TINY_TOKENIZER}: {exc}")
    path = tmp_path_factory.mktemp("tiny-llama")
    (path / "tokenizer.json").write_bytes(open(tokenizer, "rb").read())
    config = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "vocab_size": 1000, "hidden_size": 32,
              "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
              "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "tie_word_embeddings": True, "max_position_embeddings": 512}
    (path / "config.json").write_text(json.dumps(config))
    rng = np.random.default_rng(9)
    shapes = {"model.embed_tokens.weight": (1000, 32), "model.norm.weight": (32,)}
    for i in range(2):
        p = f"model.layers.{i}."
        shapes |= {p + "input_layernorm.weight": (32,), p + "post_attention_layernorm.weight": (32,),
                   p + "self_attn.q_proj.weight": (32, 32), p + "self_attn.k_proj.weight": (16, 32),
                   p + "self_attn.v_proj.weight": (16, 32), p + "self_attn.o_proj.weight": (32, 32),
                   p + "mlp.gate_proj.weight": (64, 32), p + "mlp.up_proj.weight": (64, 32), p + "mlp.down_proj.weight": (32, 64)}
    header, blobs, offset = {"__metadata__": {"format": "pt"}}, [], 0
    for name, shape in shapes.items():
        w = (1 + rng.normal(0, 0.1, shape) if name.endswith("norm.weight") else rng.normal(0, 0.3, shape)).astype(np.float32)
        blob = (w.view(np.uint32) >> 16).astype(np.uint16).tobytes()  # bf16, like the real checkpoint
        header[name] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    head = json.dumps(header).encode()
    head += b" " * (-len(head) % 8)
    (path / "model.safetensors").write_bytes(len(head).to_bytes(8, "little") + head + b"".join(blobs))
    return path


@pytest.fixture(scope="module")
def lm(tiny_llama):
    from neural_compression.models import load_model

    return load_model(str(tiny_llama), threads=2)


@pytest.mark.parametrize("context", [0, 2, "full"])
@pytest.mark.parametrize("retention", [0.25, 0.5])
def test_output_respects_payload(lm, context, retention):
    payloads = [compress(t, retention).to_bytes() for t in TEXTS]
    results = Decoder(lm, context, beam_size=2).decode_many(payloads)
    for payload, res in zip(payloads, results):
        c = Compressed.from_bytes(payload)
        assert res.failures == 0
        words = WORD_RE.findall(res.text)
        assert len(words) == len(c.prefixes)
        rebuilt = c.separators[0] + "".join(w + s for w, s in zip(words, c.separators[1:]))
        assert rebuilt == res.text  # separators untouched
        for word, slot in zip(words, c.slots()):
            assert word.startswith(slot.prefix)
            assert slot.min_len <= len(word) <= slot.max_len


def test_adaptive_payload_decodes_within_bounds(lm):
    c = compress_with_lengths("the cat sat on the mat", [3, 1, 3, 1, 2, 2], retention_pct=25)
    res = Decoder(lm, "full", beam_size=2).decode(c.to_bytes())
    words = WORD_RE.findall(res.text)
    assert words[0] == "the" and words[2] == "sat"  # complete words are kept verbatim
    assert words[1].startswith("c") and len(words[1]) >= 2


def test_decoding_is_deterministic(lm):
    payload = compress(TEXTS[0], 0.5).to_bytes()
    a = Decoder(lm, 4, beam_size=2).decode(payload).text
    b = Decoder(lm, 4, beam_size=2).decode(payload).text
    assert a == b


def test_batched_equals_single(lm):
    payloads = [compress(t, 0.5).to_bytes() for t in TEXTS]
    decoder = Decoder(lm, "full", beam_size=2)
    batched = [r.text for r in decoder.decode_many(payloads)]
    single = [decoder.decode(p).text for p in payloads]
    assert batched == single


def test_adaptive_cache_does_not_change_payloads(lm):
    from neural_compression.adaptive import compress_adaptive_many

    class Uncached(Decoder):
        def decode_many(self, payloads, cache=None):
            return super().decode_many(payloads)

    # The random model gets most words wrong, so every round and the max_rounds fallback run.
    texts = TEXTS + [TEXTS[2] + " " + TEXTS[0]]
    cached = compress_adaptive_many(texts, Decoder(lm, "full", beam_size=2), max_rounds=3)
    plain = compress_adaptive_many(texts, Uncached(lm, "full", beam_size=2), max_rounds=3)
    assert [(r.payload, r.rounds, r.decoded_text) for r in cached] == [(r.payload, r.rounds, r.decoded_text) for r in plain]
    assert all(r.rounds > 1 for r in cached)


def test_sentence_units_split_on_sentence_ends():
    c = compress(TEXTS[2], 0.5)
    units = sentence_units(c)
    assert [len(u) for u in units] == [3, 3, 4]


def test_torch_backend_matches_onnx_and_cpu_grouped_attention(lm, tiny_llama):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from neural_compression.models import load_model

    path = str(tiny_llama)
    fast = load_model(path, threads=2, backend="torch")
    reference = load_model(path, threads=2, backend="torch")
    reference.backend.model.set_attn_implementation("sdpa")
    payloads = [compress(t, 0.5).to_bytes() for t in TEXTS]
    expected = [r.text for r in Decoder(reference, "full", beam_size=2).decode_many(payloads)]
    assert [r.text for r in Decoder(lm, "full", beam_size=2).decode_many(payloads)] == expected
    if tuple(map(int, torch.__version__.split(".")[:2])) < (2, 15):
        return  # native CPU grouped attention needs PyTorch >= 2.15
    assert fast.describe()["attention_backend"] == "cpu_gqa"
    ids = torch.tensor([[0, 0, 17, 31, 42], [7, 13, 29, 41, 53]])
    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    positions = (mask.cumsum(1) - 1).clamp(min=0)
    fast_model, ref_model = fast.backend.model, reference.backend.model
    with torch.no_grad():
        # Without padding, causality still has to stop future-token attention.
        a = fast_model(input_ids=ids[1:], use_cache=False)
        b = ref_model(input_ids=ids[1:], use_cache=False)
        torch.testing.assert_close(a.logits, b.logits, rtol=1e-5, atol=1e-6)
        a = fast_model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=True)
        b = ref_model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=True)
        torch.testing.assert_close(a.logits[:, -1], b.logits[:, -1], rtol=1e-5, atol=1e-6)
        # A one-token cached step must see the entire earlier, non-padding context.
        next_mask = torch.cat([mask, torch.ones(2, 1, dtype=mask.dtype)], dim=1)
        extra = {"input_ids": torch.tensor([[23], [61]]), "attention_mask": next_mask,
                 "position_ids": positions[:, -1:] + 1, "use_cache": True}
        a = fast_model(**extra, past_key_values=a.past_key_values)
        b = ref_model(**extra, past_key_values=b.past_key_values)
        torch.testing.assert_close(a.logits, b.logits, rtol=1e-5, atol=1e-6)
    assert [r.text for r in Decoder(fast, "full", beam_size=2).decode_many(payloads)] == expected
