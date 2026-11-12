"""Decoder guarantees, checked with a tiny random-weight Llama (fast; needs a one-time tokenizer download).

A random model reconstructs nothing correctly, which is the point. Whatever the
model prefers, the output must keep every stored prefix and separator and
respect the length range implied by the payload.
"""

import json
import shutil

import numpy as np
import pytest

from neural_compression.compressor import WORD_RE, Compressed, compress, compress_with_lengths
from neural_compression.decoder import Decoder, sentence_units

TINY_TOKENIZER = "hf-internal-testing/tiny-random-gpt2"  # small byte-level BPE vocabulary

TEXTS = [
    "The programmer discovered an unexpected problem in the application.",
    "Hello, world! Is this (really) working? Yes: 42 times.",
    "First sentence here. Second one follows!\nA new line, too.",
]


@pytest.fixture(scope="module")
def tokenizer_file():
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(TINY_TOKENIZER, "tokenizer.json")
    except Exception as exc:  # offline
        pytest.skip(f"cannot download {TINY_TOKENIZER}: {exc}")


@pytest.fixture(scope="module")
def lm(tokenizer_file, tmp_path_factory):
    """A 2-layer grouped-attention Llama with random weights, written as safetensors."""
    from safetensors.numpy import save_file
    from tokenizers import Tokenizer

    from neural_compression.models import load_model

    path = tmp_path_factory.mktemp("tiny-llama")
    shutil.copy(tokenizer_file, path / "tokenizer.json")
    vocab, dim, ffn, kv_dim = Tokenizer.from_file(tokenizer_file).get_vocab_size(), 32, 64, 16
    config = {"model_type": "llama", "vocab_size": vocab, "hidden_size": dim, "intermediate_size": ffn,
              "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "rms_norm_eps": 1e-6}
    (path / "config.json").write_text(json.dumps(config))
    rng = np.random.default_rng(9)
    shapes = {"model.embed_tokens.weight": (vocab, dim), "model.norm.weight": (dim,)}
    for i in range(2):
        layer = f"model.layers.{i}."
        shapes |= {layer + "input_layernorm.weight": (dim,), layer + "post_attention_layernorm.weight": (dim,),
                   layer + "self_attn.q_proj.weight": (dim, dim), layer + "self_attn.k_proj.weight": (kv_dim, dim),
                   layer + "self_attn.v_proj.weight": (kv_dim, dim), layer + "self_attn.o_proj.weight": (dim, dim),
                   layer + "mlp.gate_proj.weight": (ffn, dim), layer + "mlp.up_proj.weight": (ffn, dim),
                   layer + "mlp.down_proj.weight": (dim, ffn)}
    save_file({k: rng.normal(0, 0.5, s).astype(np.float32) for k, s in shapes.items()}, path / "model.safetensors")
    return load_model(str(path), threads=2)


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


def test_sentence_units_split_on_sentence_ends():
    c = compress(TEXTS[2], 0.5)
    units = sentence_units(c)
    assert [len(u) for u in units] == [3, 3, 4]


def test_numpy_forward_matches_transformers(tokenizer_file, tmp_path):
    """Padding, the shared prefix, the KV cache and beam reordering must not change any logits.

    The reference runs each final sequence on its own, unpadded and uncached. The weights are
    saved in bf16, like SmolLM2, so the BF16 reader is checked too.
    """
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from neural_compression.models import load_model

    shutil.copy(tokenizer_file, tmp_path / "tokenizer.json")
    config = transformers.LlamaConfig(
        vocab_size=1000, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True,
    )
    torch.manual_seed(9)
    transformers.LlamaForCausalLM(config).to(torch.bfloat16).save_pretrained(tmp_path)
    reference = transformers.LlamaForCausalLM.from_pretrained(tmp_path, dtype=torch.float32).eval()
    model = load_model(str(tmp_path), threads=2).model

    def expected(sequence):
        with torch.no_grad():
            return reference(input_ids=torch.tensor([sequence])).logits[0, -1].numpy()

    prefix, prompts, steps = [5, 9, 400, 77], [[17, 31, 42], [7, 13, 29, 41, 53]], [23, 61, 999]
    cache = model.new_cache(1)
    model.forward(np.array([prefix]), np.arange(len(prefix))[None], cache)
    cache = model.new_cache(2, model.prefix_states(cache))
    ids, valid = np.zeros((2, 5), dtype=np.int64), np.zeros((2, 5), dtype=bool)
    for row, p in enumerate(prompts):
        ids[row, 5 - len(p) :], valid[row, 5 - len(p) :] = p, True
    positions = np.maximum(len(prefix) + valid.cumsum(1) - 1, 0)
    logits = model.forward(ids, positions, cache, valid)
    for row, p in enumerate(prompts):
        np.testing.assert_allclose(logits[row], expected(prefix + p), rtol=1e-4, atol=1e-5)
    beams = [1, 1, 0]  # row 1 splits in two, row 0 moves last
    cache.reorder(np.array(beams))
    logits = model.forward(np.array(steps)[:, None], positions[beams, -1:] + 1, cache)
    for row, (src, token) in enumerate(zip(beams, steps)):
        np.testing.assert_allclose(logits[row], expected(prefix + prompts[src] + [token]), rtol=1e-4, atol=1e-5)
