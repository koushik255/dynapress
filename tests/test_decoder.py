"""Decoder guarantees, checked with a tiny random-weight model (fast; needs a one-time download).

A random model reconstructs nothing correctly, which is the point. Whatever the
model prefers, the output must keep every stored prefix and separator and
respect the length range implied by the payload.
"""

import pytest

from neural_compression.compressor import WORD_RE, Compressed, compress, compress_with_lengths
from neural_compression.decoder import Decoder, sentence_units

TINY = "hf-internal-testing/tiny-random-gpt2"

TEXTS = [
    "The programmer discovered an unexpected problem in the application.",
    "Hello, world! Is this (really) working? Yes: 42 times.",
    "First sentence here. Second one follows!\nA new line, too.",
]


@pytest.fixture(scope="module")
def lm():
    from neural_compression.models import load_model

    try:
        return load_model(TINY, threads=2)
    except Exception as exc:  # offline
        pytest.skip(f"cannot load {TINY}: {exc}")


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


def test_cpu_grouped_attention_preserves_masked_cached_and_batched_inference(lm, tmp_path):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from neural_compression.models import load_model

    if tuple(map(int, torch.__version__.split(".")[:2])) < (2, 15):
        pytest.skip("native CPU grouped attention needs PyTorch >= 2.15")
    # Real, small grouped-attention model; no pretrained Llama download needed.
    config = LlamaConfig(
        vocab_size=len(lm.tokenizer), hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    )
    torch.manual_seed(9)
    LlamaForCausalLM(config).save_pretrained(tmp_path)
    lm.tokenizer.save_pretrained(tmp_path)
    fast = load_model(str(tmp_path), threads=2, device="cpu")
    reference = load_model(str(tmp_path), threads=2, device="cpu")
    reference.model.set_attn_implementation("sdpa")
    ids = torch.tensor([[0, 0, 17, 31, 42], [7, 13, 29, 41, 53]])
    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    positions = (mask.cumsum(1) - 1).clamp(min=0)
    with torch.no_grad():
        # Without padding, causality still has to stop future-token attention.
        a = fast.model(input_ids=ids[1:], use_cache=False)
        b = reference.model(input_ids=ids[1:], use_cache=False)
        torch.testing.assert_close(a.logits, b.logits, rtol=1e-5, atol=1e-6)
        a = fast.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=True)
        b = reference.model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=True)
        torch.testing.assert_close(a.logits[:, -1], b.logits[:, -1], rtol=1e-5, atol=1e-6)
        # A one-token cached step must see the entire earlier, non-padding context.
        next_mask = torch.cat([mask, torch.ones(2, 1, dtype=mask.dtype)], dim=1)
        extra = {"input_ids": torch.tensor([[23], [61]]), "attention_mask": next_mask,
                     "position_ids": positions[:, -1:] + 1, "use_cache": True}
        a = fast.model(**extra, past_key_values=a.past_key_values)
        b = reference.model(**extra, past_key_values=b.past_key_values)
        torch.testing.assert_close(a.logits, b.logits, rtol=1e-5, atol=1e-6)
    payloads = [compress(t, 0.5).to_bytes() for t in TEXTS]
    expected = [r.text for r in Decoder(reference, "full", beam_size=2).decode_many(payloads)]
    actual = [r.text for r in Decoder(fast, "full", beam_size=2).decode_many(payloads)]
    assert actual == expected
