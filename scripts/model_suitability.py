"""Zero-shot model suitability check (run before choosing the default model).

Compares candidate pretrained models on a small dev set that is disjoint from the
benchmark. Every model gets the same payloads and the same length/character
constraints. Only the way the model is queried differs:

* causal LMs: the package's full-sentence constrained beam search.
* ByT5 (encoder-decoder): T5-style span infilling. Each missing suffix becomes a
  sentinel in the encoder input; decoding is constrained to
  ``<s0> letters <s1> letters ...`` with the same length bounds.

Usage: python scripts/model_suitability.py [--retention 0.5] [--models ...]
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from neural_compression.benchmark import word_accuracy
from neural_compression.compressor import Compressed, compress
from neural_compression.decoder import Decoder
from neural_compression.models import load_model, set_seed

DEV = [
    "Can you pick up some milk on your way home tonight?",
    "The server returned an error because the request was malformed.",
    "Photosynthesis converts light energy into chemical energy stored in glucose.",
    "My brother moved to a small village near the mountains last year.",
    "Use a dictionary when you need fast lookups by key.",
    "The committee postponed the decision until further evidence was available.",
    "Every variable must be declared before it is used in the program.",
    "The patients reported fewer symptoms after the second treatment.",
    "We should probably leave early to avoid the traffic.",
    "Neurons communicate with each other through electrical and chemical signals.",
    "Please remember to save your changes before closing the editor.",
    "The museum displays paintings from several different centuries.",
]

CAUSAL = ["HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-360M", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen3-0.6B-Base", "Qwen/Qwen2.5-1.5B"]


def byt5_reconstruct(model, payload: Compressed, beams: int = 4) -> str:
    """Constrained sentinel infilling. ByT5 was pretrained with sentinel id 258-i (reused byte ids)."""
    letters = [b + 3 for b in b"abcdefghijklmnopqrstuvwxyz"]
    slots, ids, gaps = payload.slots(), [], []
    for i, slot in enumerate(slots):
        ids += [b + 3 for b in payload.separators[i].encode()] + [b + 3 for b in slot.prefix.encode()]
        if not slot.complete:
            ids.append(258 - len(gaps))
            gaps.append((slot.min_len - len(slot.prefix), slot.max_len - len(slot.prefix)))
    ids += [b + 3 for b in payload.separators[-1].encode()] + [1]
    if not gaps:
        return "".join(payload.body_text())

    def allowed(_batch, seq):
        current, length = -1, 0
        for x in seq.tolist()[1:]:
            if 159 <= x <= 258:
                current, length = 258 - x, 0
            else:
                length += 1
        if current == -1:
            return [258]
        lo, hi = gaps[current]
        nxt = [258 - current - 1] if current + 1 < len(gaps) else [1]
        return (letters if length < hi else []) + (nxt if length >= lo else [])

    out = model.generate(
        torch.tensor([ids]),
        max_new_tokens=sum(h for _, h in gaps) + len(gaps) + 2,
        num_beams=beams,
        prefix_allowed_tokens_fn=allowed,
    )[0].tolist()[1:]
    suffixes: list[str] = []
    for x in out:
        if 159 <= x <= 258:
            suffixes.append("")
        elif 3 <= x < 159 and suffixes:
            suffixes[-1] += chr(x - 3)
    words, g = [], 0
    for slot in slots:
        if slot.complete:
            words.append(slot.prefix)
        else:
            words.append(slot.prefix + (suffixes[g] if g < len(suffixes) else ""))
            g += 1
    parts = [payload.separators[0]]
    for w, s in zip(words, payload.separators[1:]):
        parts += [w, s]
    return "".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retention", type=float, nargs="+", default=[0.5, 0.75])
    ap.add_argument("--models", nargs="+", default=["google/byt5-small", *CAUSAL])
    ap.add_argument("--out", default="results/model_suitability.json")
    args = ap.parse_args()
    rows = []
    for name in args.models:
        set_seed(0)
        if "byt5" in name:
            from transformers import T5ForConditionalGeneration

            model = T5ForConditionalGeneration.from_pretrained(name).eval()
            params = sum(p.numel() for p in model.parameters())
            run = lambda payloads: [byt5_reconstruct(model, p) for p in payloads]  # noqa: E731
        else:
            lm = load_model(name)
            params = lm.parameters
            decoder = Decoder(lm, "full")
            run = lambda payloads: [r.text for r in decoder.decode_many(payloads)]  # noqa: E731
        for retention in args.retention:
            payloads = [Compressed.from_bytes(compress(s, retention).to_bytes()) for s in DEV]
            t0 = time.perf_counter()
            with torch.no_grad():
                outputs = run(payloads)
            seconds = time.perf_counter() - t0
            acc = sum(word_accuracy(ref, hyp)[0] for ref, hyp in zip(DEV, outputs)) / sum(word_accuracy(r, r)[1] for r in DEV)
            row = {"model": name, "parameters": params, "retention": retention, "word_accuracy": round(acc, 4), "seconds": round(seconds, 2)}
            print(json.dumps(row), "\n   e.g.", outputs[1], "|", outputs[4], flush=True)
            rows.append(row)
    import pathlib

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
