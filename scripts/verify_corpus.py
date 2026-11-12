"""Decode the merged whole-corpus adaptive payload and compare it with the original corpus.

The adaptive payloads were tuned with the decoder on a GPU. Decoding the merged
file on another device (e.g. CPU) checks whether the result survives a change
of hardware and batch composition.

    uv run python scripts/verify_corpus.py --results results --mode adaptive@1
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time

from neural_compression.benchmark import merge_payloads
from neural_compression.compressor import Compressed
from neural_compression.decoder import Decoder
from neural_compression.models import load_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--mode", default="adaptive@1")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    csv.field_size_limit(10**8)
    rows = [r for r in csv.DictReader(open(f"{args.results}/rows.csv", encoding="utf-8")) if r["mode"] == args.mode]
    merged = merge_payloads([Compressed.from_bytes(bytes.fromhex(r["payload_hex"])) for r in rows])
    payload = merged.to_bytes(use_zstd=True)
    original = "\n".join(r["original"] for r in rows)
    lm = load_model(device=args.device)
    t0 = time.perf_counter()
    decoded = Decoder(lm, rows[0]["context"] if rows[0]["context"] == "full" else int(rows[0]["context"])).decode(payload).text
    words_ref, words_out = re.findall(r"\w+", original), re.findall(r"\w+", decoded)
    wrong = [(a, b) for a, b in zip(words_ref, words_out) if a != b]
    report = {
        "mode": args.mode,
        "device": str(lm.device),
        "payload_bytes": len(payload),
        "original_bytes": len(original.encode()),
        "exact": decoded == original,
        "wrong_words": len(wrong),
        "total_words": len(words_ref),
        "examples": wrong[:10],
        "seconds": round(time.perf_counter() - t0, 1),
    }
    print(json.dumps(report, indent=2))
    with open(f"{args.results}/verify_{args.mode}_{lm.device}.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
