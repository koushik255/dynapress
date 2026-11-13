"""Each target/layout gets its own process and model, without compression-side caches."""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0, ".")

p = argparse.ArgumentParser()
p.add_argument("rows")
p.add_argument("weights", choices=["int8", "fp32"])
p.add_argument("--child", type=int)
p.add_argument("--layout", choices=["alone", "mixed", "original"])
p.add_argument("--threads", type=int, default=1)
args = p.parse_args()
rows = [r for r in csv.DictReader(open(args.rows)) if r["mode"] == "adaptive@1"]
if args.child is not None:
    from neural_compression.models import load_model
    from neural_compression.compressor import compress
    from neural_compression.decoder import Decoder
    lm = load_model(weights=args.weights, device="cpu", threads=args.threads)
    if args.layout == "original":
        results = Decoder(lm, "full", beam_size=4, batch_size=16).decode_many(
            [bytes.fromhex(r["payload_hex"]) for r in rows])
        print(json.dumps([dict(index=i, exact=result.text == row["original"],
                               original=row["original"], decoded=result.text, failures=result.failures)
                          for i, (row, result) in enumerate(zip(rows, results))]))
        sys.exit(0)
    i = args.child
    payload = bytes.fromhex(rows[i]["payload_hex"])
    payloads = [payload]
    target = 0
    if args.layout == "mixed":
        # Different batch size, order, padding and companion payloads from compression.
        companions = [compress(rows[(i+j) % len(rows)]["original"], 0.5).to_bytes() for j in (11, 7, 3)]
        payloads = [companions[0], payload, *companions[1:]]
        target = 1
    t = time.perf_counter()
    result = Decoder(lm, "full", beam_size=4, batch_size=4).decode_many(payloads)[target]
    print(json.dumps(dict(index=i, layout=args.layout, exact=result.text == rows[i]["original"],
                         original=rows[i]["original"], decoded=result.text, failures=result.failures,
                         seconds=time.perf_counter()-t, threads=args.threads)))
else:
    out = Path(args.rows).parent / f"fresh-threads{args.threads}.jsonl"
    with out.open("w") as f:
        for layout in ("alone", "mixed"):
            for i in range(len(rows)):
                cmd = [sys.executable, __file__, args.rows, args.weights, "--child", str(i),
                       "--layout", layout, "--threads", str(args.threads)]
                t = time.perf_counter()
                proc = subprocess.run(cmd, text=True, capture_output=True)
                if proc.returncode:
                    raise RuntimeError(f"{cmd}: {proc.stderr}")
                result = json.loads(proc.stdout)
                result["process_seconds"] = time.perf_counter()-t
                f.write(json.dumps(result) + "\n")
                f.flush()
                print(args.weights, layout, i, result["exact"], flush=True)
