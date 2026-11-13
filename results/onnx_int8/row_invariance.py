"""Row r's log-probs alone vs inside batches of other rows (different sizes, padding widths), real model."""
import sys
sys.path.insert(0, ".")
import numpy as np
from neural_compression.models import load_model
w, threads = sys.argv[1], int(sys.argv[2])
lm = load_model(weights=w, threads=threads)
b = lm.backend
texts = ["The quick brown fox jumps over the", "In 1985, the captain of the team was", "Deep neural networks are able to solve tasks across a variety of domains",
         "She will", "The station was purchased by Arcey Broadcasting, which changed the call letters to", "My brother brought brown bread and",
         "After attending torpedo school and naval artillery school, he served on the patrol boat", "Hello"] * 3
prefix_ids = lm.encode("Fill in the missing letters of each word.\n\n")
prefix = b.prefix(prefix_ids)
P = len(prefix_ids)
def run(idx):
    enc = [lm.encode(texts[i]) for i in idx]
    width = max(map(len, enc))
    ids = np.array([[0] * (width - len(e)) + e for e in enc]); attn = np.array([[0] * (width - len(e)) + [1] * len(e) for e in enc])
    pos = np.clip(P + attn.cumsum(1) - 1, 0, None)
    state, lp = b.start(prefix, ids, attn, pos)
    out = [lp]
    last = pos[:, -1]
    for step in range(3):  # greedy steps, each row its own argmax
        tok = out[-1].argmax(1); last = last + 1
        out.append(b.step(state, None, tok, last))
    return np.stack(out, 1)  # [rows, steps, V]
alone = {i: run([i])[0] for i in range(8)}
worst = 0.0; flips = 0
rng = np.random.default_rng(0)
for size in (2, 3, 5, 8, 16, 24):
    idx = list(rng.permutation(len(texts))[:size])
    got = run(idx)
    for r, i in enumerate(idx):
        d = np.abs(got[r] - alone[i % 8]).max(); worst = max(worst, d)
        flips += int((got[r].argmax(-1) != alone[i % 8].argmax(-1)).any())
print(f"{w} threads={threads}: max |logprob(row in batch) - logprob(row alone)| = {worst:.3g}, argmax flips {flips}")
