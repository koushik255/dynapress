# neural_compression: prefix-based lossy text compression with a small LM

A research prototype that asks:

> How much of each word can we delete and still have a small pretrained
> language model put it back, and how do context and compute affect that?

The compressor keeps only the first part of every word. The decompressor asks
a small pretrained language model to fill in the rest. **The scheme is lossy**:
the output can differ from the input, and the benchmark measures how often.

```
original      The programmer discovered an unexpected problem in the application.
stored (50%)  Th progr disco a unexp prob i th applic.          (43 bytes incl. 3-byte header, vs 67)
restored      The programmer discovered an unexpected problem in the application.
```

## Results

125 English sentences in 5 categories (25 each, `data/benchmark.jsonl`),
SmolLM2-360M, beam 4, run on Modal (NVIDIA A10/A10G). Full tables are in
`results/summary.csv`, `results/summary_by_category.csv`, and per-sentence
outputs in `results/rows.csv`.

### Accuracy vs retention and context (word accuracy / exact-sentence accuracy)

| kept per word | ratio* | no context (A) | ±1 word (B2) | ±2 (B4) | ±4 (B8) | full sentence (C) |
|---|---|---|---|---|---|---|
| 25% | 2.07 | 19.9% / 0% | 27.6% / 0% | 31.7% / 0% | 36.1% / 0.8% | **46.2% / 4.0%** |
| 50% | 1.50 | 43.6% / 0% | 61.8% / 0.8% | 70.9% / 8.0% | 74.0% / 12.8% | **81.6% / 20.8%** |
| 75% | 1.12 | 77.0% / 5.6% | 88.8% / 27.2% | 91.2% / 34.4% | 92.3% / 36.8% | **93.3% / 48.8%** |

\*original bytes / stored payload bytes, per sentence, header included. The
same payload serves every context setting. Character error rate at 50%: 20.1%
with no context, 6.8% with full context.

![retention vs accuracy](results/retention_vs_accuracy.png)
![context vs accuracy](results/context_vs_accuracy.png)

**Context matters a lot.** At 50% retention, going from no context to the full
sentence nearly doubles word accuracy (44% → 82%). More neighbours always
helped. Full-sentence decoding beats an 8-word window too, partly because it
also conditions on the already *restored* words to the left.

### Cost of context

![context vs time](results/context_vs_time.png)

| context | 0 | 2 | 4 | 8 | full |
|---|---|---|---|---|---|
| ms per word (A10, 50%) | 5.3 | 15.0 | 27.9 | 51.3 | **7.5** |
| decoder requests (all 125 sentences) | 2005 | 2005 | 2005 | 2005 | **140** |

The windowed strategies issue one request per truncated word, and each
request regenerates its whole window. So their cost grows with the window,
while full-sentence decoding issues one request per sentence. **Here the
most accurate strategy is also nearly the cheapest.** On a laptop CPU (Ryzen 7
7730U) full-sentence decoding ran at ≈100 ms per word.

### Compression vs accuracy, against lossless baselines

![ratio vs accuracy](results/ratio_vs_accuracy.png)

| method (all 125 sentences) | per-sentence files | one corpus file | word acc. | exact sentences |
|---|---|---|---|---|
| gzip -9 (lossless) | 1.02 | 2.12 | 100% | 100% |
| zstd -19 (lossless) | 1.16 | 2.18 | 100% | 100% |
| prefix 75% (+ zstd on corpus) | 1.12 | 2.42 | 93.3% | 48.8% |
| prefix 50% (+ zstd on corpus) | 1.50 | 3.12 | 81.6% | 20.8% |
| prefix 25% (+ zstd on corpus) | 2.07 | 4.26 | 46.2% | 4.0% |
| **adaptive, target 100%** (+ zstd on corpus) | **1.32** | **2.52** | **100%** | **100%** |
| adaptive, target 90% (+ zstd on corpus) | 1.37 | 2.65 | 95.9% | 47.2% |

(Ratios are original / stored bytes. All prefix rows use full-sentence context.
Model weights, 1.45 GB, are **not** counted. The corpus columns zstd-compress
the whole prefix payload as one file.)

**Fixed prefixes do not beat lossless compression in any useful sense.** The
only settings that beat zstd's 2.18 on the corpus still get 7–54% of words
wrong. That is lossy compression with silent, plausible-looking errors.

### Adaptive prefixes

![adaptive vs fixed](results/adaptive_vs_fixed.png)

Adaptive prefixes move the trade-off a long way. With target 100%,
**every one of the 125 sentences is reconstructed exactly**. A separate fresh
decode of the final payload confirmed all 125. The compressor needed 5.3
rounds on average (max 9, i.e. some words were escalated to full storage). The
result is smaller than zstd both per sentence (1.32 vs 1.16) and as a corpus
(4,990 vs 5,753 bytes, −13%). Fixed 75% is both larger *and* less accurate
than adaptive at 90%.

So the system becomes a **lossless** compressor, with conditions: receivers
must share the exact model, revision, and decoder settings, compression costs
several decoder runs per sentence, and decoding is ~10⁴× slower than zstd. It is also far from
what the same model could achieve with arithmetic coding over its next-token
probabilities, the standard way to turn an LM into a compressor. Truncating
words throws away the model's probabilities and keeps only "which prefix was
enough".

**Cross-device check.** The 125 adaptive payloads were tuned on an NVIDIA A10.
Merged into one 4,990-byte corpus file, they were decoded again on a laptop
CPU (`scripts/verify_corpus.py`) and came back **byte-identical: 2,056/2,056
words, every separator** (`results/verify_adaptive@1_cpu.json`). This is
evidence, not a guarantee: a near-tie in beam search could still flip on some
other hardware.

### By category (full-sentence context)

| category | 50%: word / sentence | 75%: word / sentence | adaptive 100%: ratio |
|---|---|---|---|
| conversational | 90.1% / 48% | 97.7% / 84% | 1.40 |
| technical docs | 75.2% / 4% | 92.2% / 36% | 1.25 |
| scientific | 83.3% / 8% | 94.9% / 44% | 1.35 |
| names & numbers | 79.9% / 12% | 90.8% / 28% | 1.32 |
| ambiguous prefixes | 80.3% / 32% | 91.5% / 52% | 1.28 |

Everyday language is easiest. Technical documentation (identifiers,
`TensorFlow`, `DETR`, `panoptic`) and names are hardest. The adaptive ratio
shows the same thing: hard text needs longer prefixes.

### What the errors look like (50%, full context)

| original | restored | kind |
|---|---|---|
| Gazi Ashraf Hossain Lipu | Gary Ashlee Hossley Lien | names → common names |
| TensorFlow, JAX, compiler | TensoiOpt, JAMA, company | jargon → non-words / wrong words |
| Monte Carlo gradient | Monday Carola graduate | rare phrase → frequent words |
| initialization, panoptic | initialeonnen, panother | hallucinated non-words |
| be, Dr, GO | b, D, G | copy bias: prefix echoed as a complete word |
| concrete, patent | concert, patie | shared-prefix ambiguity |

## Quick start

```bash
uv sync                                   # Python 3.14, ONNX Runtime (no PyTorch)
uv run pytest                             # decoder-guarantee tests (a few seconds)

echo "The programmer discovered an unexpected problem in the application." > input.txt
uv run python -m neural_compression compress input.txt --retention 0.5          # -> input.bin
uv run python -m neural_compression decompress input.bin --context full
uv run python -m neural_compression decompress input.bin --context 4
uv run python -m neural_compression compress input.txt --adaptive --target 1.0  # verified by the real decoder

uv sync --group plots                     # matplotlib, only needed for the charts
uv run python -m neural_compression benchmark --dataset data/benchmark.jsonl --out results
uv run python -m neural_compression plot --results results     # redraw charts
```

The first model call downloads `HuggingFaceTB/SmolLM2-360M` (~720 MB, pinned
revision) into the Hugging Face cache. A plain `.txt` file with one sample per
line also works as `--dataset`.

**Backends.** On CPU the model runs on ONNX Runtime (`neural_compression/onnx_llama.py`):
the Llama graph is written directly from the checkpoint's `config.json`, and the
fp32 weights are read from its `model.safetensors`, so no export step, PyTorch, or
extra download is needed. It supports Llama-architecture models such as SmolLM2.
The shared prompt prefix's keys/values are broadcast to every beam row instead of
copied, and appends to the per-row cache are in place.

The PyTorch + Transformers backend is optional: `uv sync --group torch`, then
`--backend torch` on CPU, or `--device cuda` / `--batch-size 64` on a GPU (CUDA
always uses it, and so do the Modal scripts). On PyTorch >= 2.15 it uses native
masked grouped-query attention for Llama models on CPU. The student model and the
fine-tuning scripts also need the `torch` group. The backend is recorded in
benchmark metadata. Both keep fp32 weights and the same search constraints.

**On Modal GPUs** (each configuration runs in its own container, in parallel;
finished configurations are cached in `results/_parts/`, so reruns resume):

```bash
uv run --with modal modal run scripts/modal_app.py --out results
uv run --with modal modal run scripts/modal_app.py --limit-per-category 1 --contexts "0 full" --retentions 0.5 --adaptive-targets ""
```

### Dev loop

For quick iteration on the method, `scripts/dev.py` scores the current code
(uncommitted changes included) on a separate dev set and logs every run:

```bash
uv run --with modal python scripts/dev.py --label "what I changed"   # about 1 minute on Modal
uv run python scripts/dev.py --local --limit 5 --no-adaptive         # CPU smoke check
uv run python scripts/dev.py --history                               # all logged runs
```

* **Dev set:** `data/dev_book.txt`, 50 consecutive sentences (805 words) from
  an obscure public-domain novel (*White Magic*, 1910, Project Gutenberg
  #65432), built by `scripts/build_dev_set.py`. The 125 benchmark sentences
  stay the held-out test: run them only to confirm a win found on the dev set.
* **Two configurations, full-sentence context:** fixed 50% (word accuracy, the
  quick signal) and adaptive at target 100% (lossless bytes against zstd, the
  real score).
* **Run log:** `results/dev/runs.jsonl` gets one line per run (label, commit,
  scores, bytes, seconds). `results/dev/runs/<id>.json` keeps that run's
  per-sentence rows and the uncommitted diff of tracked code.
* **Comparison:** each run prints its change against the previous run on the
  same sentences, and the words that flipped between right and wrong. A rerun
  without code changes on the same GPU type flipped no words.

Baseline (SmolLM2-360M, beam 4, NVIDIA A10): 73.8% word accuracy at fixed 50%;
adaptive 2,021 bytes against 2,286 for zstd -19 (88.4%).

### LoRA experiment

`python -m neural_compression.finetune` prepares a pilot corpus, trains a shared
prefix-restoration adapter, and compares it with the unchanged base model using
the real constrained decoder. It does not change the default model or payload.

The explicit book split is in `data/lora_books.json`: six training books,
two validation books (*The War of the Worlds*, *Alice's Adventures in Wonderland*),
and two test books (*The Picture of Dorian Gray*, *Dracula*). The existing dev
book (*White Magic*) and benchmark are not training sources. These are Gutenberg
books, public domain in the USA; check local rights before redistribution.
They may already be in the pretrained model's training data: this tests adaptation
to held-out fine-tuning books, not guaranteed unseen literature. Splits are by
book, not author (Wells occurs in training and validation).

```bash
uv sync --group training
uv run --group training python -m neural_compression.finetune prepare
uv run --group training python -m neural_compression.finetune train --out results/lora/pilot
uv run --group training python -m neural_compression.finetune evaluate --run results/lora/pilot --out results/lora/pilot-validation
# Only after choosing the experiment using validation:
uv run --group training python -m neural_compression.finetune evaluate --run results/lora/pilot --split test --final-test --out results/lora/pilot-test
```

Preparation caches raw downloads in `data/lora/books`, strips Gutenberg wrappers,
normalizes whitespace, and selects up to 2,000 shuffled sentence units per book
with seed 0. Units follow the decoder's boundaries, contain 5–48 words, and end
in sentence punctuation. Duplicate book ids/contents are rejected; sentences
shared across splits are removed. Source checksums are pinned in the book list;
changed downloads are rejected. `data/lora/manifest.json` records source and
split SHA-256 hashes. Keep this generated manifest with any reported results.
For your own corpus, `--manifest` accepts the same book entries with a local
`path` (relative to the manifest) instead of a `url`. Do not change splits after
looking at validation or test results.

Training uses the decoder's existing few-shot prompts. Each sentence contributes
one fixed-prefix example (25%, 50%, or 75%) and one mixed-length adaptive example;
numbers stay complete. Only answer tokens contribute to loss, never prompts or
padding; overlength examples are dropped rather than truncated. The base weights
stay frozen: LoRA rank 8, alpha 16, dropout 0.05 on attention `q_proj`/`v_proj`,
learning rate 0.0002, effective batch size 16, three epochs, fp32. Validation
answer-token loss selects the checkpoint; **validation adaptive-100% bytes and
runtime decide whether the experiment is worth taking to test**. Training never
loads test text. This first experiment has no paragraph context or book memory.

Outputs include `adapter/`, `merged/`, and `experiment.json` (data hashes,
hyperparameters, versions and metrics). The merged export can be passed directly
to existing commands as `--model results/lora/pilot/merged`. Both sender and
receiver must use that same export and decoder settings. Each run needs a fresh
output directory; interrupted training is not automatically resumed.

Evaluation defaults to 25 sentences from each validation book, using the same
samples, beam 4, batch size 16 and device for both models. It reports fixed-50%
accuracy, adaptive-100% merged/zstd payload bytes, mean rounds, compression and
decode time, per-book scores, and adapter storage separately. Every adaptive
payload is decoded again and exact reconstruction is required. Increase
`--per-book` for the final comparison. Shared weights are excluded from payload
sizes; transmitting a new adapter is an additional cost. `--final-test` is an
explicit safeguard, not a mechanism that prevents repeated test-set tuning.

**Compute:** the `training` group installs CPU PyTorch. A full run
is intended for a GPU environment with CUDA-enabled PyTorch, `transformers`,
`peft`, `accelerate` and `numpy` on Python 3.14 (zstd comes from the standard
library's `compression.zstd`); invoke `python -m ...` there rather
than syncing this CPU-only uv environment over it. On CPU use `--cpu`; `--max-steps 1`
and a separately prepared small corpus provide a wiring check, not evidence of
quality improvement.

For the calibrated Modal pilot, run `uv run --with modal modal run scripts/modal_lora.py`
after preparing the corpus. It uses one L4 (A10G fallback) with capped 2-core/8-GiB
CPU resources, no automatic retries, and a three-hour function timeout. The first 30 optimizer
steps are part of training; their measured throughput gates continuation against
a $5 estimate, with a 50% timing margin and $0.75 contingency. This is a runtime
safety mechanism, not a Modal account-level spending limit. The script uploads
only train/validation files, evaluates 25 sentences per validation book, saves a
source snapshot, and downloads the adapter and results under `results/lora/`.
Merged weights/checkpoints remain in `neural-compression-hf-cache` under the
reported run directory; retained storage is billed separately. It does not deploy
a persistent GPU service or open the held-out test set.
`--model`/`--revision` train another base model (the revision must be pinned),
`--budget` lowers the safety budget and the function timeout with it, `--gpu A10G`
pins the GPU type, and `--control-run <run id>` evaluates an earlier run again in
the same container so decode times share hardware and code. If the full plan does not
fit, calibration first skips the control evaluation, then trains fewer epochs (the
3-epoch learning-rate schedule is kept), and stops only if one epoch does not fit.

### Single-pass student decoder

`neural_compression/student.py` is a separate decoder for the same payloads. It
restores every truncated word of a sentence in **one forward pass** of a small
bidirectional encoder, instead of writing suffixes token by token with beam
search. It plugs into the unchanged adaptive loop (`compress_adaptive_many`
accepts any object with `decode_many`). The LM decoder is not modified.

How it decodes one sentence unit:

1. Each word becomes one input vector, built only from the payload: hashed
   position-aware characters of the stored prefix, its case pattern, the
   separators around it, the gap length range from the header/bitmap, and a
   *candidate bag*: the unigram-prior-weighted mean embedding of every
   vocabulary word that starts with the prefix and fits the length range.
2. An 8-layer pre-LN transformer encoder (d=512) contextualises the words.
3. For each truncated word, every vocabulary candidate (a sorted lower-case word
   list, found by bisection) gets `proj(h)·E[w] + bias[w]`. One extra **OOV**
   score means "not in my vocabulary". The argmax wins, and ties go to the
   earlier vocabulary entry.
4. If OOV wins, or nothing fits, a character 5-gram model spells a suffix of
   an allowed length. The suffix copies the prefix's case pattern (all caps
   after an all-caps prefix of two or more letters).
   This fallback uses a small sequential beam search; only the neural encoder
   is single-pass.

Determinism: each sentence unit is encoded on its own (batch of one, no padding).
Its features use only the unit's own text: the first and last separators are cut
at whitespace. So a sentence decodes the same alone or inside a merged document
(`tests/test_student.py` checks this), and the adaptive loop's final decode is
the receiver's decode.

**Training is distillation from SmolLM2-360M through text.** The teacher
(pinned base model) sampled 9.0M nominal token slots on an A10 in 36 minutes
(147 batches × 384 continuations × up to 160 tokens; actual non-padding tokens
were not counted), at temperature 1 and top-p 0.95, continuing passages of one
or two training-book sentences. This
gave 327,774 complete sentences (5.5M words), cut with the decoder's own unit
boundaries. Any sentence that also occurs in the validation books was removed.
They are mixed with the 20,610 eligible sentences of the six training books
(`finetune prepare --per-book 100000`, drawn 3× as often). Every time a sentence
is drawn, it gets a fresh random fixed-ratio or adaptive-style payload from the
real compressor functions. The loss is cross-entropy over exactly the
candidate set the decoder searches, plus OOV. The vocabulary (27,213 words,
count ≥ 2) and its unigram prior come from the same text, so most of the
student's word knowledge comes from what the teacher wrote. Per-slot soft labels from the teacher
were not used: the teacher decodes a whole sentence jointly over subword tokens,
and the adaptive score only rewards matching the ground truth.

**Rare words and names** are the known weak spot. A word that is not in the
vocabulary can only be restored by the character n-gram, which rarely spells a
name right. In adaptive mode such a word costs a longer prefix, usually the
whole word after six rounds, but it never breaks exactness. See the byte
breakdown below.

```bash
uv run --with modal modal run scripts/modal_student.py::generate      # teacher text, time-capped
uv run --with modal modal run --detach scripts/modal_student.py::train --teacher /cache/student/<gen-run>/teacher.jsonl.gz
uv run python -m neural_compression.student_train evaluate --model <run>/model --out <dir>   # pilot's 50 validation sentences
```

**Results** on the pilot's 50 validation sentences (941 words; 25 per validation
book, validation SHA-256 `ae91209213adbc5bc08e79e07c533f13ecbaefbdbb52c4d1e08b600487320f94`).
No test text was used for training, selection, or evaluation. Earlier corpus
preparation generated and hashed the test file; recovery did not download it. The
checkpoint was picked by fixed-50% accuracy on the next 100 sentences per
validation book (step 20,000 of 36,520).

| decoder | params | word acc. @ fixed 50% | adaptive-100% payload | **after zstd** | decode ms/word, A10 (adaptive / fixed) |
|---|---|---|---|---|---|
| SmolLM2-360M base | 362M | 81.2% | 3,827 B | 2,168 B | 12.1 / 10.1 |
| SmolLM2-360M + LoRA | 362M | 86.7% | 3,658 B | 2,062 B | 12.7 / 10.0 |
| **student, books + teacher text** | 45.8M | **87.4%** | **3,430 B** | **1,937 B** | **0.63 / 0.45** |
| student, books only (same encoder dimensions and GPU, 30 min) | 37.1M | 76.8% | 3,811 B | 2,175 B | 1.11 / 0.52 |
| (zstd -19 of the 50 originals) | | | | 2,481 B | |

* **Exact and deterministic:** every adaptive payload decodes back to its
  sentence, the merged 50-sentence corpus payload decodes back to the corpus,
  and the adaptive compressor produced **byte-identical payloads** on an A10 GPU,
  a 4-core Modal CPU, and a 2-thread orb CPU. These are observed checks on this
  corpus, not a guarantee across arbitrary hardware or floating-point libraries.
* **Speed:** about 20× faster than the original 360M pilot on the same GPU class
  (separate runs, not a paired timing experiment). On CPU it
  takes 0.73 / 1.07 ms/word on 4 cores and 1.35 / 1.89 ms/word on 2 cores
  (fixed / adaptive). A fresh recovery-orb run measured 1.45 / 2.18 ms/word
  for the teacher student and 1.63 / 2.43 for books only, with two Torch threads.
  Adaptive compression of all 50 sentences, about 5 decoder
  rounds, takes 3.5 s on the A10. Adaptive decodes are slower than fixed ones
  because the n-gram fallback runs more often there. Each sentence is encoded
  alone (no batching).
* **Size:** 45.8M parameters (27,213-word tied embedding table 14M, encoder 25M,
  hashed character/separator tables 6M). Storage is 91,925,897 B including config
  and vocabulary (91,633,327 B fp16 weights), versus about 724 MB (bf16) for the
  shared 360M model. Inference loads weights as fp32. Shared model storage is
  excluded from payload scores; sending this model for a tiny corpus would
  overwhelm the byte savings. The weights are
  not in git; they are on the `neural-compression-hf-cache` volume under
  `/student/train-20261111-215809/model`.
* **Where the adaptive bytes go.** All decoders share the 1,243 separator bytes
  and the 1,294 characters of the 25% starting prefixes; 45 one-letter words
  are always complete. Characters added beyond the start: base 1,167, LoRA 998,
  student 770. The student stores fewer in-vocabulary words in full (91, against
  124 for LoRA and 189 for base). Words outside its vocabulary are rare
  here (20 of 941: *Gryphon*, *Chertsey*, *Ealing*, *enhaloed*, …). The character
  n-gram restores half of them from a long prefix (*Martian*→*Martians*,
  *Eppi*→*Epping*); the other half are stored whole. Those cost 51 extra characters,
  against about 33 for the LoRA LM, which restores *Gryphon* from *Gry*. Mean
  adaptive rounds: 4.92 (base 4.92, LoRA 4.72).
* **Teacher data:** the books-only ablation's best checkpoint was step 4,000
  (76.84% selection-dev accuracy); later steps overfit despite 99.9% training
  accuracy. On the pilot 50, teacher text plus books improves fixed accuracy by
  10.52 percentage points and saves 238 zstd bytes (10.94%) over books only.
  Mean rounds fall from 5.76 to 4.92; complete words from 261 to 146. Vocabulary
  grows from 10,287 to 27,213 words. This compares training recipes, not teacher
  text alone: vocabulary/parameter count and training budgets (30 versus 62 min)
  differ. The cumulative evaluation fallback counter falls from 690 to 142;
  these count repeated calls across fixed/adaptive/merged checks, not unique words.

Caveats: 50 sentences from two books is a small sample. The validation books
are period fiction like the training books. Teacher sampling was prompted with
training-book passages, which favours this domain. The 360M decoder is a
general model with no task training (base) or a light adapter (LoRA). The student
knows nothing outside its 27k-word vocabulary except what the character n-gram
can spell, so technical text, names, and other domains will cost more bytes than
here (see the broader continuation measurement below). Measured-runtime compute estimates total $3.334616:
teacher text $0.851091, training/evaluation $1.668197, and ablation $0.815328
(1,827.18 seconds). The later Modal billing-report snapshot totals **$3.35537611
of the original $8 cap**: $0.85476469 teacher, $1.67766665 main, $0.82294477
ablation. Retained storage is separate. `results/student/billing.json` records
the app-level report, plus the other two experiments ($1.72594133 for 135M and
$0.11123479 profiling; **$5.19255223 combined**). All three original student apps are stopped, with zero tasks;
recovery launched no training or evaluation jobs on Modal.

Small configs, training histories, per-sentence payloads, and CPU/GPU summaries
are tracked under `results/student/teacher-books/` and `results/student/books-only/`.
Large weights stay ignored under `results/student/runs/` and on the Modal volume;
the ablation is `/student/train-20261111-230219/model`, and teacher text is
`/student/gen-20261111-211749/teacher.jsonl.gz`. `results/student/verification.json`
records validation/model hashes and fresh recovery-orb CPU checks of both models.
The result is promising for this task and domain, not a proven general-purpose
replacement for the LM decoder.

### Continuing the student on broader text

`student_train train --init-model <saved-model-directory>` starts from saved
weights with the **same vocabulary, unigram prior, n-gram fallback, and encoder
architecture**. AdamW and its learning-rate schedule start fresh; this is not
an exact optimizer-state resume. Architecture flags apply only to new models.
Use a new output directory and a lower learning rate. Optional `weight` values
on training JSONL rows override `--book-weight` for those real-text sentences;
teacher sentences retain weight 1. The source model is never overwritten.

`--select-by adaptive-bytes` picks the lowest merged development payload size
after zstd, checking exact per-sentence and merged reconstruction at each
evaluation. Otherwise selection remains fixed-50% accuracy. Continuation first
scores and saves the original as step zero, so an unimproved run keeps it.
`--dev-start 0 --dev-per-book 40` evaluates the first 40 rows per source group;
the original default remains rows 25–125 per book. Development data is used for
selection, not for a final held-out generalization claim.

`scripts/prepare_student_continuation.py` builds a small document-disjoint
experiment using DailyDialog conversations, Transformers documentation pages,
and another Austen novel, while retaining the original six training books.
It uses cached HTTP responses, records source/content/split hashes, and removes
normalized exact matches to previous training/validation, teacher text, and the
benchmark from new sentences. New training and development documents and exact
sentences are disjoint. It does not open the original book test split.
Near-duplicate text and overlap with the teacher's pretraining are not ruled out.
Raw sources, prepared text, teacher text, and model weights remain ignored.
The script expects the previous book train/validation data and teacher gzip
under its documented `data/student/` paths. Prepared continuation train and
development files are also retained on the Modal volume at
`/student/continuation-20261112/`, alongside their manifest.

**Licensing:** DailyDialog is CC BY-NC-SA 4.0 (Li et al., IJCNLP 2017), so this
new checkpoint is for noncommercial research, not commercially cleared.
The Transformers documentation dataset is Apache-2.0; Gutenberg text is public
domain in the USA, with local rights to be checked. The result manifest records
source URLs and attribution. No new dialogue text or model weights are published
in Git.

For Modal, `STUDENT_DATA=<prepared-directory>` chooses which train/validation
files to upload. Only those two files and the pilot train/validation files are
mounted, never test text or the raw-source directory. For example, the following
launches a **new paid job**, rather than reconnecting to an existing run:

```bash
STUDENT_DATA=data/student/continuation uv run --frozen --with modal modal run --detach scripts/modal_student.py::train \
  --teacher /cache/student/gen-20261111-211749/teacher.jsonl.gz \
  --options '{"init_model":"/cache/student/train-20261111-215809/model","train_seconds":1800,"lr":0.00006,"warmup":200,"steps":20000,"eval_every":2000,"dev_start":0,"dev_per_book":40,"select_by":"adaptive-bytes","seed":731}'
```

This run reuses the existing teacher corpus, checks a paired GPU baseline, and
evaluates the saved result on CPU/GPU and the original pilot. There are no
automatic paid retries or persistent deployments. At the script's list-price
estimate the complete 95-minute timeout costs at most $2.54 compute; storage is
separate. An experiment budget must include every launch, not reset per job.

**Continuation result (2026-11-12 UTC).** One 30-minute-guarded run mixed
20,610 original book sentences, 5,973 new real sentences (weight 20 each), and
327,775 eligible sentences from the existing teacher corpus (weight 1; original
books weight 3). The vocabulary and architecture stayed fixed. No new teacher
generation was launched. Step **4,000** was selected; continuing to step 15,598
did not improve its in-training development score. The checkpoint is retained
at `/student/train-20261112-000121/model` on the existing Modal volume; the
original checkpoint is unchanged. This is a continuation/data-mixture experiment,
not a controlled test of more diverse text alone.

The new development set has **120 sentences / 1,840 words**, 40 per domain.
Training uses *Pride and Prejudice*, development *Sense and Sensibility*: they
are book-disjoint, not author-disjoint. Dialogue splits are conversation-disjoint
and technical splits are documentation-page-disjoint. They are not guaranteed
absent from the teacher's base pretraining. No original test file was accessed.

| development subset (CPU, independently compressed) | original zstd bytes | continued zstd bytes | fixed-50% word accuracy, original → continued |
|---|---:|---:|---:|
| dialogue | 824 | 813 | 86.1% → 86.1% |
| technical | 1,760 | 1,601 | 59.8% → 75.5% |
| fiction | 1,761 | 1,760 | 86.6% → 87.0% |
| **merged 120 sentences** | **4,163** | **3,974** | **76.7% → 82.6%** |

The independently compressed domain sizes do not sum to the merged size.
The saved export saves **4.54%** on the orb CPU. Paired A10 measurements in the
same job are **4,161 → 3,976 B (4.45%)**, with adaptive decoding
**0.911 → 0.987 ms/word (8.3% slower)**. Alternating three CPU decoding trials
measured median **1.185 → 1.332 ms/word (12.4% slower)**. CPU fixed-50 timing
during concurrent tests is not used for the speed comparison. Selection evaluated
the live fp32 checkpoint at 3,974 B; its fp16 export reloaded as fp32 measured
3,976 B on GPU and 3,974 B on CPU, so final comparisons use the exported weights.

There is a small regression on the original pilot: fresh CPU adaptive size
**1,937 → 1,960 B (+1.19%)**, despite fixed-50 accuracy rising 87.35% → 88.10%.
The new checkpoint is therefore an optional broader-domain trade-off, not a
universal replacement for the original. No default decoder was changed.
Technical OOV words remain **147/674 (21.8%)**; fixed-vocabulary training cannot
add those words to the candidate list. Vocabulary coverage is a sensible next
experiment, not an excuse to spend the unused budget automatically.

Every new development and pilot adaptive payload and merged corpus reconstructed
exactly on its producing device. Independently on the orb CPU, **120/120 broader
and 50/50 pilot GPU-produced payloads**, plus their merged payloads, decoded
exactly. Broader CPU/GPU adaptive payloads were byte-identical for 119/120
sentences, not all 120; both different forms decoded exactly. CPU-produced
payloads were not separately rechecked on GPU, and arbitrary-platform
compatibility is not guaranteed.

`results/student/continuation/` retains the source/split manifest, training
history, exported model metadata, CPU/GPU summaries, per-domain and three-trial
CPU checks, and app billing. Latest stopped-app compute billing is **$0.86108648
of the additional $3 cap** (a billing snapshot, not a final invoice; storage
separate). The app stopped with zero tasks. All 15 tests passed. These are
checkpoint-selection development results, not a final held-out test result.

### Student runtime speed pass

The character fallback now caches at most 4,096 context probability rows per
decoder, selects the best beam entries without sorting every expansion, and
stops when no unfinished spelling can beat the best completed spelling.
Probability arithmetic and tie ordering are unchanged; model weights,
vocabulary, payload format and default adaptive search settings are unchanged.

Paired CPU measurements against [058f1f4](https://github.com/koushik255/dynapress/commit/058f1f464246b89c71003e469a03d539d2be697a), two Torch threads, medians of three
alternating old/new runs, with the probability cache cleared separately before
compression and decompression:

| checkpoint / development corpus | adaptive compression seconds, before → after | decompression seconds, before → after | merged zstd bytes, unchanged |
|---|---:|---:|---:|
| continued / broader 120 sentences | 19.020 → 9.499 | 2.026 → 1.542 | 3,974 |
| original / broader 120 sentences | 16.860 → 9.321 | 2.113 → 1.562 | 4,163 |
| continued / fiction pilot 50 sentences | 5.606 → 3.953 | 1.034 → 0.803 | 1,960 |
| original / fiction pilot 50 sentences | 5.387 → 4.308 | 0.967 → 0.856 | 1,937 |

The broader set contains 1,840 words; the pilot 941. Times exclude model loading
and final merged zstd packaging. Every default adaptive payload was identical
between implementations. These are CPU measurements, not a GPU speed claim;
they reuse development text, not a newly accessed test split.

**Fewer compression rounds are opt-in.** A separate six-setting sweep on the
continued model's broader set measured these three useful choices (three-run
medians, same cold-cache protocol):

| adaptive settings | compression seconds | mean rounds | merged zstd bytes |
|---|---:|---:|---:|
| default: step 1, full-word fallback after round 6 | 10.634 | 5.250 | 3,974 |
| step 2, fallback after round 6 | 7.211 | 3.825 | 4,087 |
| step 2, fallback after round 3 | 6.687 | 3.342 | 4,108 |

The last option is 1.59× faster at compression than the optimized default in
that sweep, but stores 3.37% more bytes. It did not improve decompression
(1.735 versus 1.694 seconds), so it is not the new default. Starting at 40% or
50% retention and shortening the round limit alone were also tested. All
settings still verify exact reconstruction; full-word fallback is followed by
another verification round, so `--max-rounds 3` is not a hard limit of three
decoder calls. Context changes can also break previously correct words.

The existing `compress_adaptive_many` API accepts `step`, `max_rounds` and
`start_retention`; the student evaluation CLI now exposes and records them:

```bash
uv run --frozen python -m neural_compression.student_train evaluate \
  --model results/student/runs/train-20261112-000121/model \
  --data data/student/continuation --per-book 40 --threads 2 \
  --step 2 --max-rounds 3 --out results/student/runs/speed-fast
```

`results/student/speed/measurements.json` retains all three-run timings, source
hashes and before/after profile counts; `verification.json` retains model hashes,
cross-version CPU decoding, CLI checks and the plain-zstd control. Both saved
models are unchanged, and all 20 tests passed. No new training or Modal jobs
were launched. Plain zstd remains much faster; the student trades runtime and
shared model storage for smaller payloads, not for beating zstd's speed.

## How it works

### 1. Compression (`compressor.py`)

* Text is split with `\w+|\W+` into **words** (maximal runs of word characters)
  and **separators** (everything else: whitespace, punctuation, symbols).
  Separators are stored verbatim.
* Fixed mode keeps `ceil(len × retention)` characters of each word, at least 1.
  Words containing a digit are stored verbatim by default (`--truncate-numbers`
  turns that off), because digits cannot be predicted.
* Capitalisation is kept as stored. The model predicts the case of the suffix
  (almost always lower case).

**Payload format** (unambiguous without delimiters):

```
byte 0   format version (1)
byte 1   flags: bit0 adaptive, bit1 verbatim-numbers, bit2 zstd-compressed body
byte 2   retention percent (1..100)
rest     [adaptive only: varint word count + 1-bit-per-word "complete" bitmap]
         UTF-8 body: sep0 prefix0 sep1 prefix1 ... sepN
```

Prefixes contain only word characters and separators contain none, so running
the same regex on the body recovers every word boundary. No length fields or
delimiters are needed.

**The header implies word lengths.** In fixed mode the decoder knows the
retention percentage, so a `k`-character prefix limits the original length to
the few values `n` with `ceil(n·pct/100) = k`. At 50%, `disco` means a 9- or
10-letter word. That information is already paid for in the 3-byte header,
and the decoder enforces it. Adaptive mode has no such rule, so it stores a
complete/truncated bit per word, and truncated words may grow by up to 20
characters.

Sizes reported: `raw_prefix_bytes` (stored word characters only),
`payload_bytes` (the real serialized file: header + separators + prefixes),
and `payload_zstd_bytes` (payload body entropy-coded with zstd).

### 2. Decoding (`models.py`, `decoder.py`)

**Model choice.** We avoided training and compared pretrained models zero-shot
(`scripts/model_suitability.py`, same payloads and constraints for every model).
See [Model selection](#model-selection) below.

**Keeping stored text fixed.** The decoder writes the expected output as a
*template*:

```
" Th" GAP(1..2) " progr" GAP(4..5) " disco" GAP(4..5) " a" GAP(0..1) ... ".\n"
```

Fixed strings (separators + stored prefixes) must be reproduced exactly. Each
`GAP` may contain only word characters, within the length range derived from
the header. Beam search runs over the model's normal tokens, but a token is
allowed only if its characters keep the output consistent with the template
(`models.advance`). A token like ` programmer` is fine because it matches
` progr` + 5 gap characters. ` program` + ` ` is rejected because the gap is
too short. Alignment is deterministic: gap characters are `\w` and the next
fixed segment always starts with a non-`\w` character. So the model **cannot
rewrite a prefix, add or drop words, or change punctuation**. It only chooses
suffix characters. The tests in `tests/test_decoder.py` check this with a
random-weight model.

Two search details mattered (both found while debugging the first version):

* **Exact valid-token selection.** The valid tokens at each step are found by
  scanning candidates in descending probability order, limited to tokens whose
  first character can come next. The search never misses a valid token just
  because the model ranks it low. An earlier top-128 shortcut did miss them,
  and it produced garbage like `lookbkjd`.
* **Per-segment beams.** Tokens cover different amounts of text, so a
  hypothesis that has paid for ` fast` must not be pruned by one that has only
  emitted the cheap fragment ` f`. Hypotheses only compete with others at the
  same template segment.

**Context strategies.** Every strategy uses the same model, the same prompt
format, the same few-shot sentences, and the same search. Only the *window* of
compressed words handed to the model changes:

```
Compressed: Th progr disco a unexp prob
Original: The programmer discovered an unexpected problem
```

| strategy | `--context` | window given to the model | requests |
|---|---|---|---|
| A: no context | `0` | the word alone | one per truncated word |
| B: local | `2`, `4`, `8` | the word plus k neighbours (k/2 on each side, fewer at sentence edges) | one per truncated word |
| C: full sentence | `full` | the whole sentence | one per sentence |

The model generates the whole window under the template, and for A/B only the
centre word is kept. Context is bidirectional in every case. Words to the
right are visible as compressed prefixes in the prompt, and words to the left
appear both compressed and restored. Few-shot examples are cut to the same
window size. (A first version gave B a different "restore the bracketed word"
prompt. It was much worse, which mixed up prompt quality with context size, so
it was dropped.)

All requests are batched. A shared prompt prefix (instructions + few-shot
examples) is encoded once and its KV cache reused. Long documents are decoded
sentence by sentence (sentences are found from separators in the payload).

### 3. Adaptive prefixes (`adaptive.py`)

The compressor (which may read the original) starts at 25% and runs the real
decoder on the real serialized payload. It lengthens every wrongly restored
word by one character and repeats until the target accuracy is reached. After
6 rounds, wrong words are stored in full.

Adding characters does **not** always help: with joint decoding, lengthening
one word can break a neighbour. The loop therefore re-checks every word each
round. Termination is still guaranteed: each round that does not stop adds at
least one character, and a fully stored word cannot be restored wrongly. The
benchmark then decodes the final payload again from scratch and reports that
result.

## Project layout

```
neural_compression/
  compressor.py   split, truncate, serialize / parse payloads, size stats
  models.py       model loading, template matching, constrained beam search
  decoder.py      prompts + templates for context strategies A/B/C
  adaptive.py     decoder-in-the-loop adaptive prefix lengths
  benchmark.py    metrics (word/sentence accuracy, CER, WER), runs, CSV/JSON export
  devlog.py       dev-loop run log: entries, comparison with the previous run
  plots.py        matplotlib charts (optional `plots` group)
  cli.py          python -m neural_compression {compress,decompress,benchmark,plot}
  student.py      single-pass student decoder: vocabulary, encoder, candidate scoring, n-gram fallback
  student_train.py  student training data, training loop, validation evaluation
scripts/
  modal_student.py       teacher text generation and student training on Modal (time-capped)
  build_dataset.py       builds data/benchmark.jsonl from public datasets
  model_suitability.py   zero-shot model comparison
  modal_app.py           benchmark on Modal GPUs, one container per configuration
  verify_corpus.py       decode the merged adaptive corpus payload on any device
  build_dev_set.py       builds data/dev_book.txt from Project Gutenberg
  dev.py                 dev loop: score the dev set, log the run, compare with the last one
results/                 CSV/JSON/PNG from the run reported above
  dev/                   dev-loop run log (runs.jsonl) and per-run rows
data/
  benchmark.jsonl  125 sentences, 5 categories (committed)
  dev_book.txt     50 sentences from one novel, the dev set (committed)
  ambiguous.txt    hand-written shared-prefix sentences
tests/
```

## Model selection

Run `scripts/model_suitability.py` to reproduce. It uses 12 dev sentences that
are not in the benchmark, full-sentence context, the same payloads and
constraints for every model, and beam 4. Raw numbers are in
`results/model_suitability.json`.

| model | params | 25% kept | 50% kept | 75% kept | CPU time, 12 sentences @50% |
|---|---|---|---|---|---|
| google/byt5-small (span infilling) | 300M | – | 31.4% | 62.8% | 14 s |
| HuggingFaceTB/SmolLM2-135M | 135M | – | 97.5% | 94.2% | 16 s |
| **HuggingFaceTB/SmolLM2-360M** | 362M | **77.7%** | 98.3% | 99.2% | 27 s |
| Qwen/Qwen2.5-0.5B | 494M | 71.9% | 93.4% | 99.2% | 23 s |
| Qwen/Qwen3-0.6B-Base | 596M | 68.6% | 99.2% | 99.2% | 54 s |
| Qwen/Qwen2.5-1.5B | 1.54B | 72.7% | 99.2% | 99.2% | 51 s |

(The dev set is easy: it uses common words and no names. Benchmark accuracy is
much lower.)

* **ByT5-small looked like the natural fit** (bytes, bidirectional encoder,
  pretrained span infilling), and we tried it first. Each missing suffix
  becomes a sentinel, and decoding is constrained to letters within the same
  length bounds. One pitfall: ByT5 was pretrained with sentinel ids `258 - i`
  (reused byte ids), *not* the tokenizer's `<extra_id_i>` tokens, which produce
  garbage. Even with correct sentinels and constraints it makes up non-words
  (`tomorrie`, `integera`). Its pretraining used rare ~20-byte spans, while our
  task cuts every word.
* **Small causal LMs do much better** once the constraints keep the output on
  track, and the compressed sentence is in the prompt. That prompt gives every
  word its right-hand context too, so the decoding is effectively
  bidirectional.
* **SmolLM2-360M** (Apache-2.0, pinned to commit `f8027fd0ea`) was the most
  accurate in the hardest setting (25%) and is among the fastest, so it is
  the default. Bigger was not better here. `--model` switches to any HF
  causal LM.

Model weights are **not** counted in per-document sizes. SmolLM2-360M is
1.45 GB in fp32 (≈724 MB in bf16), a one-time setup cost that both sides must
share. Loading takes ~2 s from a warm cache.

## Limitations

* **Lossy, and it fails silently.** Errors are fluent, plausible words, not
  visible damage: `need → neat`, `president → presidency`, `lookups → lookup`.
  A reader cannot tell which words are wrong. Sentence-level exact accuracy is
  far lower than word accuracy.
* **Ambiguity is real.** Many prefixes have several valid completions of the
  right length (`th` → the/they/this/then at 50%). Context resolves only part of
  it. The `ambiguous` category is built to stress this.
* **Hallucination.** The template stops the model from rewriting stored text,
  but inside a gap it may still produce non-words (`discothunk`) or a real word
  that changes the meaning. Gap characters are only constrained to `\w`, not to
  a dictionary.
* **Names, rare words, jargon.** Proper nouns and domain terms that the model
  rarely saw cannot be recovered from a short prefix. Numbers are stored
  verbatim by default (truncated digits are pure information loss), which
  lowers the ratio on number-heavy text.
* **Copy bias.** Small LMs like to copy prompt tokens. When an empty suffix is
  allowed (at high retention, a stored `the` might be complete or might be
  `then`), they often just echo the prefix. This is why SmolLM2-135M scored
  lower at 75% than at 50%.
* **Compression is modest.** Separators and capitalisation are stored
  verbatim, and short words cannot shrink below 1 character. Fixed prefixes
  beat zstd only where accuracy is poor. Adaptive prefixes at 100% beat zstd by
  13% on the corpus, but only by counting the 1.45 GB shared model as free.
  An LM driving an arithmetic coder would use the same model far more
  efficiently.
* **Expensive decoding.** ≈100 ms per word on a laptop CPU, 5–50 ms on an A10
  GPU, versus microseconds for zstd. Adaptive compression runs the decoder
  ~5 times per sentence. Windowed context (B) costs more than full-sentence
  context (C) because every word gets its own request that regenerates its
  whole window.
* **Adaptive mode is tied to one decoder.** Its guarantee holds only for the
  exact model, revision, decoder settings, and numeric environment used during
  compression. A different model, beam size, or context setting can
  reconstruct differently. Floating-point differences between machines could in
  principle flip a near-tie.
* **English, ASCII-centric benchmark.** Other scripts work mechanically (the
  format is UTF-8 and `\w` is Unicode-aware) but are untested.

## Reproducibility

* Fixed model revision, greedy-deterministic beam search (no sampling), seeds set
  (`--seed`), fixed few-shot examples (in `decoder.py`), fixed dataset file.
* `results/*/results.json` records the model, revision, parameter count, torch /
  transformers / Python versions, and platform.
* Decoding is deterministic on a given machine (tested). Batched and unbatched
  decoding give identical outputs (tested). Rerunning the adaptive benchmark on
  a different GPU type (A10 vs A10G) gave identical results, and CPU and GPU
  gave identical accuracies on the fixed configurations that were run on both.
* Search settings: beam 4 per template segment, the 2 most advanced segments
  kept, the same for every strategy.
# dynapress
