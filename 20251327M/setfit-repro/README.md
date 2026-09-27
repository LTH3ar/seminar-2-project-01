# SetFit reproduction and improvement — NLBSE'24

Standalone. Nothing here touches `seminar-2-project-01`; copy results in
yourself once you are satisfied with them.

Two phases, in order:

1. **Reproduce the published baseline** at its own configuration. This is not
   optional and it is not a formality — see *Why phase 1 comes first*.
2. **Try to improve on it**, selecting every choice on validation data carved
   out of the training split, then reading the test split once.

---

## Why phase 1 comes first

The competition publishes aggregate scores but **not per-issue predictions**.
Without them no paired significance test against the baseline is possible, and
a leaderboard delta of one or two points proves nothing on this benchmark.
Running the baseline ourselves gives us its predictions, and from that point on
every comparison is testable.

There is a second reason. The team's existing report runs MPNet at
`batch_size=8, max_seq_length=128` because it would not fit in 12 GB. The
official configuration is `batch_size=16` at the encoder's own 384 tokens. The
report itself measures that the reduced settings cost **0.0166** — which is
almost exactly the gap between its 0.8108 and the published 0.8270. On a card
with enough memory that gap is likely to close on its own, and the number
currently written up as "our method is weaker" turns out to be "the baseline
was starved of memory".

---

## What counts as an improvement

Three numbers to keep in view:

| | F1 |
|---|---|
| SetFit baseline, committed result file | 0.8240 |
| SetFit baseline, competition README | 0.8270 |
| Best published result on this task (fine-tuned GPT-3.5) | **0.828** |

In two years the field has moved this benchmark by **one tenth of a point**,
and a fine-tuned GPT-4o scored *below* the baseline. Treat it as close to
saturated.

Measured properties of the evaluation itself:

- Changing only the random seed moves the score by up to **2.4 points**.
- The test set resolves differences of about **3.0 points** at 80% power.

So `DECISION_THRESHOLD` is 3.0 points. Below it, `run_phase2.py` reports *no
detectable difference* and will not call anything an improvement. Beating the
baseline by a defensible margin means reaching roughly **0.857** — which would
also be about three points clear of the best published result. That is a high
bar and it is meant to be.

---

## Rules, and the one that is ambiguous

From the competition README:

> "Participants must train a multi-class classifier for each of the 5
> projects, i.e., we expect 5 classifiers per model submission."

> "Pretrained models are permitted but can only be finetuned on the given
> training set." — "Any inputs or features used to create or finetune the
> classifier, must be derived from the provided training set."

Two consequences:

**Any pretrained encoder is allowed.** Only the fine-tuning data is
restricted. Swapping `all-mpnet-base-v2` (2021) for a newer encoder is legal.

**The rules never say each classifier may see only its own project's 300
issues.** They require five classifiers and provide one undivided 1,500-issue
training set. Fine-tuning a single encoder on all of it and fitting five
per-project heads is legal *by the letter* — and it is exactly the kind of
ambiguity that should be asked about rather than quietly exploited.

That configuration is therefore gated behind `--allow-global` and skipped by
default. **Ask the lecturer or the organisers before using it, and report both
scopes either way.**

### Contamination

The test set has been public on GitHub since 2024, and the issues themselves
are public GitHub text from 2016–2023. An encoder pretrained after 2024 may
have seen them. `BAAI/bge-base-en-v1.5` and `intfloat/e5-base-v2` therefore
print a warning and carry it into their results file. If one of them wins,
that belongs in Threats to Validity, not in the abstract.

---

## Running it on RunPod

Pick a pod with **≥24 GB** VRAM. The official configuration at batch 16 and
384 tokens does not fit comfortably in 16 GB. A 24 GB card runs a seed in
roughly 15 minutes; 40 GB or more lets you raise the batch size for the
`global_pairs` configuration without gradient accumulation.

```bash
# 1. CHECK torch before touching it. RunPod's PyTorch templates ship a build
#    already matched to the pod's CUDA, and reinstalling can replace it with
#    one that cannot see the GPU.
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
#    Only if that fails or prints False:
#    pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2. everything else. Current releases, not the organisers' 2024 generation:
#    that generation does not install on Python 3.12 and is not safe with
#    torch 2.11. The drift is recorded in every results file under
#    "versions" -- see requirements.txt for why that is the honest handling.
pip install -r requirements.txt
python -c "import setfit,sentence_transformers,datasets,transformers as t; print('setfit',setfit.__version__,'| st',sentence_transformers.__version__,'| transformers',t.__version__,'| datasets',datasets.__version__)"

# 3. phase 1 -- the reproduction
python run_phase1.py --seeds 42

# 4. once that looks right, five seeds for a variance estimate
python run_phase1.py --seeds 41 42 43 44 45 --skip-existing

# 5. phase 2 -- selection, on validation data only
python run_phase2.py select

# 6. phase 2 -- one configuration, on the test split, once
python run_phase2.py final --config <winner>
```

Results are written to `results/` after every run, so an interruption costs
one configuration rather than the session. The dataset downloads itself into
`data/` on first use.

### Budget

| Stage | Runs | Rough time on a 24 GB card |
|---|---|---|
| Phase 1, one seed | 5 models | ~15 min |
| Phase 1, five seeds | 25 models | ~75 min |
| Phase 2 selection, 9 candidates | 45 models | ~2 h |
| Phase 2 final | 5 models | ~15 min |

About 3.5 hours of GPU time for the full sequence.

---

## What the code will not let you do

- **Select on the test split.** `run_phase2.py select` never reads it;
  `final` reads it once per configuration and says so on stdout.
- **Call a small gain an improvement.** Anything inside 3.0 points is
  reported as *no detectable difference*, with the McNemar counts and the
  Holm-adjusted p-values next to it.
- **Report one seed.** Phase 1 prints the spread across seeds and how much
  reporting only the best one would have inflated the number.
- **Use the ambiguous training scope silently.** It is gated and it warns.
- **Enjoy a contaminated encoder quietly.** The warning is printed and stored.

---

## Layout

```
src/data.py         Official CSVs, text assembled as the organisers assemble it
src/metrics.py      The competition metric: weighted-avg F1 per project, mean of 5
src/stats.py        Exact McNemar, Holm, stratified bootstrap, detectable difference
src/runner.py       SetFit training under a named configuration
src/preprocess.py   Text variants for the issue-template lever
run_phase1.py       Reproduce the baseline
run_phase2.py       select | final
```
