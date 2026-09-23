# RUNBOOK

Everything needed to go from a fresh clone to every result in the report.

This repository contains **source only**. No data, no trained models, no saved
results — all of it regenerates from the commands below. The dataset downloads
automatically on first use.

---

## 0. Requirements

- **Python 3.10 or newer** (3.11 recommended)
- ~3 GB disk for dependencies and cached models
- Internet access on first run (dataset + encoder downloads)
- **A GPU is optional.** Everything runs on a CPU except the SetFit stages;
  see step 5.

A `.devcontainer/` is included but is *not* the supported path — use a
virtualenv as below.

---

## 1. Set up the environment

```bash
git clone <your-repo-url>
cd ai4se-project

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

make install
```

`make install` runs `pip install -e ".[all]"` and downloads the NLTK corpora.
It takes a few minutes, mostly PyTorch.

**If PyTorch is a problem** (slow download, no GPU, disk pressure), install the
CPU build first — a few hundred MB instead of ~2 GB — then the rest:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
make install
```

**If you want to skip the heavy tracks entirely:**

```bash
make install-core      # no torch, no sentence-transformers
```

Tracks A, B and D still work; Track C and the embedding models are unavailable
and their tests skip cleanly.

**With a GPU**, for the SetFit reproduction:

```bash
make install-gpu       # adds setfit, datasets, accelerate
```

### Verify

```bash
make test
```

Expected: `74 passed` with everything installed, or `55 passed, 8 skipped` with
`install-core`. Any *failure* means the environment is wrong — stop and fix it
before going further.

---

## 2. Get the data

```bash
make data
```

Downloads the NLBSE'24 competition splits (3,000 issues, ~7 MB) into
`data/raw/`. Cached, so it runs once. Everything after this works offline
except the SetFit stages, which fetch encoder weights on first use.

---

## 3. Run the experiments

```bash
python scripts/run_experiments.py --all
```

**About 20 minutes.** Writes JSON to `results/tables/` and figures to
`results/figures/`.

Stages can be run individually, and each skips work already on disk unless
`--force` is given. Timings below are from a real run on a desktop with a GPU;
the ablation and grid stages are CPU-bound scikit-learn work and dominate,
while the neural stage is fast when a GPU is present and takes a few minutes
without one.

| Stage | Command | Observed |
|---|---|---|
| Reference floors | `--floors` | ~5 s |
| Preprocessing ablation | `--ablation` | ~8 min |
| Hyperparameter grid | `--grid` | ~7 min |
| Track B, classical ML | `--classical` | ~3 min |
| Track C, neural | `--neural` | ~15 s on GPU, ~2 min on CPU |
| Track D2, frozen encoders | `--embeddings` | ~1 min (first run downloads ~500 MB) |
| Ensembles | `--ensemble` | ~1 min |
| Error analysis | `--errors` | ~15 s |

`python scripts/run_experiments.py --list` prints them.

---

## 4. Run the notebooks

```bash
make notebooks
```

Executes all five in order and writes their outputs back in place. About 10
minutes. They read the saved JSON from step 3 rather than re-training, so run
step 3 first.

| Notebook | Content |
|---|---|
| `01_data_and_eda.ipynb` | Dataset, persistence layer, exploratory analysis |
| `02_evaluation_protocol.ipynb` | Metrics, k-fold, competition protocol, floors |
| `03_classical_models.ipynb` | Track B, preprocessing ablation, grid search |
| `04_neural_models.ipynb` | Track C, learning curves, early stopping |
| `05_final_comparison.ipynb` | Leaderboard, analysis, error analysis |

To work in them interactively instead:

```bash
make lab           # Jupyter Lab on port 8888
```

---

## 5. The SetFit reproduction (needs a GPU)

This is the only part that is impractical on a CPU — it fine-tunes an encoder
five times, once per repository.

```bash
python scripts/run_experiments.py --embeddings --setfit
python scripts/run_experiments.py --ensemble  --setfit
```

On a 12 GB card, from a real run: about **50 minutes** for the three SetFit
variants (MiniLM ~10 min, MiniLM-matched ~6 min, MPNet ~28 min) and **25
minutes** for the two SetFit ensembles, which retrain their SetFit member.

### If it runs out of GPU memory

Contrastive training embeds **both** sentences of every pair, so peak memory is
about twice plain fine-tuning. Knobs, in order of effect:

| Knob | Default | Effect |
|---|---|---|
| `max_seq_length` | 128 | **Dominant** — attention cost is quadratic in it, so halving cuts peak memory ~4× |
| `batch_size` | 16 (8 for MPNet) | Roughly linear |
| `num_iterations` | 20 | Runtime only, not peak memory |

Edit the factories in `src/ai4se/embeddings.py`, or construct directly:

```python
from ai4se.embeddings import STRONG_ENCODER, SetFitClassifier
SetFitClassifier(STRONG_ENCODER, batch_size=4, max_seq_length=96)
```

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` also helps with
fragmentation.

### Clean up afterwards

SetFit's trainer writes checkpoints into the working directory — hundreds of MB
per run. They are git-ignored, but delete them anyway:

```bash
make clean
```

---

## 6. Build the report

`results/tables/*.tex` are generated, so the report's numbers cannot drift from
what the code produced:

```latex
\input{../results/tables/final_leaderboard.tex}
```

```bash
make report        # latexmk in report/
```

Needs a LaTeX toolchain (`latexmk` + TeX Live) installed separately; it is not
a Python dependency.

---

## Full sequence, copy-paste

```bash
python3 -m venv .venv && source .venv/bin/activate
make install
make test
make data
python scripts/run_experiments.py --all     # ~20 min
make notebooks                              # ~10 min
# with a GPU:
python scripts/run_experiments.py --embeddings --setfit
python scripts/run_experiments.py --ensemble  --setfit
make clean
```

---

## Every make target

```
make help          list all targets
make venv          create a virtualenv
make install       everything, all tracks
make install-core  no torch, no sentence-transformers
make install-gpu   adds the SetFit extra
make data          download and cache the dataset
make experiments   run every experiment stage
make eda           execute notebook 01
make evaluate      execute notebook 02
make models        execute notebooks 03 and 04
make results       execute notebook 05
make notebooks     execute all five in order
make lab           start Jupyter Lab
make test          run the test suite
make check         print the persistence-equivalence table
make lint          ruff check + format
make tree          recreate any missing directories
make clean         remove caches, checkpoints and generated data
make report        compile the LaTeX report
```

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'ai4se'`** — the virtualenv is not
active, or `make install` has not run. `pip install -e .` must have been run
from the repository root.

**`ModuleNotFoundError: No module named 'sklearn'` / `'torch'`** — you used
`make install-core`, or the install failed partway. Run `make install`.

**NLTK `LookupError`** — the corpora are missing:
`python -m nltk.downloader stopwords wordnet omw-1.4`.

**`Duplicate model names in results/tables`** — two saved results carry the
same model name, usually after renaming one. Delete the stale file and re-run
that stage.

**Notebooks show no output** — expected in a fresh clone; outputs are stripped
from version control. Run step 3 then step 4.

**Results differ slightly from the report.** How much drift is acceptable
depends on the model:

- **scikit-learn pipelines reproduce exactly**, to four decimal places. Any
  difference there means the environment or the data differs.
- **Neural models (FFNN, CNN) reproduce exactly on the same device** but not
  across devices. The CNN scored 0.7578 on a CPU and 0.7438 on a GPU from
  identical code and seed — CPU and GPU use different kernels that accumulate
  in a different order, and no seeding setting removes that. Compare GPU
  numbers with GPU numbers.
- **SetFit varies by roughly 0.01 run to run**, even on one device.

So: a third-decimal difference is always fine. A second-decimal difference is
expected between a CPU and a GPU run of the neural models, and between repeated
SetFit runs — but is a real problem anywhere else.
