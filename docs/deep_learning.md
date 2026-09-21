# Track C: deep learning

Track C implements the three requested model families behind the same
`fit(texts, labels)` and `predict(texts)` interface used by the rest of the
application:

- **FFNN:** word unigram/bigram TF-IDF followed by hidden layers of 256 and 64
  ReLU units, dropout, and a three-class output layer;
- **TextCNN:** trainable word embeddings, parallel one-dimensional convolutions
  with widths 2, 3, 4, and 5, global max pooling, dropout, and a classification
  layer;
- **DistilBERT:** full fine-tuning of `distilbert-base-uncased` with a new
  three-class sequence-classification head.

DistilBERT was selected instead of RoBERTa because the brief asks for one of
the two and DistilBERT is much easier to reproduce on Colab or a modest GPU.
It has fewer parameters and lower memory use while retaining pretrained
contextual representations.

## Leakage controls

Every model receives only the training portion selected by the shared
per-repository evaluator. It then creates an internal, stratified validation
split for early stopping. Normalized duplicate texts are kept together in
that split. Crucially, the split is made **before** fitting the FFNN's TF-IDF
vectorizer or the CNN's vocabulary, so internal validation text cannot affect
the representation.

The official test data is never used for early stopping. For model selection
or hyperparameter tuning, run duplicate-safe grouped cross-validation on the
official training split. Once settings are fixed, use official mode once.

## Training and overfitting diagnostics

The FFNN and CNN use AdamW, gradient clipping, dropout, and validation-loss
early stopping. DistilBERT uses AdamW with linear warmup and fine-tunes the
complete encoder. Each result JSON records, for every repository and fold:

- training and internal-validation sample counts;
- device and trainable parameter count;
- training loss, validation loss, validation accuracy, and validation
  weighted F1 for every epoch;
- the restored best epoch and whether training stopped early.

Use `--plots` to save learning curves and confusion matrices under
`results/figures/deep_learning/`.

## Installation and commands

Install the optional dependencies:

```bash
make install-dl
```

Run the two lightweight neural models first (CPU is sufficient):

```bash
make deep-neural
```

Fine-tune DistilBERT (a CUDA GPU or Colab GPU is strongly recommended):

```bash
make deep-transformer
```

Run all three official experiments:

```bash
make deep
```

For training-only model selection:

```bash
python examples/deep_learning.py \
  --mode cross-validation \
  --models ffnn cnn \
  --plots
```

Five-fold DistilBERT cross-validation trains 25 transformer models because the
competition protocol uses one model per repository. It is valid but expensive;
run it only with an appropriate GPU budget. A shorter smoke run can override
epochs and folds, for example `--models distilbert --epochs 1 --n-splits 2`.
The `make deep-cv` target therefore defaults to FFNN and CNN; use
`make deep-cv DEEP_MODELS=distilbert` only when that compute cost is intended.

## Default preprocessing

The FFNN and CNN use the training-only Track B choice: raw normalized text,
title weight 3, and a 400-word cap. DistilBERT uses light Markdown/code cleanup,
title weight 1, and tokenizer truncation at 384 subword tokens. All defaults
and command-line overrides are written into result metadata.

## Outputs

Experiments write versioned shared-schema JSON to `results/deep_learning/` and
CSV/Markdown summaries to `results/tables/deep_learning.*`. The common table
builder automatically includes completed official Track C runs:

```bash
make results
make report-tables
```

Do not report a score until its JSON artifact has been generated on the actual
dataset. CPU and GPU kernels can produce small numerical differences even with
fixed random seeds, so the saved device metadata should accompany neural
results.
