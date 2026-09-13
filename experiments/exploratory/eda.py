# -*- coding: utf-8 -*-
"""Preliminary EDA on the NLBSE'24 issue-report-classification dataset."""
import re, sys, io, hashlib
import pandas as pd, numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os
from pathlib import Path

# Cached competition CSVs, downloaded by `make data` (see ai4se.loader).
D = str(Path(__file__).resolve().parents[2] / "data" / "raw")
tr = pd.read_csv(os.path.join(D, "issues_train.csv"))
te = pd.read_csv(os.path.join(D, "issues_test.csv"))

def sec(t): print("\n" + "=" * 70 + "\n" + t + "\n" + "=" * 70)

sec("1. SHAPE / COLUMNS")
print("train", tr.shape, "| test", te.shape)
print("cols:", list(tr.columns))
print("nulls train:\n", tr.isna().sum().to_string())
print("nulls test:\n", te.isna().sum().to_string())

sec("2. LABEL x REPO (train)")
print(pd.crosstab(tr.repo, tr.label, margins=True).to_string())
sec("2b. LABEL x REPO (test)")
print(pd.crosstab(te.repo, te.label, margins=True).to_string())

sec("3. TEXT LENGTH (chars / whitespace tokens)")
for name, df in [("train", tr), ("test", te)]:
    df["title"] = df.title.fillna(""); df["body"] = df.body.fillna("")
    df["n_title"] = df.title.str.len()
    df["n_body"] = df.body.str.len()
    df["tok"] = (df.title + " " + df.body).str.split().str.len()
    print(f"\n[{name}] body chars:", df.n_body.describe(percentiles=[.5,.75,.9,.95,.99]).round(1).to_dict())
    print(f"[{name}] title+body whitespace tokens:", df.tok.describe(percentiles=[.5,.75,.9,.95,.99]).round(1).to_dict())
    print(f"[{name}] empty body: {(df.n_body==0).sum()} ({(df.n_body==0).mean():.1%})")
    print(f"[{name}] >512 tokens: {(df.tok>512).mean():.1%} | >256: {(df.tok>256).mean():.1%} | >128: {(df.tok>128).mean():.1%}")

sec("3b. TOKENS BY LABEL (train, median / mean)")
print(tr.groupby("label").tok.agg(["median", "mean", "max"]).round(1).to_string())
print("\nby repo:")
print(tr.groupby("repo").tok.agg(["median", "mean"]).round(1).to_string())

sec("4. STRUCTURAL NOISE IN BODY (train, % of issues)")
pat = {
    "fenced code block ```": r"```",
    "indented/inline code `x`": r"`[^`\n]+`",
    "URL": r"https?://",
    "image / attachment": r"!\[|user-images\.githubusercontent|\.png|\.gif|\.jpg",
    "HTML comment <!-- -->": r"<!--",
    "markdown heading ###": r"(?m)^#{1,6} ",
    "checkbox - [ ]": r"- \[[ xX]\]",
    "stack trace hint": r"(?i)(traceback|at [\w\.$]+\(|Exception|Error:)\s",
    "version/env block": r"(?i)(version|OS|platform|browser)\s*[:=]",
    "@mention": r"(?m)(^|\s)@\w+",
    "issue ref #123": r"#\d{2,6}",
}
for k, p in pat.items():
    print(f"{k:28s} {tr.body.str.contains(p, regex=True, na=False).mean():6.1%}")

sec("4b. NOISE BY LABEL (fenced code / checkbox / stack-trace)")
for k in ["fenced code block ```", "checkbox - [ ]", "stack trace hint", "image / attachment"]:
    s = tr.groupby("label").body.apply(lambda x: x.str.contains(pat[k], regex=True, na=False).mean())
    print(f"{k:26s}", {i: f"{v:.1%}" for i, v in s.items()})

sec("5. DUPLICATES / LEAKAGE")
def norm(s): return re.sub(r"\W+", " ", str(s).lower()).strip()
tr["k"] = (tr.title + " " + tr.body).map(norm).map(lambda x: hashlib.md5(x.encode()).hexdigest())
te["k"] = (te.title + " " + te.body).map(norm).map(lambda x: hashlib.md5(x.encode()).hexdigest())
print("exact dup inside train :", tr.k.duplicated().sum())
print("exact dup inside test  :", te.k.duplicated().sum())
print("train<->test overlap   :", len(set(tr.k) & set(te.k)))
tt = set(tr.title.map(norm)); print("identical TITLE train&test:", te.title.map(norm).isin(tt).sum())

sec("6. TEMPORAL STRUCTURE (is the split random or time-based?)")
for name, df in [("train", tr), ("test", te)]:
    d = pd.to_datetime(df.created_at)
    print(f"[{name}] {d.min()}  ->  {d.max()}")
print("\nper repo (train min/max | test min/max):")
for r in sorted(tr.repo.unique()):
    a = pd.to_datetime(tr[tr.repo == r].created_at); b = pd.to_datetime(te[te.repo == r].created_at)
    print(f"  {r:26s} train {a.min().date()}..{a.max().date()}   test {b.min().date()}..{b.max().date()}")
print("\ntrain year dist:", pd.to_datetime(tr.created_at).dt.year.value_counts().sort_index().to_dict())
print("test  year dist:", pd.to_datetime(te.created_at).dt.year.value_counts().sort_index().to_dict())

sec("7. MOST DISTINCTIVE TERMS PER CLASS (log-odds w/ informative prior, train)")
from sklearn.feature_extraction.text import CountVectorizer
txt = (tr.title + " " + tr.body).str.lower()
cv = CountVectorizer(max_features=20000, stop_words="english", token_pattern=r"[a-z][a-z\-]{2,}")
X = cv.fit_transform(txt); vocab = np.array(cv.get_feature_names_out())
tot = np.asarray(X.sum(0)).ravel(); N = tot.sum()
for lab in sorted(tr.label.unique()):
    m = (tr.label == lab).values
    c = np.asarray(X[m].sum(0)).ravel(); n = c.sum()
    a0 = 0.01 * N
    lo = np.log((c + a0 * tot / N) / (n + a0 - c - a0 * tot / N)) - np.log((tot - c + a0 * tot / N) / (N - n + a0 - (tot - c) - a0 * tot / N))
    var = 1.0 / (c + a0 * tot / N) + 1.0 / (tot - c + a0 * tot / N)
    z = lo / np.sqrt(var)
    keep = tot >= 20
    idx = np.argsort(np.where(keep, z, -1e9))[::-1][:18]
    print(f"\n[{lab}] " + ", ".join(vocab[idx]))

sec("8. CROSS-REPO LABEL SEMANTICS: does a term mean the same thing in every repo?")
for r in sorted(tr.repo.unique()):
    sub = tr[tr.repo == r]
    t = (sub.title + " " + sub.body).str.lower()
    cv2 = CountVectorizer(max_features=5000, stop_words="english", token_pattern=r"[a-z][a-z\-]{2,}")
    try:
        X2 = cv2.fit_transform(t); v2 = np.array(cv2.get_feature_names_out())
        tot2 = np.asarray(X2.sum(0)).ravel()
        m = (sub.label == "question").values
        c2 = np.asarray(X2[m].sum(0)).ravel()
        r2 = (c2 + 1) / (tot2 + 3)
        keep = tot2 >= 10
        idx = np.argsort(np.where(keep, r2, -1))[::-1][:8]
        print(f"{r:26s} question-markers: " + ", ".join(v2[idx]))
    except ValueError:
        pass

sec("9. TITLE-ONLY SIGNAL (how much does the body actually add?)")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, classification_report
def run(field_fn, name, per_repo_model):
    scores = {}
    for r in sorted(tr.repo.unique()):
        a = tr[tr.repo == r]; b = te[te.repo == r]
        if per_repo_model:
            Xtr, ytr = field_fn(a), a.label
        else:
            Xtr, ytr = field_fn(tr), tr.label
        pipe = make_pipeline(TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_features=200000),
                             LinearSVC(C=1.0, class_weight="balanced"))
        pipe.fit(Xtr, ytr)
        p = pipe.predict(field_fn(b))
        scores[r] = dict(micro=f1_score(b.label, p, average="micro"), macro=f1_score(b.label, p, average="macro"))
    mi = np.mean([v["micro"] for v in scores.values()]); ma = np.mean([v["macro"] for v in scores.values()])
    print(f"\n{name}: cross-repo micro-F1 = {mi:.4f} | macro-F1 = {ma:.4f}")
    for r, v in scores.items(): print(f"   {r:26s} micro {v['micro']:.4f}  macro {v['macro']:.4f}")
    return mi

TITLE = lambda d: d.title.fillna("")
BODY = lambda d: d.body.fillna("").str.slice(0, 4000)
BOTH = lambda d: (d.title.fillna("") + " \n " + d.body.fillna("")).str.slice(0, 4000)
run(TITLE, "TF-IDF+SVM  TITLE only     (per-repo models)", True)
run(BODY, "TF-IDF+SVM  BODY only      (per-repo models)", True)
mi_both = run(BOTH, "TF-IDF+SVM  TITLE+BODY    (per-repo models)", True)
run(BOTH, "TF-IDF+SVM  TITLE+BODY    (ONE global model)", False)

sec("10. CONFUSION OF THE STRONGEST CLASSICAL BASELINE (pooled over repos)")
preds, golds = [], []
for r in sorted(tr.repo.unique()):
    a = tr[tr.repo == r]; b = te[te.repo == r]
    pipe = make_pipeline(TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2),
                         LinearSVC(C=1.0, class_weight="balanced"))
    pipe.fit(BOTH(a), a.label)
    preds += list(pipe.predict(BOTH(b))); golds += list(b.label)
print(pd.crosstab(pd.Series(golds, name="gold"), pd.Series(preds, name="pred")).to_string())
print()
print(classification_report(golds, preds, digits=4))
print(f"\nSetFit baseline (official README): 0.8270 cross-repo F1")
