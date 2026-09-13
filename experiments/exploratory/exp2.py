# -*- coding: utf-8 -*-
"""Four probe experiments that the research plan will be built on."""
import re, sys, io
import pandas as pd, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, classification_report

import os
from pathlib import Path

# Cached competition CSVs, downloaded by `make data` (see ai4se.loader).
D = str(Path(__file__).resolve().parents[2] / "data" / "raw")
tr = pd.read_csv(os.path.join(D, "issues_train.csv")).fillna("")
te = pd.read_csv(os.path.join(D, "issues_test.csv")).fillna("")
REPOS = sorted(tr.repo.unique())
def sec(t): print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)

def txt(d): return (d.title + " \n " + d.body).str.slice(0, 6000)

def cross_repo(fit_txt_tr, y_tr, repo_tr, fit_txt_te, y_te, repo_te, global_model=True, clf=None):
    """Return mean-of-5-repo micro F1."""
    out = {}
    if global_model:
        pipe = make_pipeline(TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2),
                             clf or LinearSVC(C=1.0, class_weight="balanced"))
        pipe.fit(fit_txt_tr, y_tr)
        pred = pipe.predict(fit_txt_te)
        for r in REPOS:
            m = (repo_te == r).values
            out[r] = f1_score(y_te[m], pred[m], average="micro")
    return np.mean(list(out.values())), out

# ---------------------------------------------------------------- EXP A
sec("EXP A — HOW SHOULD WE TREAT CODE / MARKDOWN NOISE?")
CODE = re.compile(r"```.*?```", re.S)
INLINE = re.compile(r"`[^`\n]+`")
URL = re.compile(r"https?://\S+")
IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)|\S+\.(?:png|jpg|jpeg|gif)\b")
HTMLC = re.compile(r"<!--.*?-->", re.S)
TRACE = re.compile(r"(?m)^\s*(?:at\s+[\w$.]+\(|File \"|Traceback).*$")

def v_raw(s): return s
def v_strip(s):   # delete noise entirely
    s = HTMLC.sub(" ", s); s = CODE.sub(" ", s); s = IMG.sub(" ", s)
    s = URL.sub(" ", s); s = INLINE.sub(" ", s); return s
def v_mask(s):    # replace noise with typed placeholders (keeps the *signal of presence*)
    s = HTMLC.sub(" ", s); s = TRACE.sub(" xxtrace ", s); s = CODE.sub(" xxcode ", s)
    s = IMG.sub(" xximg ", s); s = URL.sub(" xxurl ", s); s = INLINE.sub(" xxinline ", s); return s

for name, fn in [("raw (no cleaning)", v_raw), ("strip noise away", v_strip), ("mask -> typed tokens", v_mask)]:
    a = txt(tr).map(fn); b = txt(te).map(fn)
    m, per = cross_repo(a, tr.label, tr.repo, b, te.label, te.repo)
    print(f"{name:24s} cross-repo F1 = {m:.4f}")

# ---------------------------------------------------------------- EXP B
sec("EXP B — HOW FAR DOES *PURE STRUCTURE* GET YOU? (no words at all)")
FEATS = {
    "has_code": r"```", "has_inline": r"`[^`\n]+`", "has_url": r"https?://",
    "has_img": r"!\[|\.png|\.gif|\.jpg", "has_htmlcomment": r"<!--",
    "has_heading": r"(?m)^#{1,6} ", "has_checkbox": r"- \[[ xX]\]",
    "has_trace": r"(?i)(traceback|at [\w$.]+\(|Exception|Error:)",
    "has_version": r"(?i)(version|OS|platform|browser)\s*[:=]",
    "has_mention": r"(?m)(^|\s)@\w+", "has_issueref": r"#\d{2,6}",
    "has_qmark_title": r"\?", "starts_how": r"(?i)^(how|what|why|is|can|does|any)\b",
}
def structural(d):
    X = pd.DataFrame(index=d.index)
    for k, p in FEATS.items():
        src = d.title if k in ("has_qmark_title", "starts_how") else d.body
        X[k] = src.str.contains(p, regex=True).astype(int)
    X["log_len"] = np.log1p(d.body.str.len())
    X["n_lines"] = np.log1p(d.body.str.count("\n"))
    X["title_len"] = d.title.str.split().str.len()
    return X.values

Xs_tr, Xs_te = structural(tr), structural(te)
clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xs_tr, tr.label)
pred = clf.predict(Xs_te)
per = {r: f1_score(te.label[(te.repo == r).values], pred[(te.repo == r).values], average="micro") for r in REPOS}
print(f"structure-only (15 hand-crafted flags) cross-repo F1 = {np.mean(list(per.values())):.4f}")
print("  per repo:", {k: round(v, 3) for k, v in per.items()})
print("\n  most informative structural cues (coef, one-vs-rest):")
names = list(FEATS) + ["log_len", "n_lines", "title_len"]
for i, lab in enumerate(clf.classes_):
    top = np.argsort(clf.coef_[i])[::-1][:5]
    print(f"   {lab:9s} +: " + ", ".join(f"{names[j]}({clf.coef_[i][j]:+.2f})" for j in top))

# ---------------------------------------------------------------- EXP C
sec("EXP C — IS THE OFFICIAL SPLIT OPTIMISTIC? (random vs time-aware)")
all_df = pd.concat([tr.assign(split="train"), te.assign(split="test")], ignore_index=True)
all_df["ts"] = pd.to_datetime(all_df.created_at)
# (i) official random split, global model
m_rand, _ = cross_repo(txt(tr), tr.label, tr.repo, txt(te), te.label, te.repo)
print(f"(i)  official random split          cross-repo F1 = {m_rand:.4f}")
# (ii) time-aware: within each repo, oldest 50% -> train, newest 50% -> test
tr2, te2 = [], []
for r in REPOS:
    sub = all_df[all_df.repo == r].sort_values("ts")
    h = len(sub) // 2
    tr2.append(sub.iloc[:h]); te2.append(sub.iloc[h:])
tr2 = pd.concat(tr2); te2 = pd.concat(te2)
m_time, per_time = cross_repo(txt(tr2), tr2.label, tr2.repo, txt(te2), te2.label, te2.repo)
print(f"(ii) time-aware split (past->future) cross-repo F1 = {m_time:.4f}   Δ = {m_time - m_rand:+.4f}")
print("     per repo:", {k: round(v, 3) for k, v in per_time.items()})
print("     train-class balance under time split:", tr2.label.value_counts().to_dict())

# ---------------------------------------------------------------- EXP D
sec("EXP D — CROSS-PROJECT TRANSFER (leave-one-repo-out)")
pool = pd.concat([tr, te.assign()], ignore_index=False)  # use train only for fairness
print("train on 4 repos -> test on held-out repo's TEST set (no in-domain data at all):")
loro = {}
for r in REPOS:
    a = tr[tr.repo != r]; b = te[te.repo == r]
    pipe = make_pipeline(TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2),
                         LinearSVC(C=1.0, class_weight="balanced")).fit(txt(a), a.label)
    loro[r] = f1_score(b.label, pipe.predict(txt(b)), average="micro")
print("  ", {k: round(v, 3) for k, v in loro.items()}, "=> mean", round(np.mean(list(loro.values())), 4))
print("  (compare: in-domain per-repo model = 0.7507, global 5-repo model = 0.7640)")

# ---------------------------------------------------------------- EXP E
sec("EXP E — LEARNING CURVE: how much does each repo's 300 labels actually buy?")
for n in [25, 50, 100, 200, 300]:
    accs = []
    for r in REPOS:
        a = tr[tr.repo == r].groupby("label", group_keys=False).apply(lambda x: x.sample(min(len(x), n // 3), random_state=0))
        b = te[te.repo == r]
        pipe = make_pipeline(TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=1),
                             LinearSVC(C=1.0, class_weight="balanced")).fit(txt(a), a.label)
        accs.append(f1_score(b.label, pipe.predict(txt(b)), average="micro"))
    print(f"  {n:3d} labelled issues/repo -> cross-repo F1 = {np.mean(accs):.4f}")
