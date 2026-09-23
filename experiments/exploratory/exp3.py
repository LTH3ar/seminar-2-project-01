# -*- coding: utf-8 -*-
# ruff: noqa  -- verbatim exploratory script, kept for provenance (see README)
"""Is there a TEMPORAL CONFOUND baked into the NLBSE'24 benchmark?"""
import warnings, sys, io
warnings.filterwarnings("ignore")
import pandas as pd, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold

import os
from pathlib import Path

# Cached competition CSVs, downloaded by `make data` (see ai4se.loader).
D = str(Path(__file__).resolve().parents[2] / "data" / "raw")
tr = pd.read_csv(os.path.join(D, "issues_train.csv")).fillna("")
te = pd.read_csv(os.path.join(D, "issues_test.csv")).fillna("")
al = pd.concat([tr, te], ignore_index=True)
al["ts"] = pd.to_datetime(al.created_at)
al["t"] = al.ts.astype("int64") / 1e9
REPOS = sorted(al.repo.unique())
def sec(t): print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)
def txt(d): return (d.title + " \n " + d.body).str.slice(0, 6000)

sec("F1 — WHEN WAS EACH CLASS SAMPLED FROM? (median creation date per repo x label)")
piv = al.pivot_table(index="repo", columns="label", values="ts", aggfunc="median")
print(piv.apply(lambda c: c.dt.strftime("%Y-%m-%d")).to_string())
print("\nspan in days between earliest-median and latest-median class, per repo:")
for r in REPOS:
    s = piv.loc[r]
    print(f"  {r:24s} {(s.max() - s.min()).days:>5d} days   (earliest={s.idxmin()}, latest={s.idxmax()})")

sec("F2 — CAN THE LABEL BE PREDICTED FROM THE TIMESTAMP ALONE? (no text!)")
print("5-fold CV inside each repo, feature = created_at only. Chance level = 0.333")
tot = []
for r in REPOS:
    s = al[al.repo == r]
    X = s[["t"]].values; y = s.label.values
    p = cross_val_predict(GradientBoostingClassifier(random_state=0), X, y,
                          cv=StratifiedKFold(5, shuffle=True, random_state=0))
    f = f1_score(y, p, average="micro"); tot.append(f)
    print(f"  {r:24s} F1(timestamp only) = {f:.3f}")
print(f"  {'MEAN':24s} F1(timestamp only) = {np.mean(tot):.3f}   <-- chance would be 0.333")

sec("F3 — CLEAN TIME-AWARE SPLIT (balance preserved: oldest 50% vs newest 50% WITHIN each repo x class)")
a, b = [], []
for (r, l), g in al.groupby(["repo", "label"]):
    g = g.sort_values("ts"); h = len(g) // 2
    a.append(g.iloc[:h]); b.append(g.iloc[h:])
a = pd.concat(a); b = pd.concat(b)
print("train balance:", a.label.value_counts().to_dict(), "| test balance:", b.label.value_counts().to_dict())
pipe = make_pipeline(TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2),
                     LinearSVC(class_weight="balanced")).fit(txt(a), a.label)
pr = pipe.predict(txt(b))
per = {r: f1_score(b.label[(b.repo == r).values], pr[(b.repo == r).values], average="micro") for r in REPOS}
print(f"time-aware (balanced) cross-repo F1 = {np.mean(list(per.values())):.4f}")
print("  per repo:", {k: round(v, 3) for k, v in per.items()})

# matched random baseline, same sizes
rng = np.random.RandomState(0)
a2, b2 = [], []
for (r, l), g in al.groupby(["repo", "label"]):
    g = g.sample(frac=1, random_state=0); h = len(g) // 2
    a2.append(g.iloc[:h]); b2.append(g.iloc[h:])
a2 = pd.concat(a2); b2 = pd.concat(b2)
pipe2 = make_pipeline(TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2),
                      LinearSVC(class_weight="balanced")).fit(txt(a2), a2.label)
pr2 = pipe2.predict(txt(b2))
per2 = {r: f1_score(b2.label[(b2.repo == r).values], pr2[(b2.repo == r).values], average="micro") for r in REPOS}
print(f"random     (balanced) cross-repo F1 = {np.mean(list(per2.values())):.4f}   <-- same sizes, same balance")
print(f"  => optimism attributable purely to the random split: {np.mean(list(per2.values())) - np.mean(list(per.values())):+.4f}")

sec("F4 — DOES ADDING THE TIMESTAMP AS A FEATURE 'IMPROVE' THE OFFICIAL SCORE? (leakage smoke test)")
from scipy.sparse import hstack, csr_matrix
v = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2)
Xtr = v.fit_transform(txt(tr)); Xte = v.transform(txt(te))
t_tr = pd.to_datetime(tr.created_at).astype("int64").values.reshape(-1, 1) / 1e18
t_te = pd.to_datetime(te.created_at).astype("int64").values.reshape(-1, 1) / 1e18
for use_t in [False, True]:
    A = hstack([Xtr, csr_matrix(t_tr)]) if use_t else Xtr
    B = hstack([Xte, csr_matrix(t_te)]) if use_t else Xte
    m = LinearSVC(class_weight="balanced").fit(A, tr.label)
    p = m.predict(B)
    sc = np.mean([f1_score(te.label[(te.repo == r).values], p[(te.repo == r).values], average="micro") for r in REPOS])
    print(f"  text {'+ timestamp' if use_t else '           '} -> cross-repo F1 = {sc:.4f}")

sec("F5 — HOW SEPARABLE ARE THE THREE CLASSES REALLY? (agreement between two very different models)")
m1 = make_pipeline(TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2), LinearSVC(class_weight="balanced")).fit(txt(tr), tr.label)
m2 = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=300000), LogisticRegression(max_iter=3000, class_weight="balanced")).fit(txt(tr), tr.label)
p1, p2 = m1.predict(txt(te)), m2.predict(txt(te))
agree = (p1 == p2)
print(f"word-SVM vs char-LogReg agree on {agree.mean():.1%} of test issues")
print(f"  accuracy when they AGREE   : {(p1[agree] == te.label.values[agree]).mean():.1%}  (n={agree.sum()})")
print(f"  accuracy when they DISAGREE: {(p1[~agree] == te.label.values[~agree]).mean():.1%}  (n={(~agree).sum()})")
print("  => the disagreement set is where a better model must win; it is also where labels are most doubtful.")

sec("F6 — HARDEST CONFUSION PAIR, AND WHAT IT LOOKS LIKE")
err = te[(p1 != te.label.values)].copy(); err["pred"] = p1[p1 != te.label.values]
print(err.groupby(["label", "pred"]).size().sort_values(ascending=False).to_string())
print("\nThree bug->question errors (title only):")
for t in err[(err.label == "bug") & (err.pred == "question")].title.head(3): print("   -", t[:110])
print("\nThree question->feature errors (title only):")
for t in err[(err.label == "question") & (err.pred == "feature")].title.head(3): print("   -", t[:110])
