# -*- coding: utf-8 -*-
# ruff: noqa  -- verbatim exploratory script, kept for provenance (see README)
"""Q1-grade statistical machinery, run for real on the current baselines.
Every number printed here goes into the protocol document."""
import warnings, sys, io, json, hashlib, re
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

import os
from pathlib import Path

# Cached competition CSVs, downloaded by `make data` (see ai4se.loader).
D = str(Path(__file__).resolve().parents[2] / "data" / "raw")
tr = pd.read_csv(os.path.join(D, "issues_train.csv")).fillna("")
te = pd.read_csv(os.path.join(D, "issues_test.csv")).fillna("")
REPOS = sorted(tr.repo.unique())
OUT = {}
def sec(t): print("\n" + "=" * 74 + "\n" + t + "\n" + "=" * 74)
def txt(d): return (d.title + " \n " + d.body).str.slice(0, 6000)
def xrepo(y, p, repo):
    return np.mean([f1_score(y[repo == r], p[repo == r], average="micro") for r in REPOS])

ytr, yte = tr.label.values, te.label.values
rtr, rte = tr.repo.values, te.repo.values

# ================================================================ 1
sec("1. TIMELINE OF EACH CLASS PER REPO  (data for the smoking-gun chart)")
al = pd.concat([tr, te], ignore_index=True); al["ts"] = pd.to_datetime(al.created_at)
tl = {}
for r in REPOS:
    s = al[al.repo == r]
    lo, hi = s.ts.min(), s.ts.max(); span = (hi - lo).days or 1
    tl[r] = {"lo": str(lo.date()), "hi": str(hi.date()), "span_days": span, "classes": {}}
    for l in ["bug", "feature", "question"]:
        g = s[s.label == l].ts
        q = [(pd.Timestamp(v) - lo).days / span for v in g.quantile([.1, .25, .5, .75, .9])]
        tl[r]["classes"][l] = {"q": [round(x, 4) for x in q], "median_date": str(g.median().date())}
    print(f"{r:24s} {tl[r]['lo']} → {tl[r]['hi']}  ({span}d)")
    for l in ["bug", "feature", "question"]:
        c = tl[r]["classes"][l]
        print(f"    {l:9s} median {c['median_date']}  norm.quartiles {c['q']}")
OUT["timeline"] = tl

# ================================================================ 2
sec("2. TOKEN-LENGTH DISTRIBUTION BY CLASS  (CDF data)")
al["tok"] = (al.title + " " + al.body).str.split().str.len()
grid = [16, 32, 64, 128, 192, 256, 384, 512, 768, 1024]
cdf = {}
for l in ["bug", "feature", "question"]:
    v = al[al.label == l].tok.values
    cdf[l] = [round(float((v <= g).mean()), 4) for g in grid]
    print(f"{l:9s} " + " ".join(f"{g}:{cdf[l][i]:.2f}" for i, g in enumerate(grid)))
print("median tokens:", al.groupby("label").tok.median().to_dict())
OUT["len_cdf"] = {"grid": grid, "cdf": cdf}

# ================================================================ 3
sec("3. STRUCTURAL-CUE PREVALENCE BY CLASS  (grouped-bar data)")
PAT = {"code block": r"```", "stack trace": r"(?i)(traceback|at [\w$.]+\(|Exception|Error:)",
       "ảnh/đính kèm": r"!\[|\.png|\.gif|\.jpg", "checkbox": r"- \[[ xX]\]",
       "heading": r"(?m)^#{1,6} ", "HTML comment": r"<!--",
       "khối version/OS": r"(?i)(version|OS|platform|browser)\s*[:=]",
       "tiêu đề có ?": None, "tiêu đề how/what/why": None}
prev = {}
for k, p in PAT.items():
    row = {}
    for l in ["bug", "feature", "question"]:
        s = al[al.label == l]
        if k == "tiêu đề có ?": row[l] = float(s.title.str.contains(r"\?", regex=True).mean())
        elif k == "tiêu đề how/what/why": row[l] = float(s.title.str.contains(r"(?i)^(how|what|why|is|can|does|any)\b", regex=True).mean())
        else: row[l] = float(s.body.str.contains(p, regex=True).mean())
    prev[k] = {a: round(b, 4) for a, b in row.items()}
    print(f"{k:22s} bug {row['bug']:.1%}  feature {row['feature']:.1%}  question {row['question']:.1%}")
OUT["struct_prev"] = prev

# ================================================================ 4
sec("4. BASELINE MODELS  (these are the objects every test below operates on)")
def fit_global(clf, field=txt):
    p = make_pipeline(TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2), clf)
    p.fit(field(tr), ytr); return p.predict(field(te)), p
def fit_perrepo(clf_fn, field=txt):
    pred = np.empty(len(te), dtype=object)
    for r in REPOS:
        a = tr[tr.repo == r]; m = (rte == r)
        p = make_pipeline(TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2), clf_fn())
        p.fit(field(a), a.label); pred[m] = p.predict(field(te[m]))
    return pred

P = {}
P["global_svm"], _ = fit_global(LinearSVC(class_weight="balanced"))
P["perrepo_svm"] = fit_perrepo(lambda: LinearSVC(class_weight="balanced"))
P["title_only"], _ = fit_global(LinearSVC(class_weight="balanced"), lambda d: d.title)
P["body_only"], _ = fit_global(LinearSVC(class_weight="balanced"), lambda d: d.body.str.slice(0, 6000))
P["char_lr"] = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=300000),
                             LogisticRegression(max_iter=3000, class_weight="balanced")).fit(txt(tr), ytr).predict(txt(te))
for k, v in P.items(): print(f"  {k:14s} cross-repo F1 = {xrepo(yte, v, rte):.4f}")

# ================================================================ 5
sec("5. PER-REPO × PER-CLASS F1 OF THE GLOBAL MODEL  (heatmap data)")
hm = {}
for r in REPOS:
    m = (rte == r); row = {}
    for l in ["bug", "feature", "question"]:
        row[l] = round(float(f1_score(yte[m] == l, P["global_svm"][m] == l)), 4)
    hm[r] = row
    print(f"  {r:24s} " + "  ".join(f"{l}:{row[l]:.3f}" for l in row))
OUT["heatmap"] = hm

# ================================================================ 6
sec("6. BOOTSTRAP CONFIDENCE INTERVAL ON CROSS-REPO F1 (stratified by repo, B=10000)")
rng = np.random.default_rng(0)
B = 10000
idx_by_repo = {r: np.where(rte == r)[0] for r in REPOS}
def boot_ci(pred, B=B):
    vals = np.empty(B)
    for b in range(B):
        fs = []
        for r in REPOS:
            ii = rng.choice(idx_by_repo[r], size=len(idx_by_repo[r]), replace=True)
            fs.append(f1_score(yte[ii], pred[ii], average="micro"))
        vals[b] = np.mean(fs)
    return np.percentile(vals, [2.5, 97.5]), vals.std()
for k in ["global_svm", "perrepo_svm"]:
    (lo, hi), sd = boot_ci(P[k], 2000)
    print(f"  {k:14s} F1 = {xrepo(yte, P[k], rte):.4f}   95% CI [{lo:.4f}, {hi:.4f}]   width {hi-lo:.4f}   se {sd:.4f}")
    OUT.setdefault("ci", {})[k] = [round(lo, 4), round(hi, 4)]

# ================================================================ 7
sec("7. McNEMAR — PAIRED COMPARISON DONE PROPERLY (exact binomial, + Holm over 5 repos)")
def mcnemar_exact(a, b, y):
    """a,b = predictions; returns b01,b10,p (exact two-sided binomial)."""
    ca, cb = (a == y), (b == y)
    n01 = int(np.sum(~ca & cb)); n10 = int(np.sum(ca & ~cb))
    n = n01 + n10
    p = 1.0 if n == 0 else float(stats.binomtest(min(n01, n10), n, 0.5).pvalue)
    return n01, n10, n, p
def holm(ps):
    o = np.argsort(ps); out = np.empty(len(ps)); run = 0.0
    for rank, i in enumerate(o):
        v = (len(ps) - rank) * ps[i]; run = max(run, min(v, 1.0)); out[i] = run
    return out

COMPARISONS = [("global_svm", "perrepo_svm"), ("global_svm", "title_only"),
               ("global_svm", "body_only"), ("global_svm", "char_lr")]
for A, Bm in COMPARISONS:
    print(f"\n  {Bm}  vs  {A}   (positive n01 favours {Bm})")
    ps, rows = [], []
    for r in REPOS:
        m = (rte == r)
        n01, n10, n, p = mcnemar_exact(P[A][m], P[Bm][m], yte[m])
        rows.append((r, n01, n10, n, p)); ps.append(p)
    adj = holm(np.array(ps))
    for (r, n01, n10, n, p), pa in zip(rows, adj):
        star = "***" if pa < .001 else "**" if pa < .01 else "*" if pa < .05 else "ns"
        print(f"    {r:24s} n01={n01:3d} n10={n10:3d} discordant={n:3d}  p={p:.2e}  p_holm={pa:.2e} {star}")
    n01t, n10t, nt, pt = mcnemar_exact(P[A], P[Bm], yte)
    print(f"    {'POOLED':24s} n01={n01t:3d} n10={n10t:3d} discordant={nt:3d}  p={pt:.3e}")

# ================================================================ 8
sec("8. STATISTICAL POWER — what difference can this test set even detect?")
print("  McNemar, alpha=0.05 two-sided, 80% power, simulated 4000x per point.")
print("  Discordant rate observed between two reasonable models:")
_, _, nd_obs, _ = mcnemar_exact(P["global_svm"], P["char_lr"], yte)
print(f"    {nd_obs} / {len(yte)} = {nd_obs/len(yte):.1%} of test items\n")
def power_for(delta_acc, nd_rate, n=1500, sims=4000, alpha=.05):
    nd = int(round(nd_rate * n))
    if nd == 0: return 0.0
    shift = delta_acc * n
    p_fav = 0.5 + shift / (2 * nd)
    if not (0 < p_fav < 1): return 1.0
    b = rng.binomial(nd, p_fav, sims)
    sig = np.array([stats.binomtest(min(x, nd - x), nd, 0.5).pvalue < alpha for x in b])
    return float(sig.mean())
rate = nd_obs / len(yte)
print("  Δ F1 (điểm)   power")
mdd = None
for d in [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050]:
    pw = power_for(d, rate, sims=1500)
    if mdd is None and pw >= .8: mdd = d
    print(f"    {d*100:4.1f}        {pw:.3f}{'   <-- 80% power' if (mdd == d) else ''}")
OUT["power"] = {"discordant_rate": round(rate, 4), "mdd80": mdd}
print(f"\n  => MINIMUM DETECTABLE DIFFERENCE at 80% power ≈ {mdd*100:.1f} F1 points on the full 1500-item test set.")
print("     Per repo (n=300) it is roughly sqrt(5)x larger, i.e. ~%.1f points." % (mdd*100*np.sqrt(5)))

# ================================================================ 9
sec("9. SEED VARIANCE — how much does a single number move on its own?")
seeds = range(12)
vals = []
for s in seeds:
    m = make_pipeline(TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2),
                      SGDClassifier(loss="modified_huber", alpha=1e-5, max_iter=30, random_state=s, class_weight="balanced"))
    m.fit(txt(tr), ytr); vals.append(xrepo(yte, m.predict(txt(te)), rte))
vals = np.array(vals)
print(f"  12 seeds, identical config: min {vals.min():.4f}  max {vals.max():.4f}  mean {vals.mean():.4f}  sd {vals.std(ddof=1):.4f}")
print(f"  spread max-min = {(vals.max()-vals.min())*100:.2f} F1 points from SEED ALONE.")
print(f"  cherry-picking the best of 12 seeds inflates the reported score by {(vals.max()-vals.mean())*100:.2f} points.")
OUT["seed"] = {"sd": round(float(vals.std(ddof=1)), 4), "range": round(float(vals.max()-vals.min()), 4)}

# ================================================================ 10
sec("10. NEGATIVE CONTROL — does the structural signal survive label shuffling?")
FEATS = {"has_code": r"```", "has_trace": r"(?i)(traceback|at [\w$.]+\(|Exception|Error:)",
         "has_img": r"!\[|\.png|\.gif|\.jpg", "has_cb": r"- \[[ xX]\]", "has_head": r"(?m)^#{1,6} ",
         "has_html": r"<!--", "has_ver": r"(?i)(version|OS|platform|browser)\s*[:=]",
         "has_url": r"https?://", "has_ref": r"#\d{2,6}", "has_at": r"(?m)(^|\s)@\w+"}
def sfeat(d):
    X = pd.DataFrame(index=d.index)
    for k, p in FEATS.items(): X[k] = d.body.str.contains(p, regex=True).astype(int)
    X["q"] = d.title.str.contains(r"\?").astype(int)
    X["wh"] = d.title.str.contains(r"(?i)^(how|what|why|is|can|does|any)\b").astype(int)
    X["loglen"] = np.log1p(d.body.str.len()); X["lines"] = np.log1p(d.body.str.count("\n"))
    return X.values
Str, Ste = sfeat(tr), sfeat(te)
real = xrepo(yte, LogisticRegression(max_iter=2000, class_weight="balanced").fit(Str, ytr).predict(Ste), rte)
perm = []
for s in range(20):
    ysh = np.random.default_rng(s).permutation(ytr)
    perm.append(xrepo(yte, LogisticRegression(max_iter=2000, class_weight="balanced").fit(Str, ysh).predict(Ste), rte))
perm = np.array(perm)
z = (real - perm.mean()) / perm.std(ddof=1)
print(f"  structural features, real labels     : {real:.4f}")
print(f"  structural features, SHUFFLED labels : {perm.mean():.4f} ± {perm.std(ddof=1):.4f}  (20 permutations)")
print(f"  permutation p < {1/21:.3f} (real beats all 20)   z = {z:.1f}")
print("  => the 0.558 is signal, not capacity. This is the template for every 'is it real?' check.")
OUT["negctrl"] = {"real": round(real, 4), "perm_mean": round(float(perm.mean()), 4), "perm_sd": round(float(perm.std(ddof=1)), 4)}

# ================================================================ 11
sec("11. ACCURACY vs COVERAGE — value of an abstention option")
cal = make_pipeline(TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2),
                    CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=5, method="sigmoid"))
cal.fit(txt(tr), ytr)
pr = cal.predict_proba(txt(te)); cls = cal.classes_
conf = pr.max(1); pred = cls[pr.argmax(1)]
order = np.argsort(-conf)
cov_rows = []
for cov in [1.0, .95, .9, .8, .7, .6, .5]:
    k = int(len(order) * cov); sel = order[:k]
    acc = float((pred[sel] == yte[sel]).mean())
    cov_rows.append((cov, acc, float(conf[sel].min())))
    print(f"  coverage {cov*100:5.1f}%   accuracy {acc:.4f}   ngưỡng tin cậy ≥ {conf[sel].min():.3f}")
OUT["coverage"] = [[round(a, 3), round(b, 4)] for a, b, _ in cov_rows]

# ================================================================ 12
sec("12. LEAKAGE AUDIT — exact, near-duplicate, and vocabulary")
def norm(s): return re.sub(r"\W+", " ", str(s).lower()).strip()
ktr = txt(tr).map(norm).map(lambda x: hashlib.md5(x.encode()).hexdigest())
kte = txt(te).map(norm).map(lambda x: hashlib.md5(x.encode()).hexdigest())
print(f"  exact duplicate train↔test : {len(set(ktr) & set(kte))}")
print(f"  identical title train↔test : {te.title.map(norm).isin(set(tr.title.map(norm))).sum()}")
v = TfidfVectorizer(sublinear_tf=True, min_df=2).fit(txt(pd.concat([tr, te])))
A, Bx = v.transform(txt(tr)), v.transform(txt(te))
mx = np.zeros(len(te))
for i in range(0, len(te), 300):
    mx[i:i+300] = cosine_similarity(Bx[i:i+300], A).max(1)
for th in [.99, .95, .90, .80]:
    print(f"  near-duplicate cosine ≥ {th:.2f}   : {(mx >= th).sum():4d} test items ({(mx>=th).mean():.1%})")
OUT["leak"] = {"exact": int(len(set(ktr) & set(kte))), "near90": int((mx >= .9).sum()), "near80": int((mx >= .8).sum())}

# ================================================================ 13
sec("13. VOCABULARY DRIFT ACROSS ERAS — the mechanism behind the temporal leak")
al2 = al.sort_values("ts"); half = len(al2)//2
old, new = al2.iloc[:half], al2.iloc[half:]
cv = CountVectorizer(min_df=5, stop_words="english", token_pattern=r"[a-z][a-z\-]{2,}")
cv.fit((al2.title + " " + al2.body).str.lower())
Vo = np.asarray(cv.transform((old.title+" "+old.body).str.lower()).sum(0)).ravel()
Vn = np.asarray(cv.transform((new.title+" "+new.body).str.lower()).sum(0)).ravel()
po, pn = Vo/Vo.sum(), Vn/Vn.sum()
jsd = .5*stats.entropy(po, (po+pn)/2) + .5*stats.entropy(pn, (po+pn)/2)
print(f"  Jensen–Shannon divergence giữa từ vựng nửa cũ và nửa mới: {jsd:.4f}")
top_new = np.argsort((pn+1e-9)/(po+1e-9))[::-1][:12]
top_old = np.argsort((po+1e-9)/(pn+1e-9))[::-1][:12]
vocab = np.array(cv.get_feature_names_out())
print("  chỉ có ở giai đoạn mới :", ", ".join(vocab[top_new][:10]))
print("  chỉ có ở giai đoạn cũ  :", ", ".join(vocab[top_old][:10]))

print("\n\n" + "="*74 + "\nJSON for the document\n" + "="*74)
print(json.dumps(OUT, ensure_ascii=False)[:4000])
