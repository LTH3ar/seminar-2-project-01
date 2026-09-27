# Reflection — what went wrong, and what I would do differently

**Author:** Minh Khanh · **Branch:** `minh-khanh`

Written after the run in `docs/setfit-reproduction-report.md`. The point of this
file is the mistakes; the results document already has the results.

---

## 1. Errors I made in my own code, and how they surfaced

None of these were caught by reading the code. All of them were caught by a test
or by a number that looked wrong.

### 1.1 I scored the wrong metric

My first `cross_repo_f1` defaulted to micro-F1. For a single-label multi-class
problem micro-F1 *is* accuracy, and the competition does not rank on accuracy.
Every number I produced before catching this was incomparable with the
leaderboard while looking exactly like it belonged there.

`preregistration.md` carries the same error, amended before any model ran. I
made it twice, in two places, which suggests I had not actually understood the
metric the first time — I had copied a plausible-looking default.

**Fix:** the metric is now checked against `sklearn.classification_report` to
zero difference, as a test that runs before anything else.

**What I would change:** write the metric test first, against the library, before
writing the metric. It takes ten minutes and it is the one number the entire
project is ranked on.

### 1.2 I scored a fixed label set

`_confusion` scored a hardcoded three-label set. Any fold or bootstrap resample
missing a class awarded it F1 = 0 instead of excluding it — so every
cross-validation score and every bootstrap interval was quietly biased
downward. On balanced data with 1,500 items this almost never triggers, which is
exactly why it survived.

**What I would change:** test the degenerate cases deliberately. A test that only
exercises well-formed input tests the happy path of the data, not the code.

### 1.3 A loop over a constant silently dropped data

`train_per_repo` iterated a hardcoded `REPOSITORIES` list rather than the
projects present in the data. Pass it a subset with different names and it
trains on nothing, reports success, and returns a model. Caught by a test, not
by reading.

### 1.4 Duplicate text straddled the validation split

The training split contains one duplicated `(title, body)` pair. A row-wise
stratified split puts one copy on each side — a small leak into the validation
set I was using to *select* configurations. The split now works over distinct
texts. This is the precise error the project exists to avoid, and I wrote it
anyway.

### 1.5 The select stage printed one thing and did another

`stage_select` printed "the test split is NOT read" while calling
`load_dataset`, which loads both splits. The printed claim was false. It did not
leak anything into the selection — nothing downstream touched the test frame —
but a reassurance that is not enforced is worse than no reassurance, because it
stops you looking.

**Fix:** `load_split("train", ...)`, plus the access log that made the claim
checkable rather than printable.

### 1.6 I shipped a sentence that contradicted the line above it

`describe_claim` hardcoded the preregistered 3.0-point threshold into its output
and called it *"the resolution of this test set"* — directly under a line
reporting a resolution of 1.0 points computed from the run's own discordant
rate. Two different quantities, one name, four lines apart.

The verdict was unaffected. It is still the worst of these, because I had spent
the same week writing up exactly this class of error in someone else's report,
and I did not see it in mine until the final run printed both numbers together.

---

## 2. Predictions of mine that did not survive

### 2.1 "The gap is because the team's run was memory-starved"

The group ran MPNet at `batch_size=8, max_seq_length=128` because it would not
fit in 12 GB. The official configuration is batch 16 at 384 tokens. I predicted
running it properly would recover most of the 1.7-point gap.

Measured on a 47 GB card: **0.8138 against their 0.8108 — three tenths of a
point.** The memory explanation accounts for almost none of the gap.

I liked this hypothesis because it was tidy and because it would have made the
group's number look better. Neither is evidence.

### 2.2 "Newer encoders should help"

`bge-base-en-v1.5` and `e5-base-v2` are two years newer than
`all-mpnet-base-v2` and trained on far more text. Both lost, by 2.20 and 1.67
points.

### 2.3 "Stripping templates is the obvious fix"

I built `strip_templates` because the group's error analysis made a good case
for it — the OpenCV forum-referral line at 15.6× lift is a real defect. It was
the **worst** of nine candidates, and the only one whose harm cleared the
decision threshold.

### 2.4 "Seed spread is 2.4 points"

Carried over from `docs/benchmark-audit.md`, where it was measured on the
classical baselines. For SetFit at the official configuration it is **1.48
points**. I had been quoting 2.4 as though it were a property of the benchmark
when it is a property of the model.

---

## 3. What actually worked

**Fixing the threshold before running anything.** `DECISION_THRESHOLD = 0.030`
was set by a power analysis before the first experiment. When the results came
back with the best candidate 0.40 points behind the control, there was no
argument to have with myself. Had I set the threshold afterwards, I would have
found a reason for 0.3.

**Running five seeds.** This changed the headline. Seed 42 alone says the
reproduction is 1.02 points short; five seeds say 0.16. Both numbers are honest
and one of them is badly misleading, and I would not have known which without
paying for four extra runs.

**Logging test-split access instead of promising it.** Four lines with
timestamps, none during selection. It converts "we did not tune on test" from a
claim into a file.

**Gating the ambiguous configuration behind a flag.** `global_pairs` — one
encoder on all 1,500 issues, five heads — is legal by the letter of the rules
and is the only untested lever with real upside. It is also the one I most
wanted to run, which is why it needed to be behind a flag and not behind my own
judgement at 2 a.m.

**Committing the per-issue predictions.** Every analysis in the report was
recomputed from JSON after the GPU was released.

---

## 4. What I would do differently

1. **Write the metric test before the metric.** §1.1.
2. **Run the seeds first, not last.** I spent two hours on selection before
   knowing the noise floor. The noise floor determines whether selection is
   worth running at all.
3. **Cross-validate the selection.** Nine configurations ranked on one 375-item
   holdout, with intervals 8 points wide. The ranking below the control is not
   reliable; only "nothing beat the control" survives.
4. **Test the negative controls I preregistered.** Commitment 5 requires a
   shuffled-label control with every claimed improvement. There were no claimed
   improvements, so nothing was owed — but I also never checked the machinery
   works, and I would rather have found that out on a run that did not matter.
5. **Ask about `global_pairs` on day one.** The question has been open the whole
   time and the answer gates the only remaining idea. It costs an email.

---

## 5. On the group report

I read the group report closely and found errors, the most serious being a claim
in the conclusion that the MPNet reproduction beats the baseline on
`tensorflow/tensorflow`. The group's own generated table says 0.8642 against
0.8644.

Two things I want on the record about how I handled that.

**I checked before claiming.** The first extraction of the PDF appeared to show
several tables with columns shifted by one row, which would have been a much
worse problem. Rendering the pages as images showed the tables are correct and
the shift was an artifact of `pdftotext`. Three of my five initial findings
evaporated at that step. Had I reported them from the text dump I would have
accused my own team of errors that do not exist.

**My run makes the same point better than the criticism does.** The group's
claim is wrong by 0.02 points. My five-seed mean for `tensorflow` is 0.8650
against a published 0.8644 — nominally *above*, by 0.06 points, with a standard
deviation of 0.0035 and a spread of 0.94 points. Neither number establishes
anything. The right correction is not "you were below, not above"; it is that
**this project reproduces on `tensorflow` to within noise, and no one should
claim a direction at all.** That is a better sentence for the group than the one
currently in the conclusion, and it is also better than the one I first drafted.

The same applies to the group's statement that the remaining 1.02-point gap is
*real*. My run puts a number on why it is not: at a realistic discordant rate
between two different model families, this test set resolves about 3.5 points.
The group report already says, in §5.1.3, that differences inside one standard
deviation should not be over-interpreted. The correction it needs is to apply
its own sentence.

---

## 6. The thing I keep relearning

Every error in §1 was invisible to me while reading my own code and obvious the
moment something external checked it — a test, a library comparison, a rendered
page, a log file. I do not catch my own mistakes by being careful. I catch them
by arranging for something else to catch them, and the arranging has to happen
before I have an answer I want to be true.
