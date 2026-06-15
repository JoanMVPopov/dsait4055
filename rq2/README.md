# RQ2 statistics, explained from scratch

This document explains **every statistical method, test, and modelling choice**
used in the RQ2 analysis, why it was chosen, and what its assumptions are. It is
written to be **self-contained**: a reader who has never seen our code or
discussions should be able to follow it, and you should be able to explain any
piece of it from this document alone. Every concept is illustrated with a small
worked example using tiny made-up numbers, followed by the real result from our
study.

It also answers, head-on, a question that always comes up: **"did you test for
normality, or did you just assume non-parametric tests?"** (Short answer: we did
not run normality tests, on purpose; section 3.5 explains why that is the correct
choice here, with the actual numbers.)

---

## 0. The question and the data

**RQ2: Are LLM-related Stack Overflow questions less *answerable* than mature
programming questions?** (Secondary: are they received differently? Tertiary:
is any gap closing over time?)

The **unit of analysis is one question**. Each question belongs to exactly one of
two groups:

- **LLM** (`IsLLM = 1`): matched the LLM tag/keyword detection (≈20,000 questions).
- **Mature** (`IsLLM = 0`): a large sample of questions with the top-100
  pre-2022 programming tags (≈1.5 million questions).

**The two groups are made mutually exclusive.** A question can match *both*
populations (e.g. tagged both `python` and `langchain`). About 13,600 such
overlaps existed; we treat LLM membership as definitive and **remove those
questions from the mature side**, so no question appears in both groups (no
contradictory labels, no double-counting). Each group is also de-duplicated by
question id first. So "mature" precisely means "top-100-tag questions that are
*not* in the LLM set."

For each question we have **posting-time facts** (body length, whether it has
code, asker reputation, tag count, the month it was posted) and **outcomes**
(did it get answered, how fast, did it get an accepted answer, its score,
comments, whether it was closed).

The whole analysis is a **two-group comparison**: take a number or a rate,
compute it for LLM and for mature, and ask whether the difference is (a) real
(not just noise) and (b) big enough to matter — and later, whether it survives
**adjusting** for the posting-time facts.

---

## 1. The censoring problem (why outcomes are "windowed")

Before any test, we had to define the outcomes carefully, because of a trap
called **right-censoring**.

**The trap.** "Did the question get an answer?" depends on *how long we have been
watching it*. A question posted 3 years ago has had years to attract an answer; a
question posted 5 days before we downloaded the data has had only 5 days. LLM
questions are concentrated in recent months, so they have been watched for less
time on average. If we naively compared "fraction ever answered," LLM questions
would look worse **purely because they are younger** — biasing the result in the
exact direction of our hypothesis. That would be cheating.

**Worked example.** Two questions, both eventually get answered on day 20:

```
Question A: posted 100 days before snapshot.  On day 20 it is answered.  -> "answered" = yes
Question B: posted   8 days before snapshot.  We stop watching on day 8. -> "answered" = NO (we left before day 20)
```

B is not less answerable than A — we just stopped watching too early.

**The fix: fixed observation windows + eligibility.**

- `AnsweredWithin30Days` = 1 if the first answer arrived within 30 days of
  posting, else 0. **But** a question we have watched for fewer than 30 days is
  *not eligible* — its outcome is set to "missing" and it is **excluded**, not
  counted as a failure.
- `AcceptedWithin90Days` = same idea with a 90-day window.

So every question that *counts* has been observed for the full window, making LLM
and mature comparable. (Cost: the most recent ~month of data is dropped because
it is too young — an honest trade, documented as a limitation.)

Two small implementation details, for completeness:

- **Whole-day counting.** Window membership is computed in whole days, so "within
  30 days" tolerates up to roughly an extra day at the boundary. This is identical
  for both groups, so it cannot bias the comparison; it just means the windows are
  "≈30/≈90 days," not razor-sharp.
- **Per-dataset snapshot.** The LLM and mature exports were pulled a couple of days
  apart, so each group's window is measured against *its own* extraction date
  (inferred as the latest activity in that group, a safe lower bound). A 2-day
  difference is immaterial; the alternative — using one date for data pulled on
  different days — would be the less accurate choice.

This single decision is why the binary outcomes are trustworthy. Everything in
section 3 onward operates on these windowed outcomes.

---

## 2. Two kinds of outcome → two families of method

- **Binary outcomes** (yes/no): answered-within-30d, accepted-within-90d,
  is-closed. → proportions, chi-square, logistic regression.
- **Continuous/count outcomes** (a number): time-to-first-answer, score,
  comment count. → medians, Mann-Whitney U, survival analysis.

The method must match the kind of outcome. Using a mean-based test on a heavily
skewed count (section 3.5) is the classic mistake we deliberately avoid.

---

# PART I — Comparing the two groups (descriptive statistics)

## 3.1 Proportions and the risk difference

For a binary outcome, the basic number is a **proportion** (a rate): answered
count ÷ eligible count.

The simplest effect size is the **risk difference (RD)** = rate(LLM) −
rate(mature), in percentage points.

**Real result:** Answered-within-30d is 64.5% for LLM vs 73.1% for mature, so
**RD = −8.6 percentage points**. Accepted-within-90d: 26.5% vs 37.9%, RD = −11.4pp.

RD is the most *directly interpretable* effect size: "8.6 out of every 100 LLM
questions that would have been answered if they behaved like mature questions
are not." We lead with it.

## 3.2 Confidence intervals (why Wilson, not the textbook formula)

A rate from a sample is an estimate; a **95% confidence interval (CI)** expresses
its uncertainty. The textbook "normal approximation" CI is
`p ± 1.96 · sqrt(p(1−p)/n)`. It misbehaves when `p` is near 0 or 1, or `n` is
small — it can even give impossible values.

**Worked example.** 2 answered out of 5 (`p = 0.4`, `n = 5`):

```
normal approx:  0.4 ± 1.96 · sqrt(0.4·0.6/5) = 0.4 ± 0.43 = [-0.03, 0.83]
```

A negative probability is nonsense. The **Wilson interval** is a smarter formula
that stays inside [0, 1] and is accurate even for small `n` or extreme `p`; here
it gives roughly `[0.12, 0.78]`. At our sample sizes (tens of thousands to
millions) any method gives almost the same answer, but Wilson is the safe default,
so we use it (`statsmodels.proportion_confint(method="wilson")`).

## 3.3 The chi-square test of independence

**Question it answers:** "Is the outcome rate *different* between the two groups,
beyond what random chance would produce?"

It builds a 2×2 **contingency table** (group × outcome) and compares the
**observed** counts to the counts we would **expect if group and outcome were
unrelated** (independent). Big gaps between observed and expected → evidence of a
real association.

**Worked example.** 100 mature, 100 LLM:

```
            answered   not answered   total
mature         73           27         100
LLM            64           36         100
total         137           63         200
```

If group and answering were unrelated, each cell's expected count is
`row_total · col_total / grand_total`. Expected answered for each group =
`100 · 137 / 200 = 68.5`; expected not-answered = `31.5`.

```
chi-square = Σ (observed − expected)² / expected
           = (73−68.5)²/68.5 + (27−31.5)²/31.5 + (64−68.5)²/68.5 + (36−31.5)²/31.5
           = 0.296 + 0.643 + 0.296 + 0.643
           = 1.88     (degrees of freedom = 1)  ->  p ≈ 0.17
```

So at `n = 200`, a 9-point gap is **not** statistically significant. The same gap
at `n = 1.5 million` produces an astronomically significant result. Hold that
thought — it is the whole point of section 3.7.

**Real result:** Answered-within-30d, chi-square = 742, p ≈ 10⁻¹⁶³.

## 3.4 Cramér's V (the effect size for chi-square)

The chi-square *statistic* grows with sample size, so it cannot tell you whether
an association is **big or small** — only whether it is detectable. **Cramér's V**
rescales it to a 0–1 number (0 = no association, 1 = perfect), independent of `n`.
For a 2×2 table it equals the **phi coefficient**.

```
V = sqrt( chi-square / (n · (min(rows, cols) − 1)) )
```

For the toy table: `V = sqrt(1.88 / (200·1)) = 0.097` — a small association.

**An important subtlety we hit in the real data.** Phi/Cramér's V is *attenuated
when the two groups are very different in size*. Our groups are 20k vs 1.5M, so
even though the risk difference is a meaningful −8.6pp, Cramér's V is only **0.022**.
This is not a contradiction — it is why we **report the risk difference as the
primary effect size** and treat Cramér's V as a secondary, scale-free companion.
(We compute V with `scipy.stats.contingency.association`, a trusted
implementation, rather than by hand.)

## 3.5 Continuous outcomes, and the normality question (read this one carefully)

For the *continuous* outcomes (time-to-first-answer, score, comment count) we use
the **Mann-Whitney U test**, a **non-parametric** test, and report **medians**
rather than means. This section justifies that, because "did you check
normality?" is the obvious challenge.

### 3.5.1 We did not run a normality test. Here is why that is correct.

There exist formal normality tests (Shapiro-Wilk, Kolmogorov-Smirnov,
Anderson-Darling). **We deliberately did not use them**, for two reasons:

1. **At large n they are useless.** These tests have so much power at `n` in the
   thousands — let alone millions — that they detect *trivially small*,
   practically irrelevant deviations and reject normality essentially always. A
   "p < 0.001, data is non-normal" result at `n = 1.5M` tells you nothing you
   didn't already know. Running one would be theatre.

2. **We looked at the distributions instead — which is the right thing to do.**
   Here are the actual numbers from our data:

```
                         mean      median   skewness   notes
time-to-answer (hours)   274.5      1.46       9.5      half the answers arrive within ~1.5h,
                                                        but a long tail stretches to 37,680h
score                     0.57      0.00      77.9      53.6% of questions have score exactly 0
comment count             2.25      2.00       2.5      35.5% have zero comments
```

A normal distribution is symmetric (skewness ≈ 0), has no pile-up at a boundary,
and its mean ≈ its median. Every one of these variables violates that grossly:
they are **right-skewed, heavy-tailed, and bounded at zero**, often with a large
spike *at* zero. They are non-normal *by their very nature* (a duration cannot be
negative; a count is a non-negative integer). No test is needed to see this; the
mean-vs-median gap alone (274 vs 1.5 for time-to-answer) screams skew.

### 3.5.2 Where normality even matters (and where it doesn't)

Normality is an assumption of only *some* methods:

- **t-test** (compares two means): assumes the *sampling distribution of the
  mean* is normal.
- **linear regression**: assumes normal residuals.

It is **irrelevant** for the methods we actually use on these variables:

- **Chi-square / proportions** (binary outcomes): no normality assumption.
- **Logistic regression** (section 4): a binomial GLM — assumes **nothing** about
  normal residuals.
- **Mann-Whitney U**: distribution-free by construction.

So normality could only ever have mattered if we had chosen a **t-test** for the
continuous outcomes. We didn't — and here is the nuance most people miss:

### 3.5.3 The t-test would actually be "valid" — we still avoid it on purpose

Because of the **Central Limit Theorem**, with `n` in the hundreds of thousands
the sampling distribution of the mean is approximately normal *even though the raw
data is wildly skewed*. So a t-test would give technically valid p-values here.
Our reason for not using it is **not** "the assumption fails" — it is:

- **The mean is a misleading summary of skewed data.** The mean
  time-to-answer is 274 hours (≈11 days), but that number describes almost no real
  question — it is dragged up by a handful of questions answered months later. The
  *typical* question is answered in 1.5 hours (the median). A method built around
  the mean would let a few viral outliers (a question scored 1048, a question
  answered after 37,680 hours) dictate the conclusion.
- **We care about the typical question**, i.e. the middle of the distribution and
  whether one group's whole distribution sits higher than the other's. That is
  exactly what Mann-Whitney measures, and it is **robust to outliers** because it
  uses ranks, not raw values.

So the choice is deliberate and motivated by *what we want to claim* and
*outlier-robustness*, not by a normality test that we skipped on purpose. That is
the honest, defensible position — and it is the sentence to put in the paper.

### 3.5.4 How the Mann-Whitney U test works (worked example)

It throws away the raw magnitudes and keeps only the **ranks** (the ordering),
then asks: do one group's values tend to sit higher in the combined ordering?

**Example.** LLM times = [3, 10, 50] hours; mature times = [1, 2, 4].

```
Pool and sort, assign ranks:
  value:  1    2    3    4    10   50
  group:  m    m    L    m    L    L
  rank:   1    2    3    4    5    6

rank sum, LLM    = 3 + 5 + 6 = 14
rank sum, mature = 1 + 2 + 4 = 7

U_LLM = rank_sum_LLM − n_LLM·(n_LLM+1)/2 = 14 − 3·4/2 = 8
U_mature = 7 − 6 = 1            (check: U_LLM + U_mature = 9 = 3·3 ✓)
```

`U_LLM = 8` out of a maximum of 9 means LLM values almost always outrank mature
ones. Notice the value 50 contributed rank 6 — the same as if it had been 11 — so
**one extreme outlier cannot dominate**. That is the robustness we want.

**Real result:** time-to-first-answer (among answered) — median 9.5h for LLM vs
1.4h for mature; the test is significant with a tiny p. (Aside: a small number of
rows have a *negative* recorded time-to-answer — a data quirk where the stored
first-answer timestamp predates the question; rank-based tests are barely affected
and the survival analysis drops them. Worth a footnote / a look in the data.)

### 3.5.5 The effect sizes: CLES and rank-biserial

A p-value says "there is a difference"; these say "how big."

- **CLES (Common-Language Effect Size)** = the probability that a randomly chosen
  LLM question has a larger value than a randomly chosen mature one. It equals
  `U_LLM / (n_LLM · n_mature)`. In the toy example: `8/9 = 0.889` → an LLM
  question is slower 88.9% of the time. This is the most *human-readable* effect
  size.
- **Rank-biserial correlation (RBC)** = `2·CLES − 1`, rescaled to [−1, +1].
  Positive = LLM tends to be larger. Toy example: `2·0.889 − 1 = +0.78`.

**Sign convention (we verified this):** we always pass the LLM group first, so
**RBC > 0 / CLES > 0.5 means LLM has larger values** (e.g. waits longer).

**Real results:**

```
time-to-answer:  RBC = +0.30,  CLES = 0.648  -> a random LLM question waits longer
                                                than a random mature one 64.8% of the time
comment count:   RBC = −0.11,  CLES = 0.443  -> LLM questions get slightly FEWER comments
score:           RBC = +0.04,  CLES = 0.521  -> essentially no difference
```

> **Note on implementation.** The "obvious" library here, `pingouin`, computes
> CLES with an operation that needs ~128 GB of memory at our sample sizes, so it
> is unusable. We instead take the **U statistic from `scipy`** (computed
> efficiently with ranks) and apply the textbook identities `CLES = U/(n₁n₂)` and
> `RBC = 2·CLES − 1` (Kerby 2014). We checked these match `pingouin` exactly on a
> small example, so they are validated, not hand-waved.

## 3.6 Multiple testing: the Holm-Bonferroni correction

**The problem.** If you run many tests, each at a 5% false-positive rate, the
chance of *at least one* false positive balloons. Run 20 independent tests on pure
noise and you expect ~1 "significant" result by luck.

**Holm-Bonferroni** controls this. Sort the `m` p-values from smallest to largest;
compare the k-th smallest to `0.05 / (m − k + 1)`; stop at the first failure.

**Worked example.** Three tests, p = 0.01, 0.04, 0.20, `m = 3`:

```
smallest 0.01  vs  0.05/3 = 0.0167  ->  0.01 < 0.0167   -> reject (significant)
next     0.04  vs  0.05/2 = 0.025   ->  0.04 > 0.025    -> stop; 0.04 and 0.20 not significant
```

Equivalently, "adjusted" p-values are `0.03, 0.08, 0.20`. Plain **Bonferroni**
would multiply *all* by 3 (`0.03, 0.12, 0.60`) — Holm is uniformly less
conservative (more power) while controlling the same error rate. We apply it
across each family of related tests.

## 3.7 Why we lead with effect sizes, not p-values

At our sample sizes, **the p-value is almost meaningless** as a measure of
importance. Recall the toy chi-square: a 9-point gap gave p ≈ 0.17 at n = 200 but
would give p ≈ 10⁻¹⁰⁰ at n = 1.5M. The *gap is the same*; only the certainty
changed. With over a million questions, even a 0.2-point difference that nobody
cares about will have p < 0.0001.

**Therefore: a tiny p-value here means "we are sure the difference is not exactly
zero" — not "the difference is large."** Every claim in RQ2 is led by an **effect
size** (risk difference, Cramér's V, CLES, rank-biserial, odds ratio, hazard
ratio, the adjusted probability gap), with the p-value treated as a formality.

---

# PART II — The adjusted comparison (logistic regression)

The descriptive comparison tells us LLM questions are answered less. But **maybe
that is because they differ in other ways** — longer, more code-heavy, posted by
newer accounts. Regression asks: **does the gap survive after we hold those
things equal?**

## 4.1 Confounding (why we adjust at all)

**Worked example.** Suppose long questions are harder to answer, and LLM questions
happen to be longer. Then *some* of the raw LLM penalty is really a "length"
penalty wearing an LLM costume. To isolate the LLM-specific part, we compare LLM
and mature questions *of the same length* (and same reputation, code, etc.). That
is what putting those variables in a regression does.

**Real result:** the raw answered gap is −8.6pp; after adjustment it shrinks to
**−5.1pp** (section 4.7). So about 3.5pp of the raw gap was composition — and
5.1pp is genuinely associated with being an LLM question.

## 4.2 Odds, odds ratios, and the logit

Probabilities don't combine nicely in a linear model, so logistic regression works
in **odds** = `p / (1 − p)`.

```
mature: 73 answered / 27 not  ->  odds = 73/27 = 2.70
LLM:    64 answered / 36 not  ->  odds = 64/36 = 1.78
odds ratio (OR) = 1.78 / 2.70 = 0.66
```

An **odds ratio of 0.66** means LLM questions have 0.66× the odds of being
answered. OR < 1 = worse, OR > 1 = better, OR = 1 = no difference. (Odds ratios
are not the same as probability ratios — that is why we also report the
probability gap in 4.7.)

The model is linear in the **log-odds (logit)**:

```
logit(p) = ln( p/(1−p) ) = b0 + b1·IsLLM + b2·LogBodyLength + ... + month effects
exp(b1) = the odds ratio for IsLLM, holding everything else fixed.
```

**Real result:** main model OR for IsLLM = **0.78** (answered), 0.70 (accepted) —
the gap clearly survives adjustment.

## 4.3 Post-treatment bias (why Score / Views / Comments are excluded)

A control variable must be **fixed before the outcome happens**. Score, view
count, and comment count are measured *long after* posting and are partly *caused
by* the outcome:

```
IsLLM  ->  (gets answered?)  ->  ViewCount     (answered questions get bumped, viewed more)
```

`ViewCount` is a **descendant of the outcome**. "Controlling for" a variable on
the causal path between cause and outcome (or caused by the outcome) is
**post-treatment bias** — it removes part of the very effect we are trying to
measure and can flip or distort it.

**Real evidence this is not hypothetical:** when we add these variables (our
labelled *sensitivity* model), the answered gap moves from −5.1pp to −10.3pp and
its confidence interval blows up (from ±0.6pp to ±7.6pp). The estimate becomes
both larger and far noisier — exactly the instability post-treatment controls
cause. So the **main model uses only posting-time variables**; the version with
Score/Views/Comments is reported *only* as a sensitivity check, clearly labelled.

## 4.4 Robust (HC1) standard errors

A standard error (SE) is the uncertainty of a coefficient. The classical formula
assumes the model's variance structure is exactly right (homoskedasticity). When
that is not quite true (heteroskedasticity — the noise is larger for some
questions than others), classical SEs are wrong. **HC1 "robust" standard errors**
stay valid under heteroskedasticity. With our huge `n` the coefficients barely
move, but robust SEs are the safe, standard choice, so we use `cov_type="HC1"`.

## 4.5 Month fixed effects (`C(Month)`)

`C(Month)` adds a **separate intercept for every calendar month**. This soaks up
anything that affected *all* questions in a given month equally — a platform-wide
slow patch, a holiday, a sitewide policy change.

**Why it matters for us:** it means the LLM effect is estimated from **within-month
LLM-vs-mature differences**, not contaminated by the fact that LLM and mature
questions are spread differently across time. If answer rates fell for everyone in
December 2025, the December intercept absorbs it. (This is the regression analogue
of the "is it website-wide or LLM-specific?" question — see Part III.)

## 4.6 Reputation was missing for some askers — how we handled it, and proof it doesn't matter

Some questions have **no owner reputation** because the account was deleted or is
community-owned. This is **not random**: deleted accounts may differ systematically
(e.g. spammers), and the missing rate itself differs by group (**mature 1.42% vs
LLM 0.84%**).

**Our approach (the "missing-indicator method"):** fill the missing reputation with
the median, **and** add a 0/1 flag `ReputationMissing`. The flag lets the model
give those questions their own baseline, so the made-up median value does not
distort anything — the flag does the real work, and we keep the rows instead of
throwing them away (dropping them could bias the comparison, since the two groups
lose different fractions).

**Why this is defensible and not just convenient:** reputation here is a *control*,
not the thing we are studying, and we **proved the choice is harmless** by re-fitting
the main model three ways:

```
                              IsLLM odds ratio (answered)   IsLLM OR (accepted)
(c) per-group median + flag [MAIN]      0.780                    0.699
(a) pooled median + flag                0.780                    0.699
(b) complete-case (drop missing)        0.781                    0.701
```

The headline is **identical to three decimal places** across schemes. So how we
handle missing reputation does not drive the result — one sentence closes the
objection.

## 4.7 The adjusted probability gap (average adjusted predictions)

Odds ratios are hard to feel. So we also report the gap on the **probability
scale**, using **G-computation** (a.k.a. average adjusted predictions):

1. Take the fitted model. Pretend **every** question is mature (`IsLLM = 0`),
   predict each one's answer probability, average them → `p_mature`.
2. Pretend **every** question is LLM (`IsLLM = 1`), predict, average → `p_LLM`.
3. The **gap = p_LLM − p_mature**, holding the real covariate mix fixed.

This answers "if the same population of questions were labelled LLM vs mature,
what is the difference in answer rate?" in plain percentage points.

**Real result:** answered gap = **−5.1pp**, 95% CI [−5.7, −4.4]; accepted gap =
**−7.7pp**, 95% CI [−8.3, −7.0].

The confidence interval on that gap comes from the **delta method** (it propagates
the model's coefficient uncertainty to the averaged prediction) via the
`marginaleffects` library. We verified its point estimate matches our manual
G-computation exactly, then use the library so the gap carries a proper CI rather
than being a bare number.

---

# PART III — Did the gap change over time? (interaction model)

This ties RQ2 to the paper's "emerging topic → knowledge domain" framing: as the
LLM area matures, is it *catching up* to the baseline?

## 5.1 What an interaction is

An **interaction** lets one variable's effect *depend on another*. We add a linear
time index (months since the start) and the product `IsLLM × time`:

```
logit(p) = b0 + b1·IsLLM + b2·time + b3·(IsLLM × time) + controls
```

- `b1` = the LLM gap at time = 0.
- `b3` = **how the gap changes per month**. This is the headline number.
  - `b3 > 0` → the LLM penalty shrinks over time (catching up / maturing).
  - `b3 < 0` → the penalty grows (falling further behind).
  - `b3 = 0` → the gap is stable.

We **center** the time index (so 0 = the middle of the window, not month 1), which
just makes `b1` interpretable as "the gap in the middle of the period."

**Worked intuition.** If `b3 = −0.006` per month, then over 12 months the LLM
log-odds penalty deepens by `12 × 0.006 = 0.072` — the gap is *widening*, not
closing.

## 5.2 Why this also separates "website-wide" from "LLM-specific"

The `time` main effect (`b2`) captures the trend shared by **both** groups (e.g.
answer rates drifting down platform-wide). The **interaction** (`b3`) captures only
the part **specific to LLM questions**. The monthly **gap series** (LLM rate minus
mature rate, plotted per month) shows the same thing visually: if a dip hits both
groups, it cancels in the difference, so a moving gap is LLM-specific by
construction. This is the direct answer to "is it just a Stack-Overflow-wide
thing?" — no, because the mature group rides the same platform and we subtract it
out.

**Real result:** the answered-gap interaction is ≈ 0 (p = 0.06 — no significant
change); the accepted-gap interaction is **negative and significant**
(p = 0.0003). So the gap is **not closing** — for accepted answers it is slightly
**widening**. Maturation in *volume* (RQ1) has not translated into closing the
*answerability* gap.

---

# PART IV — Survival analysis (the honest time-to-answer)

Two earlier numbers are each a bit unsatisfying: the windowed binary outcome
throws away *when* within 30 days an answer arrived, and the median
time-to-answer is computed **only among answered questions** — which is a
selection bias (it ignores the questions that were never answered, and those are
exactly the interesting failures). **Survival analysis fixes both.**

## 6.1 Censoring, the survival function, and the hazard

- **Censoring** (again): if a question was watched for 10 days and never answered,
  we don't know its true time-to-answer — only that it is **> 10 days**. That is a
  *censored* observation, and survival analysis uses it correctly instead of
  discarding it.
- **Survival function S(t)** = probability a question is **still unanswered** at
  time `t`. It starts at 1 and steps down as answers arrive.
- **Hazard h(t)** = the instantaneous rate of getting answered *right now, given
  not yet answered*. Higher hazard = answered faster.

We define: `duration` = time to first answer (if answered) or time observed (if
never answered, censored); `event = 1` if answered, `0` if censored.

## 6.2 Kaplan-Meier estimator (worked example)

The **Kaplan-Meier (KM)** curve estimates `S(t)` from censored data. At each time
an answer occurs, it multiplies the survival by `(1 − answers/at-risk)`.

**Example.** Five questions:

```
q1: answered day 1
q2: answered day 2
q3: censored day 2   (stopped watching, never answered)
q4: answered day 5
q5: censored day 6

day 1: at risk = 5, 1 answer  -> S = 1 × (1 − 1/5) = 0.80
day 2: at risk = 4, 1 answer  -> S = 0.80 × (1 − 1/4) = 0.60   (q3 then leaves, censored)
day 5: at risk = 2, 1 answer  -> S = 0.60 × (1 − 1/2) = 0.30   (only q4, q5 remain)
day 6: censored, no answer    -> S stays 0.30
```

Curve: `1.0 → 0.80 → 0.60 → 0.30`. The **median survival** is the time `S` crosses
0.5 (here day 5). Crucially, q3 (censored at day 2) **counts in the at-risk set**
on days 1–2 but is **not** treated as a failure — that is how censoring is honored.

**Real result:** median time-to-answer is **0.20 days (~5h) for mature** vs
**1.99 days (~48h) for LLM** — LLM questions take roughly 10× longer to reach the
half-answered mark.

## 6.3 Log-rank test

The **log-rank test** asks whether two KM curves differ. At every time an answer
occurs, it compares the **observed** number of answers in each group to the number
**expected** if both groups shared one survival curve, then accumulates the
discrepancies (much like a chi-square spread over time). Big accumulated
discrepancy → the curves really differ.

**Real result:** log-rank χ² = 1109, p ≈ 10⁻²⁴³ — the two curves differ strongly.

## 6.4 Cox proportional-hazards model and the hazard ratio

**Cox regression** is the survival analogue of logistic regression: it models the
hazard with covariates and yields a **hazard ratio (HR)** for IsLLM, adjusted for
the posting-time controls.

```
h(t) = h0(t) · exp( b·IsLLM + controls ),   HR = exp(b)
```

- **HR < 1** → lower hazard of being answered → answered slower / less. **HR > 1**
  → faster.

**Real result:** **HR for IsLLM = 0.81** (95% CI 0.79–0.82). At any given moment,
an LLM question is answered at ~81% the rate of an otherwise-identical mature
question.

**Its assumption — proportional hazards:** Cox assumes the hazard *ratio* between
groups is roughly constant over time (the curves don't cross). It is the main
caveat of this model.

**Why we sub-sample for Cox:** Cox on all 1.5M mature rows is needlessly heavy, so
we fit it on all LLM questions plus a random 100k mature sample. Kaplan-Meier and
the log-rank test use the full data.

## 6.5 Why three methods instead of one

The windowed logistic regression, the descriptive comparison, and survival
analysis each have different blind spots, but **all three agree**: LLM questions
are answered less and slower (OR 0.78, HR 0.81, median 5h vs 48h). Agreement
across methods with different assumptions is the strongest evidence we can offer
short of an experiment.

---

## 7. One-page summary

| Method | Outcome type | Question it answers | Key effect size | Main assumption | Our result |
|---|---|---|---|---|---|
| Proportion + Wilson CI | binary | what is the rate? | rate, risk difference | none | answered 64.5% vs 73.1% |
| Chi-square + Cramér's V | binary | are the rates different? how strongly? | risk diff (−8.6pp), V (0.022) | expected counts not tiny | yes, p≈10⁻¹⁶³ |
| Mann-Whitney U + RBC/CLES | continuous/skewed | does one group's distribution sit higher? | CLES (0.648), RBC (+0.30) | none (rank-based) | LLM slower to answer |
| Holm-Bonferroni | any family | controlling false positives | — | — | all headline results survive |
| Logistic regression | binary | does the gap survive adjustment? | odds ratio (0.78), adj. gap (−5.1pp) | correct link; no post-treatment controls | gap survives |
| Robust (HC1) SE | — | valid uncertainty under heteroskedasticity | — | — | CIs barely change |
| Interaction (IsLLM×time) | binary | is the gap closing over time? | per-month change in gap (b3) | linear time trend | not closing; accepted widening |
| Kaplan-Meier + log-rank | time-to-event | full time-to-answer with censoring | median survival; curve difference | independent censoring | 5h vs 48h, p≈10⁻²⁴³ |
| Cox PH | time-to-event | adjusted hazard of being answered | hazard ratio (0.81) | proportional hazards | LLM answered ~19% slower |

---

## 8. Glossary

- **Right-censoring** — we stopped observing before the event; we know only a
  lower bound on the time.
- **Proportion / rate** — fraction with a yes outcome.
- **Risk difference** — rate(LLM) − rate(mature), in percentage points.
- **Confidence interval** — plausible range for an estimate; 95% = if we repeated
  the study many times, ~95% of such intervals would contain the truth.
- **p-value** — probability of seeing data this extreme if there were truly no
  difference. Small = "unlikely to be pure chance." Says nothing about *size*.
- **Effect size** — how *big* the difference is (vs the p-value's "is it real").
- **Odds** = p/(1−p). **Odds ratio** = ratio of two groups' odds.
- **Logit** = ln(odds); the scale logistic regression is linear on.
- **Confounder** — a variable linked to both group and outcome that can fake or
  hide an effect; we adjust for it.
- **Post-treatment variable** — measured after / caused by the outcome;
  controlling for it biases the estimate.
- **Heteroskedasticity** — unequal noise across observations; robust SEs handle it.
- **Fixed effect** — a separate intercept per category (here, per month).
- **Interaction** — when one variable's effect depends on another.
- **Survival function S(t)** — probability the event hasn't happened by time t.
- **Hazard** — instantaneous event rate given it hasn't happened yet.
- **Hazard ratio** — multiplicative effect on the hazard; <1 = slower.
- **Non-parametric** — makes no assumption about the data's distribution shape.

---

## 9. Caveats to state alongside the numbers

- **These numbers are provisional.** They were produced on a *test* version of the
  validated LLM set (annotation not yet finalised by the team). Magnitudes will
  move once real annotation drops low-precision detection rules; the *methods*
  here are final.
- **Maturity confound.** LLM topics are also simply *new* topics, and the mature
  baseline is *old* topics. So the contrast partly mixes "LLM-ness" with "newness."
  Fully separating them would need a young-non-LLM baseline (cut for scope). State
  conclusions as about "LLM-related questions," avoid causal claims about LLM-ness
  itself.
- **Windowed outcomes exclude the youngest ~month** of questions (the price of the
  censoring fix).
- **Closure rates are conservative** (deleted posts are invisible in the data).
- **LLM-set precision and recall.** The LLM set's precision is whatever the
  validation pipeline measured (low-precision detection rules are dropped);
  **recall is unknown and fixed at SEDE extraction time** — questions that discuss
  LLMs but matched no tag/keyword were never collected and cannot be recovered
  offline. Phrase conclusions as being about "detected LLM-related questions."
- **No subset / alternative-window robustness checks.** The planned robustness
  section was dropped for scope: detection precision is already handled by the
  validation pipeline, and the 30/90-day windows follow common practice rather
  than being stress-tested. (The reputation-imputation robustness check *was* run —
  the IsLLM estimate is stable across imputation schemes.)
- **A few negative time-to-answer values** exist (exactly 2 rows, both mature):
  the stored first-answer timestamp predates the question, a known Stack Exchange
  artifact of **merged-duplicate / migrated** posts (the earlier answer from the
  other post gets associated, so `MIN(answer date)` predates `CreationDate`). The
  pipeline now guards against this: a within-window answer must satisfy
  `0 ≤ gap ≤ window`, and negative times are set to missing (consistent with the
  survival code, which drops non-positive durations). Impact either way is ~0.0002%.
```
