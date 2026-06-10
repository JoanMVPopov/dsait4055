# RQ2 - Plan

**RQ2: Are LLM-related questions less answerable than mature programming
questions?**

Primary outcome: answerability. Secondary outcome: community reception
(absorbs the dropped RQ3 on friction, scoped down). A third angle ties RQ2
back to the paper's title: does the answerability gap change over time as
the LLM domain matures?

---

## 1. Inputs

- `outputs/validation/llm-data-cleaned.csv` - the validated LLM question
  set. RQ2 must NOT run on the raw export for final numbers.
- Both datasets deduplicated by QuestionId before anything else.

## 2. Outcome definitions (the censoring fix)

Raw "has an answer" / "has an accepted answer" flags are right-censored:
recently posted questions haven't had the same time to be answered, and
LLM questions are concentrated late in the window. This biases results in
the direction of our hypothesis, so it must be fixed, not footnoted.

Windowed outcomes, identical for both datasets:

- **AnsweredWithin30Days** - first answer arrived ≤ 30 days after the
  question. Questions younger than 30 days at extraction time are
  excluded from this outcome.
- **AcceptedWithin90Days** - accepted answer arrived ≤ 90 days after the
  question. Questions younger than 90 days are excluded.
- **TimeToFirstAnswerHours** - kept for descriptives, but interpreted
  with care: comparing it only among answered questions conditions on
  the outcome (selection bias). The survival analysis (4) is the honest
  version of this comparison.

Window lengths (30/90) are choices, not truths - we state them, and one
robustness run with alternative windows (e.g., 7/30) confirms conclusions
don't hinge on them.

## 3. Descriptive comparison + tests

For LLM vs mature, on the windowed outcomes:

- Rates with 95% CIs, presented as a small table.
- Chi-square tests for the binary outcomes; report **risk difference**
  and Cramér's V as effect sizes.
- Mann-Whitney U for skewed continuous variables (time to answer among
  answered, answer count); report **rank-biserial correlation**.
- **Holm-Bonferroni** correction across the family of tests.
- Writing guidance: with our sample sizes every p-value will be tiny;
  the text should lead with effect sizes, p-values are a formality.

## 4. Survival analysis (stretch goal, drop first if out of time)

Time-to-first-answer analyzed properly, including unanswered questions:

- **Kaplan-Meier curves** of "probability of still being unanswered" over
  time since posting, LLM vs mature, with a log-rank test.
- **Cox proportional hazards** model: hazard of receiving a first answer,
  with IsLLM and the pre-posting controls from 5.

Handles censoring natively and replaces boxplots that silently exclude
unanswered questions. `lifelines` library; ~40 lines. If skipped, the
windowed outcomes in 2 already neutralize the censoring problem - this
adds rigor and a strong figure, not correctness.

## 5. Logistic regression (the adjusted comparison)

Question: does the LLM gap survive controlling for question and asker
characteristics?

- Outcome: AnsweredWithin30Days (and AcceptedWithin90Days as a second
  model).
- Predictor of interest: IsLLM.
- Controls: **only variables fixed at posting time** - body length,
  title length, log owner reputation, has-code-block indicator, tag
  count, month fixed effects.
- Explicitly EXCLUDED from the main model: Score, ViewCount,
  CommentCount. These are measured after posting and are partly caused
  by the outcome (post-treatment bias). The old specification including
  them is kept as a reported sensitivity check only.
- Report odds ratios with CIs and adjusted predicted probabilities
  (rates at average covariate values), robust (HC) standard errors.

## 6. Evolution of the gap over time (ties RQ2 to the title)

Three pieces:

1. Monthly plots of answered-rate and accepted-rate for both groups on
   shared axes (windowed definitions, so late months aren't artifacts).
2. The **gap series**: LLM rate minus mature rate per month, with a zero
   line. One glance answers "converging or diverging?"
3. Formal test: a model variant replacing month fixed effects with a
   linear month index plus an **IsLLM × time** interaction. The
   interaction coefficient is the headline number: negative = the gap
   widens, positive = the LLM domain is maturing toward the baseline.

## 7. Community reception (the friction remnant)

One subsection, not a separate RQ. Same two-group comparison machinery:

- **Closure rate** (ClosedDate not null) - chi-square + risk difference.
- **Score** distribution - Mann-Whitney + rank-biserial.
- **Comment count** - same.

Framing: "beyond whether questions get answered, how does the community
receive them?" Caveat for Limitations: SEDE shows closure only for
not-yet-deleted posts, so closure rates are conservative lower bounds.

## 8. Robustness checks

- Re-run headline comparisons on the **tag-only detected** subset
  (highest-precision detection). One sentence in the paper if results
  hold.
- Re-run by detection type (tag-only vs keyword-only) - if keyword-only
  questions behave very differently, that says something about detection
  quality and belongs in Limitations.
- Alternative windows (2).

## 9. Limitations to document (collect as we go)

- Mature sample capped-month truncation (if not re-extracted).
- Windowed outcomes exclude the youngest questions.
- Closure rates conservative (deleted posts invisible).
- LLM set precision is X% (from validation), recall unknown and fixed at
  extraction time.
- Maturity confound: LLM topics are also simply NEW topics; without a
  young-tag baseline we cannot separate "LLM-specific" from "any emerging
  topic" effects. State this honestly - it is the main internal-validity
  limitation of RQ2.

## 10. Deliverables

Figures: KM curves (if 4 done), monthly rates + gap series, adjusted
predicted probabilities.
Tables: descriptive comparison with effect sizes and corrected p-values,
regression table (main + sensitivity), community reception table.
All written to `outputs/rq2/`.

Suggested order of work: 2 → 3 → 5 → 6 → 7 → 8 → 4.
4 is last on purpose: most impressive, least essential.