# LLM Question Detection - Validation Pipeline

This folder contains the offline validation pipeline for the LLM-related
question dataset extracted from SEDE. Its purpose is to (1) attribute every
question's inclusion to the specific detection rule(s) that caused it,
(2) measure the precision of each rule through manual annotation, and
(3) produce a cleaned dataset by dropping rules with empirically low
precision - without re-running any SEDE queries.

The cleanup is **data-driven**: no tag or keyword is removed by judgment
call. Rules are kept or dropped based on precision measured on a manually
labeled sample, against a written definition agreed on before labeling.

## Files

| File | Role |
|------|------|
| `llm_detection.py` | Re-derives detection offline, per rule. Run once. |
| `llm_validation.py` | Sampling, annotation handling, precision analysis, filtering. Run in three stages. |
| `llm_posthoc_manual_review.py` | Adjudicates the `MANUAL_REVIEW` rules left by `analyze` (KEEP/DROP per rule). Run after Step 3 if any questions need review. |

## Expected folder layout

```
project/
├── data/
│   └── llm/
│       └── llm-combined-data.csv      # raw SEDE export (input)
├── outputs/
│   └── validation/                    # everything below is written here
└── validation/
    ├── README.md
    ├── llm_detection.py
    ├── llm_validation.py
    └── llm_posthoc_manual_review.py
```

Paths are resolved relative to the script location (same convention as the
rq1/rq2 scripts). Run everything from inside `validation/`.

## Detection semantics

`llm_detection.py` mirrors the original SEDE query exactly, so the offline
attribution is auditable against the extraction:

- **Tag rules** - one rule per tag in the original `@LLMTags` list
  (the *full* list, including suspect entries like `openai-gym` and
  `chatbot`).
- **Keyword rules** - one rule per keyword, matched as a case-insensitive
  substring against the **raw HTML body** and title, exactly like SQL
  `LOWER(col) LIKE '%kw%'`. The `' llm '` keyword keeps its literal
  surrounding spaces, as in the query.

The script also recomputes `IsLLMTagged` / `IsLLMKeywordDetected` from
scratch and reports disagreements with the SEDE-exported flags. A nonzero
disagreement count usually indicates an HTML-escaping or encoding mismatch
and should be investigated **before** trusting per-rule results.

## Workflow

### Step 0 - Re-detection (once)

```
python llm_detection.py
```

Reads `data/llm/llm-combined-data.csv`, deduplicates by `QuestionId`,
writes to `outputs/validation/`:

- `llm-data-with-rules.csv` - the dataset plus `MatchedTagRules`,
  `MatchedKeywordRules`, `MatchedRules` (pipe-separated, prefixed
  `tag:` / `kw:`).
- `rule-frequency-table.csv` - per rule: total questions matched and
  questions matched *exclusively* (i.e., how much data is lost if the
  rule is dropped).

### Step 1 - Draw the annotation sample

```
python llm_validation.py sample
```

Draws a stratified sample. Each question is
assigned to exactly one stratum: the **rarest** of its matched rules.
This guarantees low-frequency rules are
represented instead of being absorbed by broad rules like `kw:chatgpt`.
Allocation is proportional to stratum size with a floor of 5 per stratum.

Outputs:

- `annotation-A.csv`, `annotation-B.csv` - identical files, one per
  annotator. Columns: `QuestionId`, `Link`, `Title`, `BodyExcerpt`,
  `IsLLMRelated` (to fill: 1 or 0), `Notes`. Annotators do **not** see
  which rules matched - this prevents anchoring on the detection
  mechanism.
- `annotation-key.csv` - rule attribution and stratum per sampled
  question. Keep this away from annotators until labeling is done.
- `annotation-definition.txt` - the written definition of "LLM-related"
  both annotators must use. See variable `ANNOTATION_DEFINITION` in the script

Annotation protocol: two team members label all rows independently,
without discussing individual questions. Open the `Link` if the excerpt
is insufficient.

### Step 2 - Agreement check

```
python llm_validation.py agreement
```

Requires both annotation files fully filled with 0/1 (the script rejects
blanks or other values). Computes raw agreement and **Cohen's κ**, and
writes:

- `merged-annotations.csv` - both label columns side by side.
- `disagreements.csv` - rows where annotators disagree, with an empty
  `FinalLabel` column. The **third** team member fills it (0/1) to
  resolve each disagreement.

### Step 3 - Precision analysis and filtering

```
python llm_validation.py analyze
```

Requires all disagreements resolved. Produces:

- **Per-rule precision** - computed over every labeled question the rule
  matched (not only its sampling stratum), with Wilson 95% confidence
  intervals.
- **Rule decisions** - `KEEP` if precision ≥ threshold (default 0.70),
  `DROP` if below, `MANUAL_REVIEW` if the rule has fewer than 5 labeled
  examples. Review rules never keep a question on their own; questions
  matched *only* by such rules are exported to
  `questions-needing-review.csv` for a manual call.
- **Filtering** - a question is kept iff at least one of its matched
  rules is `KEEP`.
- **Sanity check** - monthly counts of cleaned questions before
  November 2022 (ChatGPT launch). Assuming that LLMs were not as widely used and recognized before 
the launch, we should expect a small percentage (or maybe even zero)

Outputs:

- `rule-decision-table.csv` - precision, CI, counts, decision per rule
  (paper appendix material).
- `llm-data-cleaned.csv` - the final dataset. Point the RQ1/RQ2 scripts
  at this file instead of `llm-combined-data.csv`. (If any rules came out
  `MANUAL_REVIEW`, run Step 4 before treating this as final.)

### Step 4 - Post-hoc manual review (only if needed)

`analyze` cannot auto-decide a rule with fewer than 5 labeled examples, so
it marks those `MANUAL_REVIEW` and exports every question gated *only* by
such rules to `questions-needing-review.csv`. Those questions are **not**
in the cleaned set yet. This step records an explicit decision for each
such rule and folds the result back in - **without** modifying the frozen
pipeline above (it imports and reuses `llm_validation.filter_dataset`, so
the filtering logic lives in exactly one place).

The unit of decision is the **rule**, not the individual question: you are
vouching that a low-evidence detection rule is (un)trustworthy, the same
rule-based logic the precision threshold uses - not hand-picking questions.

```
python llm_posthoc_manual_review.py        # 1st run: writes a template
# fill the Decision column (KEEP/DROP) in manual-rule-decisions.csv
python llm_posthoc_manual_review.py        # 2nd run: applies the decisions
```

The **first run** writes `manual-rule-decisions.csv`, one row per rule that
gates a review question, with context to make the call:

| Rule | LabeledQuestions | Precision | QuestionsGated | ExampleQuestionId | ExampleTitle | Decision |
|------|-----------------|-----------|----------------|-------------------|--------------|----------|
| `tag:gemini-code-assist` | 1 | 0.0 | 1 | 79724577 | How do I save the file using Cmd+S... | *(fill: KEEP/DROP)* |
| `tag:h2ogpt` | 3 | 1.0 | 1 | 77985749 | h2ogpt: unable to add document...    | *(fill: KEEP/DROP)* |

Fill `Decision` with `KEEP` or `DROP` per rule. The **second run** applies
them and rewrites, in `outputs/validation/`:

- `rule-decision-table.csv` - the adjudicated rules flip from
  `MANUAL_REVIEW` to `KEEP`/`DROP`, with a `ManualOverride` flag set so the
  manual calls stay auditable in the appendix table.
- `llm-data-cleaned.csv` - now includes the questions gated by KEEP'd
  rules (same schema as the `analyze` output).
- `questions-needing-review.csv` - rewritten with any still-undecided rows
  (empty once everything is resolved).

Properties: the script is **idempotent** (re-running after a successful
apply finds nothing to do) and **re-analyze-safe** (if you re-run `analyze`,
which resets the table, just run this script again - the saved
`manual-rule-decisions.csv` is re-applied automatically). It **fails loud**
if a gating rule has no valid `KEEP`/`DROP`, rather than silently dropping
questions, matching the "nothing decided implicitly" property of the rest
of the pipeline.

## Configuration

All knobs sit at the top of `llm_validation.py`:

| Constant | Default | Meaning |
|----------|---------|---------|
| `RANDOM_SEED` | 42 | Reproducible sampling (report it in the paper) |
| `TOTAL_SAMPLE_SIZE` | 250 | Annotation budget |
| `MIN_PER_STRATUM` | 5 | Floor per stratum |
| `BODY_EXCERPT_CHARS` | 1500 | Body text shown to annotators |
| `PRECISION_THRESHOLD` | 0.70 | KEEP/DROP cutoff |
| `MIN_LABELS_PER_RULE` | 5 | Below this: MANUAL_REVIEW |
| `ANNOTATION_DEFINITION` | (text) | The labeling definition |

## Things that can be reported

- Sample size, stratification scheme (disjoint strata by rarest matched
  rule, proportional allocation with floor), and the random seed.
- The written annotation definition (appendix).
- Raw agreement and Cohen's κ; disagreement resolution by a third
  annotator.
- Per-rule precision with Wilson 95% CIs and the threshold decision
  (the rule decision table as an appendix table), including any rules
  resolved by post-hoc manual review (the `ManualOverride` column).
- Size of the dataset before and after filtering, and the precision among
  labeled questions in the cleaned set.

## Caveats and limitations

- Per-rule precision estimates are valid, but because the sample is
  **stratified** (rare rules oversampled), the pooled "precision among
  labeled cleaned questions" is not an unbiased estimate of dataset-level
  precision - report it with that caveat.
- The pipeline can only improve **precision**. Recall was fixed at SEDE
  extraction time; questions discussing LLMs that matched neither tags
  nor keywords were
  never collected. This as a limitation.
