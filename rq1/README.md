# RQ1 - Volume Evolution Analysis

**RQ1: How has the volume of LLM-related Stack Overflow questions evolved
over time compared with mature programming topics?**

This folder contains `analysis.py`, which produces all RQ1 figures,
tables, and trend statistics. The analysis is descriptive plus formal
trend testing: it characterizes how LLM question volume rose, peaked, and
declined, compares it against the mature-topic baseline, and tracks how
the detection-channel composition shifted over time.

## Inputs

| File | Required | Description |
|------|----------|-------------|
| `outputs/validation/llm-data-cleaned.csv` | preferred | Validated LLM question set from the validation pipeline |
| `data/llm/llm-combined-data.csv` | fallback | Raw SEDE export; used with a loud warning if the cleaned file is missing. Final paper numbers must come from the cleaned set |
| `outputs/validation/rule-decision-table.csv` | preferred | KEEP/DROP decision per detection rule; used to recompute detection types |
| `data/mature/mature-monthly-counts.csv` | yes | Full (unsampled) monthly counts of mature-topic questions, columns `Month`, `MatureQuestionCount` |
| `data/total-monthly-counts.csv` | optional | Monthly counts of ALL Stack Overflow questions, columns `Month`, `TotalQuestionCount`. Enables the share-of-platform plot. Produced by a single fast SEDE aggregate (no sampling) |

## How to run

From inside `rq1/`:

```
python analysis.py
```


## What the script does

1. **Loading** - prefers the cleaned dataset, falls back to raw with a
   warning. Deduplicates by `QuestionId` before any counting.

2. **Detection types, recomputed from kept rules** - each question is
   classified as Tag only, Keyword only, or Tag and keyword, based ONLY
   on rules the validation pipeline kept. This matters: a question
   originally matched by a dropped tag and a kept keyword is keyword-only
   after cleaning, and the original SEDE flags would misclassify it. An
   assert guarantees no question matches zero kept channels.

3. **Monthly aggregation** - unique-question counts over an explicit
   month index for the full analysis window (2022-01 to 2026-05).
   Months with missing mature or total data appear as gaps with a printed
   note, never as silent zeros. `DROP_LAST_MONTH` exists in case the
   final extraction month is incomplete.

4. **Trend tests** - Mann-Kendall test with Sen's slope (via
   `pymannkendall`) on three segments: the full series, the growth phase
   (start to peak month), and the post-peak phase (peak to end).
   Interpretation note for the paper: "no significant trend" on the full
   series is NOT a null result when the curve reverses - it is the reason
   the segmented tests exist. Sen's slope is the effect size, in
   questions per month.

5. **Figures** - five, saved as 200 dpi PNGs:
   - `rq1_llm_volume.png` - monthly LLM volume with the peak annotated
   - `rq1_llm_vs_mature_log.png` - LLM vs mature volume, log scale
   - `rq1_llm_share_of_total.png` - LLM share of all SO questions
     (skipped with a notice if the optional denominator file is absent).
     This is the headline RQ1 figure when available: it is the only one
     immune to the platform-wide volume decline confound
   - `rq1_detection_stacked.png` - detection channel, absolute counts
   - `rq1_detection_composition_pct.png` - detection channel, normalized
     to 100 percent. If the tagged share grows over time, that is the
     community formalizing the topic into proper tags - direct evidence
     for the "emerging topic to knowledge domain" framing

## Outputs

Everything is written to `outputs/rq1/`:

| File | Content |
|------|---------|
| `rq1_monthly_series.csv` | Merged monthly series (LLM, mature, total, share) - source for any number quoted in the paper |
| `rq1_trend_tests.csv` | Mann-Kendall results per segment |
| `rq1_summary.csv` | Headline statistics (totals, peak month, peak share) |
| five PNG figures | listed above |

## Configuration

Constants at the top of the script:

| Constant | Default | Meaning |
|----------|---------|---------|
| `ANALYSIS_START` / `ANALYSIS_END` | 2022-01 / 2026-05 | Month window, must match the SEDE extraction |
| `DROP_LAST_MONTH` | False | Drop the final month if extraction cut mid-month |
| `SHOW_PLOTS` | True | Also display figures interactively |


## Relation to the validation pipeline

Run order: `validation/llm_detection.py`, then the three
`validation/llm_validation.py` stages, then this script. RQ1 runs
without the validation outputs (raw fallback mode) for development, but
final numbers require the cleaned dataset and the rule decision table.