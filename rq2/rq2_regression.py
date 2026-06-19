"""
RQ2 adjusted logistic regression (plan section 5).

Question: does the LLM answerability gap survive controlling for question
and asker characteristics that are FIXED AT POSTING TIME?

Main model covariates are all known the moment the question is posted:
  IsLLM, log body length, title length, log owner reputation (+ missing
  flag), has-code-block, tag count, and month fixed effects.

Deliberately EXCLUDED from the main model: Score, ViewCount,
CommentCount. These are measured after posting and are partly CAUSED by
the outcome (an accepted answer drives views; unanswerable questions
attract clarifying comments). Conditioning on them is post-treatment bias.
They appear only in a separate sensitivity model, reported as such.

Outputs odds ratios with robust (HC1) CIs and average adjusted
predictions (the AME-style probability gap), which are far more
interpretable than raw logit coefficients.

Run:
    python rq2_regression.py
"""

import os

import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf

from marginaleffects import avg_comparisons

from rq2_prepare import (
    load_or_build_rq2_frame, ANSWER_WINDOW_DAYS, ACCEPT_WINDOW_DAYS,
    PARENT_PATH,
)

# ============================================================
# SECTION 1: MODEL SPECIFICATIONS
# ============================================================

PRE_POSTING_TERMS = [
    "IsLLM",
    "LogBodyLength",       # now PROSE length (code blocks excluded in prepare)
    "TitleLength",
    "LogOwnerReputation",
    "ReputationMissing",
    "HasCodeBlock",        # presence of a <pre> code block
    "HasInlineCode",       # presence of inline <code>; distinct signal now
    "TagCount",
    "C(Month)",
]
# NOTE on code variables: prepare also exposes CodeLength/LogCodeLength
# (amount of code). Do NOT add LogCodeLength alongside HasCodeBlock - they
# are collinear (LogCodeLength > 0 iff HasCodeBlock == 1). If you want code
# VOLUME instead of presence, swap HasCodeBlock -> LogCodeLength.

# Post-treatment variables - sensitivity model ONLY.
POST_TREATMENT_TERMS = [
    "ScoreNum",
    "LogViewCount",
    "CommentCountNum",
]


def build_formula(outcome, terms):
    return f"{outcome} ~ " + " + ".join(terms)


# ============================================================
# SECTION 2: FITTING
# ============================================================

def fit_logit(df, outcome, terms):
    """
    Fit a logistic GLM with robust (HC1) standard errors on the rows
    eligible for this (windowed) outcome.
    """
    model_columns = [outcome] + [
        t for t in terms if t != "C(Month)"
    ] + ["Month"]
    data = df[model_columns].dropna().copy()
    # marginaleffects requires categorical (not string) factors; C(Month)
    # still treats it as a fixed effect.
    data["Month"] = data["Month"].astype("category")

    formula = build_formula(outcome, terms)
    model = smf.glm(
        formula=formula, data=data, family=sm.families.Binomial()
    ).fit(cov_type="HC1")

    return model, data


# ============================================================
# SECTION 3: REPORTING HELPERS
# ============================================================

def odds_ratio_table(model):
    """Odds ratios with robust CIs. Drops the month fixed-effect dummies
    from the printed table - they are nuisance controls."""
    params = model.params
    conf = model.conf_int()
    conf.columns = ["low", "high"]

    table = pd.DataFrame({
        "Coefficient": params,
        "OddsRatio": np.exp(params),
        "OR_CI_Low": np.exp(conf["low"]),
        "OR_CI_High": np.exp(conf["high"]),
        "PValue": model.pvalues,
    })

    table = table[~table.index.str.startswith("C(Month)")]
    return table


def average_adjusted_prediction_gap(model, data):
    """
    Average adjusted prediction (AAP) on the probability scale: the LLM gap
    in probability terms, holding the covariate distribution fixed (a.k.a.
    G-computation / average marginal effect of IsLLM).

    The two point predictions are counterfactual means (force IsLLM to 0,
    then 1, and average). The GAP and its 95% CI come from
    marginaleffects.avg_comparisons, which applies the delta method to the
    model's (HC1-robust) covariance - so the gap now carries a CI instead
    of being a bare point estimate. The estimate matches the manual
    p1 - p0 (verified); we report marginaleffects' value for consistency
    with its CI.
    """
    d0 = data.copy()
    d0["IsLLM"] = 0
    p0 = model.predict(d0).mean()

    d1 = data.copy()
    d1["IsLLM"] = 1
    p1 = model.predict(d1).mean()

    cmp = avg_comparisons(
        model, variables={"IsLLM": [0, 1]}
    ).to_pandas().iloc[0]

    return {
        "AAP_Mature": p0,
        "AAP_LLM": p1,
        "AAP_Gap_LLMminusMature": float(cmp["estimate"]),
        "AAP_Gap_SE": float(cmp["std_error"]),
        "AAP_Gap_CI_Low": float(cmp["conf_low"]),
        "AAP_Gap_CI_High": float(cmp["conf_high"]),
    }


# ============================================================
# SECTION 4: RUN ALL MODELS
# ============================================================

def run_models(df):
    outcomes = [
        f"AnsweredWithin{ANSWER_WINDOW_DAYS}Days",
        f"AcceptedWithin{ACCEPT_WINDOW_DAYS}Days",
    ]

    or_tables = {}
    gap_rows = []

    for outcome in outcomes:
        # --- Main model (pre-posting controls only) ---
        model, data = fit_logit(df, outcome, PRE_POSTING_TERMS)
        or_tables[f"{outcome} [main]"] = odds_ratio_table(model)

        gap = average_adjusted_prediction_gap(model, data)
        gap.update({"Outcome": outcome, "Model": "main (pre-posting)",
                    "N": int(model.nobs)})
        gap_rows.append(gap)

        # --- Sensitivity model (adds post-treatment controls) ---
        model_s, data_s = fit_logit(
            df, outcome, PRE_POSTING_TERMS + POST_TREATMENT_TERMS
        )
        or_tables[f"{outcome} [sensitivity]"] = odds_ratio_table(model_s)

        gap_s = average_adjusted_prediction_gap(model_s, data_s)
        gap_s.update({"Outcome": outcome,
                      "Model": "sensitivity (+post-treatment)",
                      "N": int(model_s.nobs)})
        gap_rows.append(gap_s)

    gap_table = pd.DataFrame(gap_rows)[
        ["Outcome", "Model", "N", "AAP_Mature", "AAP_LLM",
         "AAP_Gap_LLMminusMature", "AAP_Gap_SE",
         "AAP_Gap_CI_Low", "AAP_Gap_CI_High"]
    ]
    return or_tables, gap_table


if __name__ == "__main__":
    frame = load_or_build_rq2_frame()
    or_tables, gap_table = run_models(frame)

    out_path = os.path.join(PARENT_PATH, "outputs", "rq2")
    os.makedirs(out_path, exist_ok=True)

    print("\n=== Average adjusted predictions (probability gap) ===")
    print(gap_table.round(4).to_string(index=False))
    gap_table.to_csv(
        os.path.join(out_path, "rq2_adjusted_prediction_gaps.csv"),
        index=False
    )

    for name, table in or_tables.items():
        print(f"\n=== Odds ratios: {name} ===")
        print(table.round(4).to_string())
        safe = name.replace(" ", "_").replace("[", "").replace("]", "")
        table.to_csv(os.path.join(out_path, f"rq2_oddsratios_{safe}.csv"))

    print(f"\nWrote regression tables to {out_path}")
