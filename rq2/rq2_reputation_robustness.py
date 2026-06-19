"""
RQ2 robustness check (HANDOFF issue #3): is the IsLLM estimate stable under
different ways of handling MISSING owner reputation?

Missing OwnerReputation = deleted / community account. It is NOT missing at
random and is plausibly correlated with the outcome, so how we handle it
could in principle move the headline. We confirm it does not, by re-fitting
the MAIN answerability model under three schemes and comparing the IsLLM
odds ratio:

  (c) group-median + flag  -> per-dataset median impute + ReputationMissing
                              flag. THIS IS THE MAIN MODEL.
  (a) pooled-median + flag -> single median across both datasets + flag.
  (b) complete-case        -> drop rows with missing reputation entirely
                              (no flag; it would be all-zero).

If IsLLM is stable across all three, the imputation choice is not driving
the result - one sentence in the paper closes the objection.

We also report the missing-reputation RATE by group: if it differs (it
does: mature is higher), that is exactly why complete-case could bias the
comparison and why we keep the rows via the flag in the main model.

Raw reputation is reconstructed from the cached frame as
expm1(LogOwnerReputation) for non-missing rows, so no re-streaming is
needed.

Run:
    python rq2_reputation_robustness.py
"""

import os

import numpy as np
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf

from rq2_prepare import (
    load_or_build_rq2_frame, ANSWER_WINDOW_DAYS, ACCEPT_WINDOW_DAYS,
    PARENT_PATH,
)

OUTCOMES = [
    f"AnsweredWithin{ANSWER_WINDOW_DAYS}Days",
    f"AcceptedWithin{ACCEPT_WINDOW_DAYS}Days",
]

# Controls shared by every scheme (the reputation term is swapped in).
COMMON_TERMS = [
    "IsLLM", "LogBodyLength", "TitleLength",
    "HasCodeBlock", "HasInlineCode", "TagCount", "C(Month)",
]


def reconstruct_raw_reputation(df):
    """Raw reputation for non-missing rows (NaN for missing). The cached
    LogOwnerReputation is log1p(reputation) with missing rows already
    median-imputed, so expm1 recovers the real value for non-missing rows;
    the flag tells us which to blank out."""
    df = df.copy()
    df["RepRaw"] = np.expm1(df["LogOwnerReputation"])
    df.loc[df["ReputationMissing"] == 1, "RepRaw"] = np.nan

    # (c) group/per-dataset median + flag = exactly the main-model column.
    df["LogRep_group"] = df["LogOwnerReputation"]
    # (a) pooled median + flag.
    pooled_median = df["RepRaw"].median()
    df["LogRep_pooled"] = np.log1p(df["RepRaw"].fillna(pooled_median))
    return df


def fit_scheme(df, outcome, rep_term, include_flag, complete_case):
    data = df
    if complete_case:
        data = data[data["ReputationMissing"] == 0]

    terms = (
        ["IsLLM", "LogBodyLength", "TitleLength", rep_term]
        + (["ReputationMissing"] if include_flag else [])
        + ["HasCodeBlock", "HasInlineCode", "TagCount", "C(Month)"]
    )
    columns = [outcome] + [t for t in terms if t != "C(Month)"] + ["Month"]
    d = data[columns].dropna().copy()
    d["Month"] = d["Month"].astype("category")

    formula = f"{outcome} ~ " + " + ".join(terms)
    model = smf.glm(
        formula=formula, data=d, family=sm.families.Binomial()
    ).fit(cov_type="HC1")

    conf = model.conf_int().loc["IsLLM"]
    return {
        "IsLLM_Coef": model.params["IsLLM"],
        "IsLLM_OR": np.exp(model.params["IsLLM"]),
        "OR_CI_Low": np.exp(conf[0]),
        "OR_CI_High": np.exp(conf[1]),
        "PValue": model.pvalues["IsLLM"],
        "N": int(model.nobs),
    }


SCHEMES = [
    # label, rep_term, include_flag, complete_case
    ("(c) group-median + flag [MAIN]", "LogRep_group", True, False),
    ("(a) pooled-median + flag", "LogRep_pooled", True, False),
    ("(b) complete-case", "LogOwnerReputation", False, True),
]


def run_robustness(df):
    rows = []
    for outcome in OUTCOMES:
        for label, rep_term, flag, cc in SCHEMES:
            res = fit_scheme(df, outcome, rep_term, flag, cc)
            res.update({"Outcome": outcome, "Scheme": label})
            rows.append(res)
    cols = ["Outcome", "Scheme", "N", "IsLLM_OR",
            "OR_CI_Low", "OR_CI_High", "IsLLM_Coef", "PValue"]
    return pd.DataFrame(rows)[cols]


if __name__ == "__main__":
    frame = load_or_build_rq2_frame()
    frame = reconstruct_raw_reputation(frame)

    print("\nMissing-reputation rate by group:")
    print(frame.groupby("Dataset")["ReputationMissing"].mean().round(4)
          .to_string())

    table = run_robustness(frame)
    print("\n=== IsLLM estimate under reputation-handling schemes ===")
    print(table.round(4).to_string(index=False))

    out_path = os.path.join(PARENT_PATH, "outputs", "rq2")
    os.makedirs(out_path, exist_ok=True)
    table.to_csv(
        os.path.join(out_path, "rq2_reputation_robustness.csv"), index=False
    )
    print(f"\nWrote rq2_reputation_robustness.csv to {out_path}")
