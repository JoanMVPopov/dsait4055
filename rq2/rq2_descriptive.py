"""
RQ2 descriptive comparison (plan section 3).

Compares LLM vs mature questions on the censoring-fixed outcomes:
  - binary outcomes (answered/accepted within window):
      rates with 95% CIs, chi-square test, risk difference, Cramer's V
  - skewed continuous outcomes (time to first answer, etc.):
      Mann-Whitney U, rank-biserial correlation
  - Holm-Bonferroni correction across the whole family of tests.

Reporting guidance: with these sample sizes every p-value will be tiny.
Lead with effect sizes (risk difference, Cramer's V, rank-biserial);
treat p-values as a formality.

Run:
    python rq2_descriptive.py
"""

import os

import numpy as np
import pandas as pd

from scipy.stats import chi2_contingency, mannwhitneyu
from scipy.stats.contingency import association
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.multitest import multipletests

from rq2_prepare import (
    load_or_build_rq2_frame, ANSWER_WINDOW_DAYS, ACCEPT_WINDOW_DAYS,
    PARENT_PATH,
)

# Effect sizes:
#   - Cramer's V : scipy.stats.contingency.association(method="cramer").
#   - Mann-Whitney: scipy gives U efficiently (rank-based). From it we get
#     CLES = U / (n1*n2) = P(LLM > mature), and the rank-biserial
#     RBC = 2*CLES - 1 (Kerby 2014). These identities were checked to match
#     pingouin.mwu exactly on a small case; we do NOT call pingouin at scale
#     because its CLES uses an O(n1*n2) outer difference (~128 GiB here).
# Convention: LLM is group 1, so RBC>0 / CLES>0.5 => LLM has LARGER values
# (e.g. longer time-to-answer).


# ============================================================
# SECTION 2: BINARY OUTCOME COMPARISON
# ============================================================

def compare_binary(df, outcome):
    """Two-group comparison for one binary, censoring-eligible outcome."""
    sub = df[[outcome, "IsLLM"]].dropna()

    llm = sub.loc[sub["IsLLM"] == 1, outcome]
    mature = sub.loc[sub["IsLLM"] == 0, outcome]

    n_llm, n_mature = len(llm), len(mature)
    k_llm, k_mature = int(llm.sum()), int(mature.sum())

    p_llm = k_llm / n_llm
    p_mature = k_mature / n_mature

    ci_llm = proportion_confint(k_llm, n_llm, method="wilson")
    ci_mature = proportion_confint(k_mature, n_mature, method="wilson")

    # 2x2 contingency: rows = group, cols = outcome 0/1
    table = np.array([
        [n_mature - k_mature, k_mature],
        [n_llm - k_llm, k_llm],
    ])
    chi2, p_value, _, _ = chi2_contingency(table, correction=False)

    return {
        "Outcome": outcome,
        "Test": "chi-square",
        "N_LLM": n_llm, "N_Mature": n_mature,
        "Rate_LLM": p_llm, "Rate_Mature": p_mature,
        "CI_LLM": f"[{ci_llm[0]:.3f}, {ci_llm[1]:.3f}]",
        "CI_Mature": f"[{ci_mature[0]:.3f}, {ci_mature[1]:.3f}]",
        "RiskDifference_LLMminusMature": p_llm - p_mature,
        "EffectSize_CramersV": float(association(table, method="cramer")),
        "Statistic": chi2,
        "PValue": p_value,
    }


# ============================================================
# SECTION 3: CONTINUOUS OUTCOME COMPARISON
# ============================================================

def compare_continuous(df, outcome):
    """Mann-Whitney U for a skewed continuous outcome."""
    sub = df[[outcome, "IsLLM"]].dropna()

    llm = sub.loc[sub["IsLLM"] == 1, outcome]
    mature = sub.loc[sub["IsLLM"] == 0, outcome]

    n_llm, n_mature = len(llm), len(mature)

    # scipy U (efficient), then effect sizes from U (see header note).
    u_statistic, p_value = mannwhitneyu(llm, mature, alternative="two-sided")
    cles = u_statistic / (n_llm * n_mature)        # P(LLM > mature)
    rbc = 2 * cles - 1                             # rank-biserial (Kerby 2014)

    return {
        "Outcome": outcome,
        "Test": "Mann-Whitney U",
        "N_LLM": n_llm, "N_Mature": n_mature,
        "Median_LLM": float(llm.median()),
        "Median_Mature": float(mature.median()),
        "RiskDifference_LLMminusMature": np.nan,
        "EffectSize_RankBiserial": float(rbc),
        "EffectSize_CLES_P_LLMgtMature": float(cles),
        "Statistic": float(u_statistic),
        "PValue": float(p_value),
    }


# ============================================================
# SECTION 4: ASSEMBLE, CORRECT, REPORT
# ============================================================

def run_descriptive(df):
    binary_outcomes = [
        f"AnsweredWithin{ANSWER_WINDOW_DAYS}Days",
        f"AcceptedWithin{ACCEPT_WINDOW_DAYS}Days",
    ]
    continuous_outcomes = [
        "TimeToFirstAnswerHours",
    ]

    results = [compare_binary(df, o) for o in binary_outcomes]
    results += [compare_continuous(df, o) for o in continuous_outcomes]

    table = pd.DataFrame(results)

    # Holm-Bonferroni across the whole family.
    reject, p_adjusted, _, _ = multipletests(
        table["PValue"].values, method="holm"
    )
    table["PValue_HolmAdjusted"] = p_adjusted
    table["Significant_05"] = reject

    return table


def format_for_display(table):
    display = table.copy()
    for column in display.columns:
        if display[column].dtype == float:
            display[column] = display[column].map(
                lambda v: f"{v:.4g}" if pd.notna(v) else ""
            )
    return display


if __name__ == "__main__":
    frame = load_or_build_rq2_frame()
    table = run_descriptive(frame)

    print("\n=== RQ2 descriptive comparison (LLM vs mature) ===")
    print(format_for_display(table).to_string(index=False))

    out_path = os.path.join(PARENT_PATH, "outputs", "rq2")
    os.makedirs(out_path, exist_ok=True)
    table.to_csv(
        os.path.join(out_path, "rq2_descriptive_comparison.csv"), index=False
    )
    print(f"\nWrote rq2_descriptive_comparison.csv to {out_path}")
