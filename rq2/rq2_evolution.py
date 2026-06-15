"""
RQ2 section 6: evolution of the answerability gap over time.

This is the part that ties RQ2 back to the paper title ("from emerging
topic to knowledge domain"): is the LLM answerability gap CLOSING as the
domain matures, or widening?

Three pieces, all on the censoring-fixed windowed outcomes so late months
are not artifacts:

  1. Monthly answered-rate and accepted-rate for both groups.
  2. The gap series: LLM rate minus mature rate per month, with a zero
     line. One glance answers "converging or diverging?".
  3. Formal test: a logistic model with a LINEAR month index plus an
     IsLLM x month interaction (replacing the month fixed effects of the
     main regression). The interaction coefficient is the headline:
       > 0  -> gap shrinking on the odds scale (LLM maturing toward baseline)
       < 0  -> gap widening.

Month index is centered, so the IsLLM main effect reads as the gap at the
MIDDLE of the window rather than at month 0.

Caveat: the final ~month is sparse (few questions are old enough to be
window-eligible), so the gap-series tail is noisy - read trends, not the
last point.

Run:
    python rq2_evolution.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # save figures without a display
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf

from rq2_prepare import (
    load_or_build_rq2_frame, ANSWER_WINDOW_DAYS, ACCEPT_WINDOW_DAYS,
    ANALYSIS_START, PARENT_PATH,
)

OUTCOMES = [
    f"AnsweredWithin{ANSWER_WINDOW_DAYS}Days",
    f"AcceptedWithin{ACCEPT_WINDOW_DAYS}Days",
]

# Posting-time controls, same family as the main regression (minus the
# month fixed effects, which the linear index + interaction replace).
INTERACTION_CONTROLS = [
    "LogBodyLength", "TitleLength", "LogOwnerReputation",
    "ReputationMissing", "HasCodeBlock", "HasInlineCode", "TagCount",
]

OUTPUT_PATH = os.path.join(PARENT_PATH, "outputs", "rq2")


# ============================================================
# SECTION 1: MONTH INDEX + MONTHLY RATES
# ============================================================

def add_month_index(df):
    """Numeric months since ANALYSIS_START (centered), plus a timestamp for
    plotting."""
    mp = pd.PeriodIndex(df["Month"].astype(str), freq="M")
    start = pd.Period(ANALYSIS_START, freq="M")
    idx = np.asarray((mp.year - start.year) * 12 + (mp.month - start.month),
                     dtype=float)

    df = df.copy()
    df["MonthIndex"] = idx
    df["MonthIndexC"] = idx - idx.mean()
    return df


def monthly_rates(df, outcome):
    """Per-month rate and N for each group, eligible rows only."""
    sub = df.dropna(subset=[outcome])
    rate = sub.groupby(["Month", "Dataset"])[outcome].mean().unstack("Dataset")
    n = sub.groupby(["Month", "Dataset"])[outcome].size().unstack("Dataset")
    rate = rate.sort_index()
    n = n.sort_index()
    return rate, n


def gap_series(rate):
    if "LLM" in rate.columns and "Mature" in rate.columns:
        return (rate["LLM"] - rate["Mature"]).rename("LLMminusMature")
    return None


def _month_axis(index):
    return pd.PeriodIndex(index.astype(str), freq="M").to_timestamp()


# ============================================================
# SECTION 2: FIGURES
# ============================================================

def plot_rates(df):
    fig, axes = plt.subplots(len(OUTCOMES), 1, figsize=(12, 9), sharex=True)
    for ax, outcome in zip(axes, OUTCOMES):
        rate, _ = monthly_rates(df, outcome)
        x = _month_axis(rate.index)
        for group in ["Mature", "LLM"]:
            if group in rate.columns:
                ax.plot(x, rate[group], marker="o", ms=3, label=group)
        ax.set_ylabel(f"{outcome}\n(rate)")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_title("Monthly answer / accept rates: LLM vs mature "
                      "(windowed outcomes)")
    axes[-1].set_xlabel("Month")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, "rq2_evolution_rates.png"), dpi=200)
    plt.close()


def plot_gaps(df):
    plt.figure(figsize=(12, 5))
    for outcome in OUTCOMES:
        rate, _ = monthly_rates(df, outcome)
        gap = gap_series(rate)
        if gap is not None:
            plt.plot(_month_axis(gap.index), gap.values, marker="o", ms=3,
                     label=outcome)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Month")
    plt.ylabel("LLM rate - mature rate")
    plt.title("Answerability gap over time (LLM minus mature)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, "rq2_evolution_gap.png"), dpi=200)
    plt.close()


# ============================================================
# SECTION 3: INTERACTION MODEL
# ============================================================

def fit_interaction(df, outcome):
    """Logistic GLM: outcome ~ IsLLM * MonthIndexC + posting-time controls,
    robust (HC1) SEs. The IsLLM:MonthIndexC term is the headline."""
    columns = [outcome, "IsLLM", "MonthIndexC"] + INTERACTION_CONTROLS
    data = df[columns].dropna().copy()

    formula = (
        f"{outcome} ~ IsLLM * MonthIndexC + "
        + " + ".join(INTERACTION_CONTROLS)
    )
    model = smf.glm(
        formula=formula, data=data, family=sm.families.Binomial()
    ).fit(cov_type="HC1")
    return model, data


def interaction_summary(df):
    rows = []
    for outcome in OUTCOMES:
        model, data = fit_interaction(df, outcome)
        for term in ["IsLLM", "MonthIndexC", "IsLLM:MonthIndexC"]:
            rows.append({
                "Outcome": outcome,
                "Term": term,
                "Coefficient": model.params[term],
                "OddsRatio": np.exp(model.params[term]),
                "PValue": model.pvalues[term],
                "N": int(model.nobs),
            })
    return pd.DataFrame(rows)


# ============================================================
# SECTION 4: MAIN
# ============================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    frame = load_or_build_rq2_frame()
    frame = add_month_index(frame)

    # Monthly rate + gap tables (source for any quoted number).
    for outcome in OUTCOMES:
        rate, n = monthly_rates(frame, outcome)
        rate.to_csv(os.path.join(OUTPUT_PATH, f"rq2_evolution_rate_{outcome}.csv"))
        gap = gap_series(rate)
        if gap is not None:
            gap.to_csv(os.path.join(OUTPUT_PATH,
                                    f"rq2_evolution_gap_{outcome}.csv"))

    plot_rates(frame)
    plot_gaps(frame)

    summary = interaction_summary(frame)
    print("\n=== IsLLM x time interaction "
          "(IsLLM:MonthIndexC is the headline) ===")
    print(summary.round(5).to_string(index=False))
    summary.to_csv(
        os.path.join(OUTPUT_PATH, "rq2_evolution_interaction.csv"), index=False
    )

    print(f"\nWrote evolution tables + figures to {OUTPUT_PATH}")
