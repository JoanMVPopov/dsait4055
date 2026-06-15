"""
RQ2 section 4: survival analysis of time-to-first-answer.

The honest version of "how long until a question gets answered". Unlike
the median-time-to-answer descriptive (which conditions on being answered,
a selection bias) and unlike the windowed binary outcomes (which threshold
at 30/90 days), survival analysis uses EVERY question and handles censoring
natively: a never-answered question contributes its full observed time as
a censored observation.

  - duration = days to first answer (answered) OR days observed until the
    snapshot (never answered, i.e. right-censored).
  - event    = 1 if answered within the observation window, else 0.

Three pieces, all via lifelines (no hand-rolled survival math):
  1. Kaplan-Meier curves: P(still unanswered) over time, LLM vs mature.
  2. Log-rank test for the difference between the two curves.
  3. Cox proportional-hazards model: hazard of receiving a first answer,
     with IsLLM + the posting-time controls. HR < 1 for IsLLM => LLM
     questions are answered at a lower rate, all else equal.

The Cox model is fit on ALL LLM questions plus a random sub-sample of
mature questions (Cox on the full ~1.5M mature rows is needlessly heavy);
KM and the log-rank test use the full data.

Run:
    python rq2_survival.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test

from rq2_prepare import load_or_build_rq2_frame, PARENT_PATH

# Posting-time controls (same family as the main regression).
COX_COVARIATES = [
    "IsLLM", "LogBodyLength", "TitleLength", "LogOwnerReputation",
    "ReputationMissing", "HasCodeBlock", "HasInlineCode", "TagCount",
]

# Mature rows sampled for the Cox fit (all LLM rows are kept).
COX_MATURE_SAMPLE = 100_000
RANDOM_SEED = 42

# KM plot horizon (days). Most of the action is early; the tail is flat.
KM_HORIZON_DAYS = 90

OUTPUT_PATH = os.path.join(PARENT_PATH, "outputs", "rq2")


# ============================================================
# SECTION 1: BUILD SURVIVAL VARIABLES
# ============================================================

def build_survival(df):
    """duration (days) + event from the windowed-outcome columns already
    in the frame. Answered => time to first answer; otherwise censored at
    the observed time. Non-positive durations are dropped."""
    df = df.copy()
    df["event"] = df["TimeToFirstAnswerHours"].notna().astype(int)
    df["duration_days"] = np.where(
        df["event"] == 1,
        df["TimeToFirstAnswerHours"] / 24.0,
        df["ObservedDays"],
    )

    n_before = len(df)
    df = df[df["duration_days"] > 0].copy()
    dropped = n_before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with non-positive duration "
              f"(posted at snapshot / answered same instant).")
    return df


# ============================================================
# SECTION 2: KAPLAN-MEIER + LOG-RANK
# ============================================================

def kaplan_meier(df):
    kmf = KaplanMeierFitter()

    fig, ax = plt.subplots(figsize=(10, 6))
    medians = {}
    for is_llm, label in [(0, "Mature"), (1, "LLM")]:
        mask = df["IsLLM"] == is_llm
        kmf.fit(df.loc[mask, "duration_days"],
                df.loc[mask, "event"], label=label)
        kmf.plot_survival_function(ax=ax, ci_show=True)
        medians[label] = kmf.median_survival_time_

    ax.set_xlim(0, KM_HORIZON_DAYS)
    ax.set_xlabel("Days since posting")
    ax.set_ylabel("P(still unanswered)")
    ax.set_title("Kaplan-Meier: probability a question is still unanswered")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, "rq2_survival_km.png"), dpi=200)
    plt.close()

    return medians


def log_rank(df):
    llm = df["IsLLM"] == 1
    mature = df["IsLLM"] == 0
    result = logrank_test(
        df.loc[llm, "duration_days"], df.loc[mature, "duration_days"],
        event_observed_A=df.loc[llm, "event"],
        event_observed_B=df.loc[mature, "event"],
    )
    return result.test_statistic, result.p_value


# ============================================================
# SECTION 3: COX PROPORTIONAL HAZARDS
# ============================================================

def cox_model(df):
    """Cox PH on all LLM + a mature sub-sample. HR > 1 => higher hazard of
    being answered (faster); HR < 1 => slower / less likely."""
    llm = df[df["IsLLM"] == 1]
    mature = df[df["IsLLM"] == 0]
    mature_sample = mature.sample(
        n=min(COX_MATURE_SAMPLE, len(mature)), random_state=RANDOM_SEED
    )
    cox_df = pd.concat([llm, mature_sample], ignore_index=True)
    cox_df = cox_df[["duration_days", "event"] + COX_COVARIATES].dropna()

    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="duration_days", event_col="event")

    summary = cph.summary[["coef", "exp(coef)",
                           "exp(coef) lower 95%", "exp(coef) upper 95%",
                           "p"]].copy()
    summary = summary.rename(columns={
        "exp(coef)": "HazardRatio",
        "exp(coef) lower 95%": "HR_CI_Low",
        "exp(coef) upper 95%": "HR_CI_High",
    })
    return summary, int(cph.event_observed.shape[0])


# ============================================================
# SECTION 4: MAIN
# ============================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    frame = load_or_build_rq2_frame()
    surv = build_survival(frame)

    print("\nEvents (answered) by group:")
    print(surv.groupby("Dataset")["event"].agg(["size", "mean"]).round(3)
          .to_string())

    medians = kaplan_meier(surv)
    print("\nMedian time-to-answer (days; KM, NaN => >50% never answered "
          "in window):")
    for label, m in medians.items():
        print(f"  {label}: {m}")

    stat, p_value = log_rank(surv)
    print(f"\nLog-rank test: chi2 = {stat:.1f}, p = {p_value:.3e}")

    cox_summary, cox_n = cox_model(surv)
    print(f"\nCox PH (all LLM + {COX_MATURE_SAMPLE} mature sub-sample, "
          f"N = {cox_n}):")
    print(cox_summary.round(4).to_string())

    pd.DataFrame(
        [{"Group": k, "MedianSurvivalDays": v} for k, v in medians.items()]
        + [{"Group": "logrank_chi2", "MedianSurvivalDays": stat},
           {"Group": "logrank_p", "MedianSurvivalDays": p_value}]
    ).to_csv(os.path.join(OUTPUT_PATH, "rq2_survival_km_logrank.csv"),
             index=False)
    cox_summary.to_csv(os.path.join(OUTPUT_PATH, "rq2_survival_cox.csv"))

    print(f"\nWrote survival tables + KM figure to {OUTPUT_PATH}")
