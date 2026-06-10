"""
RQ1: How has the volume of LLM-related Stack Overflow questions evolved
over time compared with mature programming topics?

Pipeline:
  1. Load the CLEANED LLM dataset produced by the validation pipeline
     (falls back to the raw export with a warning).
  2. Recompute detection types from the KEPT rules only.
  3. Aggregate monthly volumes with explicit handling of missing months.
  4. Plot: LLM volume, LLM vs mature (log), LLM share of all SO
     questions, detection-type composition (absolute + 100%).
  5. Mann-Kendall trend tests with Sen's slope, on the full series and
     on the pre-peak / post-peak segments.

Run from inside rq1/:
    python rq1_analysis.py
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pymannkendall as mk

# ============================================================
# SECTION 1: CONFIGURATION AND PATHS
# ============================================================

CURRENT_PATH = os.path.dirname(__file__)
PARENT_PATH = os.path.dirname(CURRENT_PATH)
DATA_PATH = os.path.join(PARENT_PATH, "data")

# Preferred input: the cleaned dataset from the validation pipeline.
CLEANED_LLM_FILENAME = os.path.join(
    PARENT_PATH, "outputs", "validation", "llm-data-cleaned.csv"
)
RAW_LLM_FILENAME = os.path.join(DATA_PATH, "llm", "llm-combined-data.csv")

RULE_DECISIONS_FILENAME = os.path.join(
    PARENT_PATH, "outputs", "validation", "rule-decision-table.csv"
)

MATURE_COUNTS_FILENAME = os.path.join(
    DATA_PATH, "mature", "mature-monthly-counts.csv"
)

# total monthly SO question counts (single SEDE aggregate).
# Expected columns: Month, TotalQuestionCount
TOTAL_COUNTS_FILENAME = os.path.join(DATA_PATH, "utility", "total-monthly-counts.csv")

OUTPUT_PATH = os.path.join(PARENT_PATH, "outputs", "rq1")

# Analysis window (matches the SEDE extraction window).
ANALYSIS_START = "2022-01"
ANALYSIS_END = "2026-05"

# The final month of extraction may be incomplete; set True to drop it.
DROP_LAST_MONTH = False

SHOW_PLOTS = True


# ============================================================
# SECTION 2: DATA LOADING
# ============================================================

def load_llm_data():
    """Load the cleaned LLM dataset; fall back to raw with a warning."""
    if os.path.exists(CLEANED_LLM_FILENAME):
        print(f"Using cleaned dataset: {CLEANED_LLM_FILENAME}")
        df = pd.read_csv(CLEANED_LLM_FILENAME)
    else:
        warnings.warn(
            "Cleaned dataset not found - falling back to the RAW export. "
            "Results will include unvalidated false positives. Run the "
            "validation pipeline before producing final paper numbers."
        )
        df = pd.read_csv(RAW_LLM_FILENAME)

    df = df.drop_duplicates(subset="QuestionId").copy()

    df["CreationDate"] = pd.to_datetime(df["CreationDate"], errors="coerce")
    df["Month"] = df["CreationDate"].dt.to_period("M")

    return df


def load_kept_rules():
    """Set of rules the validation pipeline decided to KEEP, or None if
    the decision table does not exist (raw-data fallback mode)."""
    if not os.path.exists(RULE_DECISIONS_FILENAME):
        return None

    decisions = pd.read_csv(RULE_DECISIONS_FILENAME)
    return set(decisions.loc[decisions["Decision"] == "KEEP", "Rule"])


# ============================================================
# SECTION 3: DETECTION TYPE (RECOMPUTED FROM KEPT RULES)
# ============================================================

def add_detection_type(df, kept_rules):
    """
    Classify each question as Tag only / Keyword only / Tag and keyword.

    IMPORTANT: after rule filtering, the classification must be based on
    the KEPT rules only. A question originally matched by tag:chatbot and
    kw:chatgpt becomes keyword-only once tag:chatbot is dropped. Using
    the original SEDE flags here would misclassify such questions.
    """
    df = df.copy()

    if kept_rules is not None and "MatchedRules" in df.columns:
        def kept_channels(rules_str):
            rules = set(str(rules_str).split("|")) & kept_rules
            has_tag = any(r.startswith("tag:") for r in rules)
            has_kw = any(r.startswith("kw:") for r in rules)
            return has_tag, has_kw

        channels = df["MatchedRules"].fillna("").apply(kept_channels)
        df["HasTagDetection"] = [c[0] for c in channels]
        df["HasKeywordDetection"] = [c[1] for c in channels]
    else:
        # Fallback: original SEDE flags (raw-data mode).
        df["HasTagDetection"] = df["IsLLMTagged"].fillna(0).astype(int) == 1
        df["HasKeywordDetection"] = (
            df["IsLLMKeywordDetected"].fillna(0).astype(int) == 1
        )

    df["DetectionType"] = np.select(
        [
            df["HasTagDetection"] & df["HasKeywordDetection"],
            df["HasTagDetection"] & ~df["HasKeywordDetection"],
            ~df["HasTagDetection"] & df["HasKeywordDetection"],
        ],
        ["Tag and keyword", "Tag only", "Keyword only"],
        default="Other",
    )

    n_other = (df["DetectionType"] == "Other").sum()
    assert n_other == 0, (
        f"{n_other} questions match no kept detection channel - the "
        f"cleaned dataset and the rule decision table are out of sync."
    )

    return df


# ============================================================
# SECTION 4: MONTHLY AGGREGATION
# ============================================================

def full_month_index():
    """Explicit monthly index over the analysis window, so missing
    months are visible decisions rather than fillna(0) accidents."""
    return pd.period_range(ANALYSIS_START, ANALYSIS_END, freq="M")


def build_monthly_counts(llm_df):
    """Unique-question counts per month, reindexed to the full window."""
    counts = (
        llm_df.groupby("Month")["QuestionId"]
        .nunique()
        .reindex(full_month_index(), fill_value=0)
        .rename("LLMQuestionCount")
        .rename_axis("Month")
        .reset_index()
    )

    if DROP_LAST_MONTH:
        counts = counts.iloc[:-1].copy()

    return counts


def build_detection_monthly(llm_df):
    """Unique-question counts per month and detection type."""
    detection = (
        llm_df.groupby(["Month", "DetectionType"])["QuestionId"]
        .nunique()
        .rename("Questions")
        .reset_index()
    )

    pivot = (
        detection.pivot(index="Month", columns="DetectionType",
                        values="Questions")
        .reindex(full_month_index(), fill_value=0)
        .fillna(0)
        .astype(int)
    )

    if DROP_LAST_MONTH:
        pivot = pivot.iloc[:-1].copy()

    return pivot


def load_external_monthly(filename, count_column):
    """Load mature/total monthly count files; None if absent."""
    if not os.path.exists(filename):
        return None

    df = pd.read_csv(filename)
    df["Month"] = (
        pd.to_datetime(df["Month"], errors="coerce").dt.to_period("M")
    )
    return df[["Month", count_column]]


def merge_series(llm_counts, mature_counts, total_counts):
    """
    Merge LLM counts with the mature baseline and (optionally) total
    platform counts. Left-join on the LLM month index: months without
    mature/total data stay NaN (visible), they are NOT silently zeroed.
    """
    merged = llm_counts.copy()

    if mature_counts is not None:
        merged = merged.merge(mature_counts, on="Month", how="left")
        missing = merged["MatureQuestionCount"].isna().sum()
        if missing > 0:
            print(f"NOTE: {missing} months have no mature-count data; "
                  f"they appear as gaps, not zeros.")

    if total_counts is not None:
        merged = merged.merge(total_counts, on="Month", how="left")
        merged["LLMShareOfTotalPct"] = (
            100 * merged["LLMQuestionCount"] / merged["TotalQuestionCount"]
        )

    return merged


# ============================================================
# SECTION 5: TREND TESTING (MANN-KENDALL + SEN'S SLOPE)
# ============================================================
# Standard formulas with tie correction; verified against the
# pymannkendall package. The MK test detects a monotonic trend without
# distributional assumptions; Sen's slope is the median of all pairwise
# slopes and serves as the effect size (questions per month).

def mann_kendall_test(values):
    values = np.asarray(values, dtype=float)
    n = len(values)

    if n < 8:
        return {"n": n, "S": np.nan, "Z": np.nan, "PValue": np.nan,
                "Trend": "series too short", "SenSlopePerMonth": np.nan}

    result = mk.original_test(values)

    trend = result.trend if result.h else "no significant trend"

    return {"n": n, "S": int(result.s), "Z": float(result.z),
            "PValue": float(result.p), "Trend": trend,
            "SenSlopePerMonth": float(result.slope)}


def run_trend_tests(monthly_counts):
    """
    MK tests on three segments:
      - full series (often uninformative when the trend reverses),
      - growth phase: start -> peak month,
      - post-peak phase: peak month -> end.
    """
    series = monthly_counts.set_index("Month")["LLMQuestionCount"]
    peak_month = series.idxmax()

    segments = {
        "Full series": series,
        f"Growth phase (start to peak {peak_month})": series.loc[:peak_month],
        f"Post-peak phase ({peak_month} to end)": series.loc[peak_month:],
    }

    rows = []
    for name, segment in segments.items():
        result = mann_kendall_test(segment.values)
        result["Segment"] = name
        rows.append(result)

    columns = ["Segment", "n", "Trend", "SenSlopePerMonth", "Z", "PValue", "S"]
    return pd.DataFrame(rows)[columns], peak_month


# ============================================================
# SECTION 6: FIGURES
# ============================================================

def save_and_show(figure_name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, figure_name), dpi=200)
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def month_axis(months):
    """Period index -> timestamps for clean matplotlib date handling."""
    return pd.PeriodIndex(months).to_timestamp()


def plot_llm_volume(monthly_counts, peak_month):
    plt.figure(figsize=(12, 5))
    x = month_axis(monthly_counts["Month"])
    plt.plot(x, monthly_counts["LLMQuestionCount"], marker="o", ms=3)

    peak_value = monthly_counts["LLMQuestionCount"].max()
    plt.annotate(
        f"peak: {peak_month} ({peak_value:,})",
        xy=(peak_month.to_timestamp(), peak_value),
        xytext=(10, -5), textcoords="offset points",
    )

    plt.xlabel("Month")
    plt.ylabel("LLM-related questions")
    plt.title("Monthly volume of LLM-related Stack Overflow questions")
    save_and_show("rq1_llm_volume.png")


def plot_llm_vs_mature(merged):
    if "MatureQuestionCount" not in merged.columns:
        return

    plt.figure(figsize=(12, 5))
    x = month_axis(merged["Month"])
    plt.plot(x, merged["LLMQuestionCount"], marker="o", ms=3,
             label="LLM-related")
    plt.plot(x, merged["MatureQuestionCount"], marker="o", ms=3,
             label="Mature topics")
    plt.yscale("log")
    plt.xlabel("Month")
    plt.ylabel("Questions (log scale)")
    plt.title("LLM-related vs mature-topic question volume")
    plt.legend()
    save_and_show("rq1_llm_vs_mature_log.png")


def plot_llm_share(merged):
    if "LLMShareOfTotalPct" not in merged.columns:
        print("No total-monthly-counts.csv found - skipping the share "
              "plot. See README/report plan: this controls for the "
              "platform-wide volume decline.")
        return

    plt.figure(figsize=(12, 5))
    x = month_axis(merged["Month"])
    plt.plot(x, merged["LLMShareOfTotalPct"], marker="o", ms=3)
    plt.xlabel("Month")
    plt.ylabel("LLM-related share of all SO questions (%)")
    plt.title("LLM-related questions as a share of all Stack Overflow "
              "questions")
    save_and_show("rq1_llm_share_of_total.png")


def plot_detection_composition(pivot):
    x = month_axis(pivot.index)

    # Absolute stacked
    plt.figure(figsize=(12, 5))
    bottom = np.zeros(len(pivot))
    for column in pivot.columns:
        plt.bar(x, pivot[column], width=20, bottom=bottom, label=column)
        bottom += pivot[column].values
    plt.xlabel("Month")
    plt.ylabel("LLM-related questions")
    plt.title("Detection channel of LLM-related questions over time")
    plt.legend()
    save_and_show("rq1_detection_stacked.png")

    # 100% normalized: shows composition shift (topic formalization)
    shares = pivot.div(pivot.sum(axis=1).replace(0, np.nan), axis=0) * 100
    plt.figure(figsize=(12, 5))
    bottom = np.zeros(len(shares))
    for column in shares.columns:
        values = shares[column].fillna(0).values
        plt.bar(x, values, width=20, bottom=bottom, label=column)
        bottom += values
    plt.xlabel("Month")
    plt.ylabel("Share of monthly LLM questions (%)")
    plt.title("Detection channel composition (normalized)")
    plt.legend()
    save_and_show("rq1_detection_composition_pct.png")


# ============================================================
# SECTION 7: MAIN
# ============================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    llm_df = load_llm_data()
    kept_rules = load_kept_rules()
    llm_df = add_detection_type(llm_df, kept_rules)

    monthly_counts = build_monthly_counts(llm_df)
    detection_pivot = build_detection_monthly(llm_df)

    mature_counts = load_external_monthly(
        MATURE_COUNTS_FILENAME, "MatureQuestionCount"
    )
    total_counts = load_external_monthly(
        TOTAL_COUNTS_FILENAME, "TotalQuestionCount"
    )

    merged = merge_series(monthly_counts, mature_counts, total_counts)

    # ---------------- Trend tests ----------------
    trend_results, peak_month = run_trend_tests(monthly_counts)
    print("\nMann-Kendall trend tests (Sen's slope = questions/month):")
    print(trend_results.to_string(index=False))
    trend_results.to_csv(
        os.path.join(OUTPUT_PATH, "rq1_trend_tests.csv"), index=False
    )

    # ---------------- Summary table ----------------
    summary = {
        "Total LLM questions": int(monthly_counts["LLMQuestionCount"].sum()),
        "Analysis window": f"{ANALYSIS_START} to {ANALYSIS_END}",
        "Peak month": str(peak_month),
        "Peak monthly count": int(monthly_counts["LLMQuestionCount"].max()),
        "Mean monthly count": round(
            float(monthly_counts["LLMQuestionCount"].mean()), 1
        ),
    }
    if "MatureQuestionCount" in merged.columns:
        summary["Total mature questions (overlapping months)"] = int(
            merged["MatureQuestionCount"].sum(skipna=True)
        )
    if "LLMShareOfTotalPct" in merged.columns:
        peak_share_idx = merged["LLMShareOfTotalPct"].idxmax()
        summary["Peak LLM share of all SO questions"] = (
            f"{merged.loc[peak_share_idx, 'LLMShareOfTotalPct']:.2f}% "
            f"({merged.loc[peak_share_idx, 'Month']})"
        )

    summary_df = pd.DataFrame(summary.items(), columns=["Statistic", "Value"])
    print("\nRQ1 summary:")
    print(summary_df.to_string(index=False))
    summary_df.to_csv(
        os.path.join(OUTPUT_PATH, "rq1_summary.csv"), index=False
    )

    merged.to_csv(
        os.path.join(OUTPUT_PATH, "rq1_monthly_series.csv"), index=False
    )

    # ---------------- Figures ----------------
    plot_llm_volume(monthly_counts, peak_month)
    plot_llm_vs_mature(merged)
    plot_llm_share(merged)
    plot_detection_composition(detection_pivot)

    print(f"\nWrote tables and figures to {OUTPUT_PATH}")