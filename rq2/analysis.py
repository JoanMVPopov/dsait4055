import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu
from scipy.stats.contingency import association
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

LLM_PATH = Path("outputs/validation/llm-data-cleaned.csv")
MAT_PATH = Path("outputs/validation/mature-data-cleaned.csv")   # adjust if needed
OUT_DIR  = Path("outputs/rq2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {"LLM": "#E05A5A", "Mature": "#4A90D9"}
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

LLM_EXTRACTION_DATE = None
MAT_EXTRACTION_DATE = None



def load_data():
    llm = pd.read_csv(LLM_PATH, low_memory=False)
    mat = pd.read_csv(MAT_PATH, low_memory=False)

    llm = llm.drop_duplicates(subset="QuestionId")
    mat = mat.drop_duplicates(subset="QuestionId")

    date_cols = ["CreationDate", "ClosedDate", "LastActivityDate",
                 "AcceptedAnswerCreationDate"]
    for df in (llm, mat):
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    overlap_ids = set(llm["QuestionId"]) & set(mat["QuestionId"])
    n_overlap = len(overlap_ids)
    if n_overlap:
        mat = mat[~mat["QuestionId"].isin(overlap_ids)].copy()

    llm["group"] = "LLM"
    mat["group"] = "Mature"

    return llm, mat

def infer_extraction_date(df):
    candidates = [df["CreationDate"].max()]
    if "LastActivityDate" in df.columns:
        candidates.append(df["LastActivityDate"].max())
    return max(candidates) + pd.Timedelta(days=1)


def build_windowed_outcomes(df, extraction_date, answer_window_days=30,
                            accept_window_days=90, suffix=""):
    age_days = (extraction_date - df["CreationDate"]).dt.total_seconds() / 86_400

    answered_col  = f"AnsweredWithin{answer_window_days}Days{suffix}"
    eligible_a_col = f"EligibleAnswered{answer_window_days}{suffix}"
    accepted_col  = f"AcceptedWithin{accept_window_days}Days{suffix}"
    eligible_c_col = f"EligibleAccepted{accept_window_days}{suffix}"

    eligible_a = age_days >= answer_window_days
    if "TimeToFirstAnswerHours" in df.columns:
        answered = (df["TimeToFirstAnswerHours"].notna() &
                    (df["TimeToFirstAnswerHours"] <= answer_window_days * 24))
    else:
        answered = df.get("AnswerCount", pd.Series(0, index=df.index)) > 0
        print(f"  WARNING: TimeToFirstAnswerHours missing; "
              f"using AnswerCount>0 fallback for {answered_col}")

    df[answered_col]   = np.where(eligible_a, answered.astype(float), np.nan)
    df[eligible_a_col] = eligible_a

    eligible_c = age_days >= accept_window_days
    if "AcceptedAnswerCreationDate" in df.columns:
        tt_accept = (df["AcceptedAnswerCreationDate"] - df["CreationDate"]
                    ).dt.total_seconds() / 3_600
        accepted = tt_accept.notna() & (tt_accept <= accept_window_days * 24)
    elif "AcceptedAnswerId" in df.columns:
        accepted = df["AcceptedAnswerId"].notna()
        print(f"  WARNING: AcceptedAnswerCreationDate missing; using "
              f"AcceptedAnswerId presence (ignores timing) for {accepted_col}")
    else:
        accepted = pd.Series(False, index=df.index)
        print(f"  WARNING: no acceptance timing column; {accepted_col} all-NaN")

    df[accepted_col]   = np.where(eligible_c, accepted.astype(float), np.nan)
    df[eligible_c_col] = eligible_c

    return df


def wilson_ci(k, n):
    if np.isscalar(n) and n == 0:
        return np.nan, np.nan
    return proportion_confint(k, n, alpha=0.05, method="wilson")


def cramers_v_2x2(ct):
    return association(ct, method="cramer", correction=False)


def cles_and_rbc(u_stat, n1, n2):
    cles = u_stat / (n1 * n2)
    rbc = 2 * cles - 1
    return cles, rbc


def run_descriptives(llm, mat, answer_col, accept_col):
    rate_rows = []
    raw_tests = []   
    
    def compare_binary(outcome, label):
        l = llm[outcome].dropna()
        m = mat[outcome].dropna()
        lk, ln = l.sum(), len(l)
        mk, mn = m.sum(), len(m)
        lrate, (lcilo, lcihi) = lk / ln, wilson_ci(lk, ln)
        mrate, (mcilo, mcihi) = mk / mn, wilson_ci(mk, mn)
        rd = lrate - mrate

        ct = np.array([[lk, ln - lk], [mk, mn - mk]]).astype(int)
        chi2, p, dof, _ = chi2_contingency(ct, correction=False)
        v = cramers_v_2x2(ct)

        rate_rows.append({"Outcome": label, "Group": "LLM",
                          "n": ln, "Rate": lrate, "CI_lo": lcilo, "CI_hi": lcihi})
        rate_rows.append({"Outcome": label, "Group": "Mature",
                          "n": mn, "Rate": mrate, "CI_lo": mcilo, "CI_hi": mcihi})
        rate_rows.append({"Outcome": label, "Group": "Risk Difference (LLM-Mature)",
                          "n": "", "Rate": rd, "CI_lo": "", "CI_hi": ""})
        raw_tests.append((label, chi2, p, v, "Cramér's V"))

    def compare_continuous(col, label):
        l = llm[col].dropna()
        m = mat[col].dropna()
        u, p = mannwhitneyu(l, m, alternative="two-sided")
        cles, rbc = cles_and_rbc(u, len(l), len(m))
        rate_rows.append({"Outcome": label, "Group": "LLM",
                          "n": len(l), "Rate": l.median(),
                          "CI_lo": np.percentile(l, 25), "CI_hi": np.percentile(l, 75)})
        rate_rows.append({"Outcome": label, "Group": "Mature",
                          "n": len(m), "Rate": m.median(),
                          "CI_lo": np.percentile(m, 25), "CI_hi": np.percentile(m, 75)})
        raw_tests.append((f"{label} (CLES)", u, p, cles, "CLES"))
        raw_tests.append((f"{label} (RBC)",  u, p, rbc,  "rank-biserial r"))

    compare_binary(answer_col, "Answered within window")
    compare_binary(accept_col, "Accepted within window")
    if "TimeToFirstAnswerHours" in llm.columns:
        compare_continuous("TimeToFirstAnswerHours",
                           "Time to first answer (h, answered only)")
    if "Score" in llm.columns:
        compare_continuous("Score", "Score")
    if "CommentCount" in llm.columns:
        compare_continuous("CommentCount", "Comment count")

    p_values = [t[2] for t in raw_tests]
    reject, p_adj, _, _ = multipletests(p_values, method="holm")
    tests_df = pd.DataFrame([{
        "Test": t[0], "Statistic": t[1], "p_raw": t[2],
        "p_adj_Holm": pa, "Reject_H0": r, "EffectSize": t[3], "ES_name": t[4],
    } for t, pa, r in zip(raw_tests, p_adj, reject)])

    return pd.DataFrame(rate_rows), tests_df


def build_model_data(llm, mat, outcome, eligible_col, reputation_method="per_group_median"):
    combined = pd.concat([llm, mat], ignore_index=True)
    combined = combined[combined[eligible_col] == True].copy()
    combined["IsLLM"] = (combined["group"] == "LLM").astype(int)

    if "BodyLength" not in combined.columns:
        combined["BodyLength"] = combined.get(
            "Body", pd.Series("", index=combined.index)).str.len()
    if "TitleLength" not in combined.columns:
        combined["TitleLength"] = combined.get(
            "Title", pd.Series("", index=combined.index)).str.len()

    rep_col = "OwnerReputation" if "OwnerReputation" in combined.columns else (
        "OwnerUserReputation" if "OwnerUserReputation" in combined.columns else None)

    if rep_col is not None:
        combined["ReputationMissing"] = combined[rep_col].isna().astype(int)
        if reputation_method == "complete_case":
            combined = combined[combined[rep_col].notna()].copy()
            combined["LogReputation"] = np.log1p(combined[rep_col])
        elif reputation_method == "pooled_median":
            pooled_median = combined[rep_col].median()
            filled = combined[rep_col].fillna(pooled_median)
            combined["LogReputation"] = np.log1p(filled)
        else:
            group_medians = combined.groupby("group")[rep_col].median()
            filled = combined[rep_col].fillna(combined["group"].map(group_medians))
            combined["LogReputation"] = np.log1p(filled)
    else:
        combined["LogReputation"] = 0.0
        combined["ReputationMissing"] = 0

    combined["HasCodeBlock"] = combined.get(
        "Body", pd.Series("", index=combined.index)
    ).str.contains(r"<code>|```", regex=True, na=False).astype(int)
    combined["TagCount"] = combined.get(
        "Tags", pd.Series("", index=combined.index)).str.count(r"<").fillna(0)
    combined["Month"] = combined["CreationDate"].dt.to_period("M").astype(str)

    combined = combined[combined[outcome].notna()].copy()
    return combined


MAIN_FORMULA_RHS = (
    "IsLLM + BodyLength + TitleLength + LogReputation + ReputationMissing "
    "+ HasCodeBlock + TagCount + C(Month)"
)


def fit_logit(formula, data, cov_type="HC1"):
    try:
        return smf.logit(formula, data=data).fit(
            cov_type=cov_type, disp=False, maxiter=200)
    except np.linalg.LinAlgError:
        pass

    print("  WARNING: singular design matrix; refitting without month dummies.")
    formula_no_month = formula.replace(" + C(Month)", "")
    try:
        return smf.logit(formula_no_month, data=data).fit(
            cov_type=cov_type, disp=False, maxiter=200)
    except np.linalg.LinAlgError:
        pass

    rhs_terms = formula_no_month.split("~")[1].strip().split(" + ")
    keep_terms = []
    for term in rhs_terms:
        col = term.strip()
        if col in data.columns and data[col].nunique(dropna=True) <= 1:
            print(f"  WARNING: dropping constant column '{col}' from model "
                  f"(zero variance in this subset).")
            continue
        keep_terms.append(term)
    formula_reduced = formula_no_month.split("~")[0] + "~ " + " + ".join(keep_terms)
    print(f"  Refitting with reduced formula: {formula_reduced}")
    return smf.logit(formula_reduced, data=data).fit(
        cov_type=cov_type, disp=False, maxiter=200)


def adjusted_probability_gap(model, data, n_boot=300, seed=0):
    p_llm_point = model.predict(data.assign(IsLLM=1)).mean()
    p_mat_point = model.predict(data.assign(IsLLM=0)).mean()
    gap_point = p_llm_point - p_mat_point

    rng = np.random.default_rng(seed)
    beta_hat = model.params.values
    cov = model.cov_params().values
    boot_gaps = []
    X1 = model.model.exog.copy()
    X0 = model.model.exog.copy()
    llm_idx = list(model.params.index).index("IsLLM")
    X1[:, llm_idx] = 1
    X0[:, llm_idx] = 0
    try:
        draws = rng.multivariate_normal(beta_hat, cov, size=n_boot)
        for b in draws:
            p1 = sm.families.links.Logit().inverse(X1 @ b)
            p0 = sm.families.links.Logit().inverse(X0 @ b)
            boot_gaps.append(p1.mean() - p0.mean())
        ci_lo, ci_hi = np.percentile(boot_gaps, [2.5, 97.5])
    except Exception as e:
        print(f"  WARNING: bootstrap CI for adjusted gap failed ({e}); "
              "reporting point estimate only.")
        ci_lo, ci_hi = np.nan, np.nan

    return {
        "adj_prob_LLM": p_llm_point, "adj_prob_Mature": p_mat_point,
        "adj_prob_gap": gap_point, "gap_CI_lo": ci_lo, "gap_CI_hi": ci_hi,
        "n": len(data),
    }


def run_main_logistic(llm, mat, outcome, eligible_col):
    data = build_model_data(llm, mat, outcome, eligible_col, "per_group_median")
    if len(data) < 100:
        print(f"  Too few rows ({len(data)}) for {outcome}; skipping.")
        return None, None
    model = fit_logit(f"{outcome} ~ {MAIN_FORMULA_RHS}", data)

    or_df = pd.DataFrame({
        "OR": np.exp(model.params),
        "CI_lo": np.exp(model.conf_int()[0]),
        "CI_hi": np.exp(model.conf_int()[1]),
        "p": model.pvalues,
    })
    adj = adjusted_probability_gap(model, data)
    return or_df, adj


def run_reputation_robustness(llm, mat, outcome, eligible_col):
    rows = []
    for method in ["per_group_median", "pooled_median", "complete_case"]:
        data = build_model_data(llm, mat, outcome, eligible_col, method)
        if len(data) < 100:
            continue
        model = fit_logit(f"{outcome} ~ {MAIN_FORMULA_RHS}", data)
        rows.append({
            "Reputation method": method,
            "IsLLM OR": np.exp(model.params.get("IsLLM", np.nan)),
            "CI_lo": np.exp(model.conf_int().loc["IsLLM", 0]) if "IsLLM" in model.params.index else np.nan,
            "CI_hi": np.exp(model.conf_int().loc["IsLLM", 1]) if "IsLLM" in model.params.index else np.nan,
            "n": len(data),
        })
    return pd.DataFrame(rows)


def run_sensitivity_model(llm, mat, outcome, eligible_col):
    data = build_model_data(llm, mat, outcome, eligible_col, "per_group_median")
    post_vars = [v for v in ["Score", "ViewCount", "CommentCount"] if v in data.columns]
    if not post_vars:
        return None, None
    for v in post_vars:
        data[v] = data[v].fillna(0)
    formula = f"{outcome} ~ {MAIN_FORMULA_RHS} + " + " + ".join(post_vars)
    model = fit_logit(formula, data)
    or_df = pd.DataFrame({
        "OR": np.exp(model.params),
        "CI_lo": np.exp(model.conf_int()[0]),
        "CI_hi": np.exp(model.conf_int()[1]),
        "p": model.pvalues,
    })
    adj = adjusted_probability_gap(model, data)
    adj["note"] = "SENSITIVITY ONLY - includes post-treatment variables (see PART II.4.3)"
    return or_df, adj


def compute_monthly_rates(df, outcome, eligible_col):
    sub = df[df[eligible_col]].copy()
    sub["YearMonth"] = sub["CreationDate"].dt.to_period("M")
    grp = sub.groupby("YearMonth")[outcome].agg(["sum", "count"]).reset_index()
    grp.columns = ["YearMonth", "n_event", "n_total"]
    grp["rate"]  = grp["n_event"] / grp["n_total"]
    ci_lo, ci_hi = proportion_confint(grp["n_event"], grp["n_total"],
                                      alpha=0.05, method="wilson")
    grp["ci_lo"], grp["ci_hi"] = ci_lo, ci_hi
    return grp


def plot_monthly_rates(llm, mat, outcome, eligible_col, label, fname):
    gl = compute_monthly_rates(llm, outcome, eligible_col)
    gm = compute_monthly_rates(mat, outcome, eligible_col)

    merged = gl.merge(gm, on="YearMonth", suffixes=("_llm", "_mat"))
    merged["gap"] = merged["rate_llm"] - merged["rate_mat"]
    merged["ym"] = merged["YearMonth"].dt.to_timestamp()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})

    x_l, x_m = gl["YearMonth"].dt.to_timestamp(), gm["YearMonth"].dt.to_timestamp()
    ax1.plot(x_l, gl["rate"], color=PALETTE["LLM"], lw=2, label="LLM")
    ax1.fill_between(x_l, gl["ci_lo"], gl["ci_hi"], color=PALETTE["LLM"], alpha=.15)
    ax1.plot(x_m, gm["rate"], color=PALETTE["Mature"], lw=2, label="Mature")
    ax1.fill_between(x_m, gm["ci_lo"], gm["ci_hi"], color=PALETTE["Mature"], alpha=.15)
    ax1.set_ylabel(label)
    ax1.legend()
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    ax2.bar(merged["ym"], merged["gap"],
            color=[PALETTE["LLM"] if g < 0 else PALETTE["Mature"] for g in merged["gap"]],
            width=20, alpha=.8)
    ax2.axhline(0, color="black", lw=.8, ls="--")
    ax2.set_ylabel("Gap (LLM − Mature)")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax2.set_xlabel("Month")

    fig.suptitle(f"Monthly {label} — LLM vs Mature", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / fname}")
    return merged


def run_interaction_model(llm, mat, outcome, eligible_col):
    combined = pd.concat([llm, mat], ignore_index=True)
    combined = combined[combined[eligible_col] == True].copy()
    combined["IsLLM"] = (combined["group"] == "LLM").astype(int)

    month_ordinal = combined["CreationDate"].dt.to_period("M").apply(lambda p: p.ordinal)
    combined["TimeIndex"] = month_ordinal - month_ordinal.mean()   # centered

    if "BodyLength" not in combined.columns:
        combined["BodyLength"] = combined.get(
            "Body", pd.Series("", index=combined.index)).str.len()
    if "TitleLength" not in combined.columns:
        combined["TitleLength"] = combined.get(
            "Title", pd.Series("", index=combined.index)).str.len()
    for col in ["LogReputation", "ReputationMissing", "HasCodeBlock", "TagCount"]:
        if col not in combined.columns:
            combined[col] = 0

    combined = combined[combined[outcome].notna()].copy()
    formula = (f"{outcome} ~ IsLLM * TimeIndex + BodyLength + TitleLength + "
               f"LogReputation + ReputationMissing + HasCodeBlock + TagCount")
    try:
        model = fit_logit(formula, combined)
        term = "IsLLM:TimeIndex"
        if term not in model.params.index:
            return {}
        ci = model.conf_int()
        return {
            "coef": model.params[term],
            "OR": np.exp(model.params[term]),
            "CI_lo_OR": np.exp(ci.loc[term, 0]),
            "CI_hi_OR": np.exp(ci.loc[term, 1]),
            "p": model.pvalues[term],
        }
    except Exception as e:
        print(f"  Interaction model failed: {e}")
        return {}


def run_community_reception(llm, mat):
    combined = pd.concat([llm, mat], ignore_index=True)
    combined["IsLLM"] = (combined["group"] == "LLM").astype(int)
    rows, raw_tests = [], []

    if "ClosedDate" in combined.columns:
        combined["IsClosed"] = combined["ClosedDate"].notna().astype(int)
        lc = combined.loc[combined["IsLLM"] == 1, "IsClosed"]
        mc = combined.loc[combined["IsLLM"] == 0, "IsClosed"]
        ct = np.array([[lc.sum(), len(lc) - lc.sum()],
                       [mc.sum(), len(mc) - mc.sum()]]).astype(int)
        rd = lc.mean() - mc.mean()
        if ct.min() == 0:
            print("  WARNING: zero cell in closure-rate table; using Fisher exact.")
            _, p = stats.fisher_exact(ct)
            chi2, v = np.nan, np.nan
        else:
            chi2, p, dof, _ = chi2_contingency(ct, correction=False)
            v = cramers_v_2x2(ct)
        rows.append({"Metric": "Closure rate", "LLM": lc.mean(), "Mature": mc.mean(),
                     "RiskDiff": rd, "EffectSize": v, "ES_name": "Cramér's V"})
        raw_tests.append(("Closure rate", chi2, p, v))

    for col, label in [("Score", "Score"), ("CommentCount", "Comment count")]:
        if col not in combined.columns:
            continue
        l = combined.loc[combined["IsLLM"] == 1, col].dropna()
        m = combined.loc[combined["IsLLM"] == 0, col].dropna()
        u, p = mannwhitneyu(l, m, alternative="two-sided")
        cles, rbc = cles_and_rbc(u, len(l), len(m))
        rows.append({"Metric": f"{label} (median)", "LLM": l.median(), "Mature": m.median(),
                     "RiskDiff": l.median() - m.median(), "EffectSize": rbc,
                     "ES_name": "rank-biserial r"})
        raw_tests.append((label, u, p, rbc))

    if not raw_tests:
        return pd.DataFrame(), pd.DataFrame()

    p_values = [t[2] for t in raw_tests]
    reject, p_adj, _, _ = multipletests(p_values, method="holm")
    tests_df = pd.DataFrame([{
        "Test": t[0], "Statistic": t[1], "p_raw": t[2],
        "p_adj_Holm": pa, "Reject_H0": r, "EffectSize": t[3],
    } for t, pa, r in zip(raw_tests, p_adj, reject)])

    return pd.DataFrame(rows), tests_df


def plot_community_reception(llm, mat):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, label in zip(axes, ["Score", "CommentCount"],
                              ["Question Score", "Comment Count"]):
        for grp, df in [("LLM", llm), ("Mature", mat)]:
            if col not in df.columns:
                continue
            vals = df[col].dropna()
            vals = vals.clip(upper=np.percentile(vals, 99))
            ax.hist(vals, bins=30, alpha=.6, color=PALETTE[grp], label=grp, density=True)
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.legend()
    fig.suptitle("Community reception distributions", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_community_reception.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'fig_community_reception.png'}")


def run_robustness(llm, mat):
    notes = []
    alt_rates = {}

    llm_alt = build_windowed_outcomes(llm.copy(), infer_extraction_date(llm),
                                      answer_window_days=7, accept_window_days=30,
                                      suffix="_alt")
    mat_alt = build_windowed_outcomes(mat.copy(), infer_extraction_date(mat),
                                      answer_window_days=7, accept_window_days=30,
                                      suffix="_alt")
    for label, col, elig in [
        ("Answered within 7d",  "AnsweredWithin7Days_alt",  "EligibleAnswered7_alt"),
        ("Accepted within 30d", "AcceptedWithin30Days_alt", "EligibleAccepted30_alt"),
    ]:
        lr = llm_alt.loc[llm_alt[elig], col].mean()
        mr = mat_alt.loc[mat_alt[elig], col].mean()
        alt_rates[label] = {"LLM_rate": lr, "Mature_rate": mr, "RiskDiff": lr - mr}
    notes.append("Alternative windows (7d/30d) computed - compare RiskDiff to "
                "main 30d/90d windows to confirm conclusions don't hinge on window length.")

    detection_subsets = {}
    if "DetectionType" in llm.columns:
        for dtype in llm["DetectionType"].dropna().unique():
            sub = llm[llm["DetectionType"] == dtype]
            n_elig = sub["EligibleAnswered30"].sum() if "EligibleAnswered30" in sub.columns else 0
            rate = sub.loc[sub.get("EligibleAnswered30", False), "AnsweredWithin30Days"].mean() \
                if "AnsweredWithin30Days" in sub.columns else np.nan
            detection_subsets[str(dtype)] = {"n_eligible": int(n_elig), "answered_rate": rate}
        notes.append("Detection-type subsets (tag-only vs keyword-only) computed; "
                    "large divergence between them would flag a detection-quality limitation.")
    else:
        notes.append("No DetectionType column found; skipping detection-subset robustness.")

    return {"alt_windows": alt_rates, "detection_subsets": detection_subsets}, notes


def plot_adj_prob(adj_answer, adj_accept):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, adj, title in zip(axes, [adj_answer, adj_accept],
                              ["Answered (window)", "Accepted (window)"]):
        if adj is None:
            ax.set_visible(False)
            continue
        groups, vals = ["LLM", "Mature"], [adj["adj_prob_LLM"], adj["adj_prob_Mature"]]
        colors = [PALETTE["LLM"], PALETTE["Mature"]]
        bars = ax.bar(groups, vals, color=colors, width=.5)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.set_title(title)
        ax.set_ylabel("Adjusted predicted probability")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + .01,
                    f"{val:.1%}", ha="center", va="bottom", fontsize=10)
    fig.suptitle("Adjusted predicted probabilities (G-computation)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_adj_prob.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'fig_adj_prob.png'}")


def main():
    print("=" * 64)
    print("RQ2 ANALYSIS  (Parts I-III; survival analysis excluded)")
    print("=" * 64)

    print("\n[0] Loading data…")
    llm, mat = load_data()

    llm_extraction = LLM_EXTRACTION_DATE or infer_extraction_date(llm)
    mat_extraction = MAT_EXTRACTION_DATE or infer_extraction_date(mat)
    print(f"  LLM extraction date (inferred):    {llm_extraction.date()}")
    print(f"  Mature extraction date (inferred): {mat_extraction.date()}")

    print("\n[1] Building windowed outcomes (30d answered / 90d accepted)…")
    llm = build_windowed_outcomes(llm, llm_extraction)
    mat = build_windowed_outcomes(mat, mat_extraction)

    ANSWER_COL, ELIG_A = "AnsweredWithin30Days", "EligibleAnswered30"
    ACCEPT_COL, ELIG_C = "AcceptedWithin90Days", "EligibleAccepted90"

    print(f"  LLM  eligible(30d): {llm[ELIG_A].sum():,}  "
          f"answered rate: {llm.loc[llm[ELIG_A], ANSWER_COL].mean():.1%}")
    print(f"  Mat  eligible(30d): {mat[ELIG_A].sum():,}  "
          f"answered rate: {mat.loc[mat[ELIG_A], ANSWER_COL].mean():.1%}")
    print(f"  LLM  eligible(90d): {llm[ELIG_C].sum():,}  "
          f"accepted rate: {llm.loc[llm[ELIG_C], ACCEPT_COL].mean():.1%}")
    print(f"  Mat  eligible(90d): {mat[ELIG_C].sum():,}  "
          f"accepted rate: {mat.loc[mat[ELIG_C], ACCEPT_COL].mean():.1%}")

    print("\n[I] Descriptive comparison…")
    rates_df, tests_df = run_descriptives(llm, mat, ANSWER_COL, ACCEPT_COL)
    rates_df.to_csv(OUT_DIR / "table_descriptive_rates.csv", index=False)
    tests_df.to_csv(OUT_DIR / "table_descriptive_tests.csv", index=False)
    print(rates_df.to_string(index=False))
    print()
    print(tests_df.to_string(index=False))

    print("\n[II] Logistic regression (main model, HC1 SEs, month FE)…")
    or_answer, adj_answer = run_main_logistic(llm, mat, ANSWER_COL, ELIG_A)
    or_accept, adj_accept = run_main_logistic(llm, mat, ACCEPT_COL, ELIG_C)

    if or_answer is not None:
        or_answer.to_csv(OUT_DIR / "table_logistic_answered.csv")
        print(f"\n  Answered model - IsLLM OR: {or_answer.loc['IsLLM','OR']:.3f} "
              f"[{or_answer.loc['IsLLM','CI_lo']:.3f}-{or_answer.loc['IsLLM','CI_hi']:.3f}], "
              f"p={or_answer.loc['IsLLM','p']:.4g}")
        print(f"  Adjusted prob gap: {adj_answer['adj_prob_gap']:+.1%} "
              f"[{adj_answer['gap_CI_lo']:+.1%}, {adj_answer['gap_CI_hi']:+.1%}]  "
              f"(LLM={adj_answer['adj_prob_LLM']:.1%}, Mature={adj_answer['adj_prob_Mature']:.1%})")
    if or_accept is not None:
        or_accept.to_csv(OUT_DIR / "table_logistic_accepted.csv")
        print(f"\n  Accepted model - IsLLM OR: {or_accept.loc['IsLLM','OR']:.3f} "
              f"[{or_accept.loc['IsLLM','CI_lo']:.3f}-{or_accept.loc['IsLLM','CI_hi']:.3f}], "
              f"p={or_accept.loc['IsLLM','p']:.4g}")
        print(f"  Adjusted prob gap: {adj_accept['adj_prob_gap']:+.1%} "
              f"[{adj_accept['gap_CI_lo']:+.1%}, {adj_accept['gap_CI_hi']:+.1%}]")

    plot_adj_prob(adj_answer, adj_accept)

    print("\n[II.a] Missing-reputation handling robustness (3-way)…")
    rep_rob_answer = run_reputation_robustness(llm, mat, ANSWER_COL, ELIG_A)
    rep_rob_accept = run_reputation_robustness(llm, mat, ACCEPT_COL, ELIG_C)
    rep_rob_answer.to_csv(OUT_DIR / "table_reputation_robustness_answered.csv", index=False)
    rep_rob_accept.to_csv(OUT_DIR / "table_reputation_robustness_accepted.csv", index=False)
    print("  Answered:"); print(rep_rob_answer.to_string(index=False))
    print("  Accepted:"); print(rep_rob_accept.to_string(index=False))

    print("\n[II.b] Sensitivity model (post-treatment vars included; NOT the main result)…")
    sen_or, sen_adj = run_sensitivity_model(llm, mat, ANSWER_COL, ELIG_A)
    if sen_or is not None:
        sen_or.to_csv(OUT_DIR / "table_logistic_answered_sensitivity.csv")
        print(f"  Sensitivity adjusted prob gap: {sen_adj['adj_prob_gap']:+.1%} "
              f"[{sen_adj['gap_CI_lo']:+.1%}, {sen_adj['gap_CI_hi']:+.1%}]")
        if adj_answer is not None:
            print(f"  (main model gap was: {adj_answer['adj_prob_gap']:+.1%} "
                  f"[{adj_answer['gap_CI_lo']:+.1%}, {adj_answer['gap_CI_hi']:+.1%}])")
            print("  -> compare CI widths above: a much wider sensitivity CI is the "
                  "expected signature of post-treatment bias instability.")
    with open(OUT_DIR / "sensitivity_vs_main_gap.json", "w") as f:
        json.dump({"main": adj_answer, "sensitivity": sen_adj}, f, indent=2, default=str)

    print("\n[III] Monthly trends…")
    gap_answer = plot_monthly_rates(llm, mat, ANSWER_COL, ELIG_A,
                                    "Answered rate (30-day window)", "fig_monthly_answered.png")
    gap_accept = plot_monthly_rates(llm, mat, ACCEPT_COL, ELIG_C,
                                    "Accepted rate (90-day window)", "fig_monthly_accepted.png")
    gap_answer.to_csv(OUT_DIR / "data_gap_answered.csv", index=False)
    gap_accept.to_csv(OUT_DIR / "data_gap_accepted.csv", index=False)

    print("\n[III.a] Interaction model (IsLLM × centered time index)…")
    ia_answer = run_interaction_model(llm, mat, ANSWER_COL, ELIG_A)
    ia_accept = run_interaction_model(llm, mat, ACCEPT_COL, ELIG_C)
    for label, ia in [("Answered", ia_answer), ("Accepted", ia_accept)]:
        if ia:
            direction = "narrowing (catching up)" if ia.get("coef", 0) > 0 else "widening (falling behind)"
            sig = "significant" if ia.get("p", 1) < 0.05 else "not significant"
            print(f"  {label}: coef={ia.get('coef', np.nan):.5f}  "
                  f"OR={ia.get('OR', np.nan):.5f}  "
                  f"[{ia.get('CI_lo_OR', np.nan):.5f}-{ia.get('CI_hi_OR', np.nan):.5f}]  "
                  f"p={ia.get('p', np.nan):.4g} ({sig}) → gap is {direction}")
    with open(OUT_DIR / "interaction_model_results.json", "w") as f:
        json.dump({"answered": ia_answer, "accepted": ia_accept}, f, indent=2, default=str)

    print("\n[Community reception]…")
    recep_df, recep_tests = run_community_reception(llm, mat)
    if not recep_df.empty:
        recep_df.to_csv(OUT_DIR / "table_community_reception.csv", index=False)
        recep_tests.to_csv(OUT_DIR / "table_community_reception_tests.csv", index=False)
        print(recep_df.to_string(index=False))
        print(recep_tests.to_string(index=False))
    plot_community_reception(llm, mat)

    print("\n[Robustness]…")
    rob, notes = run_robustness(llm, mat)
    for n in notes:
        print(f"  {n}")
    with open(OUT_DIR / "robustness_results.json", "w") as f:
        json.dump(rob, f, indent=2, default=str)
    for n in notes:
        pass
    with open(OUT_DIR / "robustness_notes.json", "w") as f:
        json.dump({"notes": notes}, f, indent=2)

    print("\n" + "=" * 64)
    print(f"All outputs written to {OUT_DIR}/")
    print("=" * 64)


if __name__ == "__main__":
    main()