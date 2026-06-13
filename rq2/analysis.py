import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
import json

LLM_PATH   = Path("outputs/validation/llm-data-cleaned.csv")
MAT_PATH   = Path("outputs/validation/mature-data-cleaned.csv")   
OUT_DIR    = Path("outputs/rq2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {"LLM": "#E05A5A", "Mature": "#4A90D9"}
plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

EXTRACTION_DATE = None


def load_data():
    llm = pd.read_csv(LLM_PATH, low_memory=False)
    mat = pd.read_csv(MAT_PATH, low_memory=False)

    llm = llm.drop_duplicates(subset="QuestionId")
    mat = mat.drop_duplicates(subset="QuestionId")

    date_cols = ["CreationDate", "ClosedDate", "LastActivityDate"]
    for df in (llm, mat):
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    llm["group"] = "LLM"
    mat["group"] = "Mature"

    print(f"LLM rows after dedup:    {len(llm):,}")
    print(f"Mature rows after dedup: {len(mat):,}")
    return llm, mat


def build_windowed_outcomes(df, extraction_date):
    age_days = (extraction_date - df["CreationDate"]).dt.total_seconds() / 86_400

    has_ttfa = "TimeToFirstAnswerHours" in df.columns

    eligible_30 = age_days >= 30
    if has_ttfa:
        answered_in_30 = (df["TimeToFirstAnswerHours"].notna() &
                          (df["TimeToFirstAnswerHours"] <= 30 * 24))
    else:
        answered_in_30 = df.get("AnswerCount", pd.Series(0, index=df.index)) > 0
        print("  WARNING: TimeToFirstAnswerHours not found; using AnswerCount>0 as fallback")

    df["AnsweredWithin30Days"] = np.where(eligible_30, answered_in_30.astype(float), np.nan)
    df["Eligible30"]           = eligible_30

    # ── AcceptedWithin90Days ────────────────────────────────────────────────
    eligible_90 = age_days >= 90
    has_accept_time = "AcceptedAnswerCreationDate" in df.columns

    if has_accept_time:
        ttaccept = (pd.to_datetime(df["AcceptedAnswerCreationDate"], utc=True, errors="coerce")
                    - df["CreationDate"]).dt.total_seconds() / 3_600
        accepted_in_90 = ttaccept.notna() & (ttaccept <= 90 * 24)
    elif "AcceptedAnswerId" in df.columns:
        accepted_in_90 = df["AcceptedAnswerId"].notna()
        print("  WARNING: AcceptedAnswerCreationDate not found; using AcceptedAnswerId presence")
    else:
        accepted_in_90 = pd.Series(False, index=df.index)
        print("  WARNING: No acceptance timing column; AcceptedWithin90Days will be all-NaN")

    df["AcceptedWithin90Days"] = np.where(eligible_90, accepted_in_90.astype(float), np.nan)
    df["Eligible90"]           = eligible_90

    return df


def ci95_proportion(k, n):
    """Wilson score 95% CI."""
    if n == 0:
        return np.nan, np.nan
    z = 1.96
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return centre - margin, centre + margin


def cramers_v(chi2, n, dof):
    return np.sqrt(chi2 / (n * dof))


def rank_biserial(u, n1, n2):
    return 1 - (2*u) / (n1 * n2)


def run_descriptives(llm, mat):
    """Sections 3: rates, CIs, chi-square, MWU, Holm-Bonferroni."""
    results = []
    raw_tests = []

    def compare_binary(outcome, label):
        l = llm[outcome].dropna()
        m = mat[outcome].dropna()
        lk, ln = l.sum(), len(l)
        mk, mn = m.sum(), len(m)
        lrate, lcilo, lcihi = lk/ln, *ci95_proportion(lk, ln)
        mrate, mcilo, mcihi = mk/mn, *ci95_proportion(mk, mn)
        rd = lrate - mrate

        ct = np.array([[lk, ln-lk], [mk, mn-mk]])
        chi2, p, dof, _ = chi2_contingency(ct, correction=False)
        v = cramers_v(chi2, ln+mn, dof)

        results.append({
            "Outcome": label, "Group": "LLM",
            "n": ln, "Rate": lrate, "CI_lo": lcilo, "CI_hi": lcihi,
        })
        results.append({
            "Outcome": label, "Group": "Mature",
            "n": mn, "Rate": mrate, "CI_lo": mcilo, "CI_hi": mcihi,
        })
        results.append({
            "Outcome": label, "Group": "Diff (LLM-Mature)",
            "n": "", "Rate": rd, "CI_lo": "", "CI_hi": "",
        })
        raw_tests.append((label, chi2, p, v, "Cramér's V"))
        return chi2, p, v

    def compare_continuous(col, label):
        l = llm[col].dropna()
        m = mat[col].dropna()
        u, p = mannwhitneyu(l, m, alternative="two-sided")
        rb = rank_biserial(u, len(l), len(m))
        results.append({
            "Outcome": label, "Group": "LLM",
            "n": len(l), "Rate": l.median(), "CI_lo": np.percentile(l, 25), "CI_hi": np.percentile(l, 75),
        })
        results.append({
            "Outcome": label, "Group": "Mature",
            "n": len(m), "Rate": m.median(), "CI_lo": np.percentile(m, 25), "CI_hi": np.percentile(m, 75),
        })
        raw_tests.append((label, u, p, rb, "rank-biserial r"))
        return u, p, rb

    compare_binary("AnsweredWithin30Days", "Answered within 30 days")
    compare_binary("AcceptedWithin90Days", "Accepted within 90 days")
    if "TimeToFirstAnswerHours" in llm.columns:
        compare_continuous("TimeToFirstAnswerHours", "Time to first answer (h, answered only)")
    if "AnswerCount" in llm.columns:
        compare_continuous("AnswerCount", "Answer count")

    p_values = [t[2] for t in raw_tests]
    reject, p_adj, _, _ = multipletests(p_values, method="holm")

    tests_df = pd.DataFrame([{
        "Test": t[0], "Statistic": t[1], "p_raw": t[2],
        "p_adj_Holm": pa, "Reject_H0": r,
        "EffectSize": t[3], "ES_name": t[4],
    } for t, pa, r in zip(raw_tests, p_adj, reject)])

    rates_df = pd.DataFrame(results)
    return rates_df, tests_df


def build_model_data(llm, mat, outcome):
    """Combine, keep eligible rows, build feature matrix."""
    combined = pd.concat([llm, mat], ignore_index=True)
    combined = combined[combined[f"Eligible{'30' if '30' in outcome else '90'}"] == True].copy()

    combined["IsLLM"] = (combined["group"] == "LLM").astype(int)

    if "BodyLength" not in combined.columns:
        combined["BodyLength"] = combined.get("Body", pd.Series("", index=combined.index)).str.len()
    if "TitleLength" not in combined.columns:
        combined["TitleLength"] = combined.get("Title", pd.Series("", index=combined.index)).str.len()
    if "OwnerReputation" in combined.columns:
        combined["LogReputation"] = np.log1p(combined["OwnerReputation"].fillna(1))
    elif "OwnerUserReputation" in combined.columns:
        combined["LogReputation"] = np.log1p(combined["OwnerUserReputation"].fillna(1))
    else:
        combined["LogReputation"] = 0.0

    combined["HasCodeBlock"] = combined.get("Body", pd.Series("", index=combined.index)).str.contains(
        r"<code>|```", regex=True, na=False).astype(int)
    combined["TagCount"] = combined.get("Tags", pd.Series("", index=combined.index)).str.count(r"<").fillna(0)
    combined["Month"] = combined["CreationDate"].dt.to_period("M").astype(str)

    combined = combined[combined[outcome].notna()].copy()
    return combined


def run_logistic(llm, mat, outcome="AnsweredWithin30Days"):
    data = build_model_data(llm, mat, outcome)
    if len(data) < 100:
        print(f"  Too few rows ({len(data)}) for logistic regression on {outcome}. Skipping.")
        return None, None

    formula = (f"{outcome} ~ IsLLM + BodyLength + TitleLength + "
               f"LogReputation + HasCodeBlock + TagCount + C(Month)")

    try:
        model = smf.logit(formula, data=data).fit(
            cov_type="HC3", disp=False, maxiter=200)
    except np.linalg.LinAlgError:
        print("  WARNING: Singular matrix with month FE; refitting without month dummies.")
        formula = (f"{outcome} ~ IsLLM + BodyLength + TitleLength + "
                   f"LogReputation + HasCodeBlock + TagCount")
        try:
            model = smf.logit(formula, data=data).fit(
                cov_type="HC3", disp=False, maxiter=200)
        except Exception as e:
            print(f"  Logistic regression failed: {e}")
            return None, None
    except Exception as e:
        print(f"  Logistic regression failed: {e}")
        return None, None

    or_df = pd.DataFrame({
        "OR":    np.exp(model.params),
        "CI_lo": np.exp(model.conf_int()[0]),
        "CI_hi": np.exp(model.conf_int()[1]),
        "p":     model.pvalues,
    })

    pred_llm  = model.predict(data.assign(IsLLM=1)).mean()
    pred_mat  = model.predict(data.assign(IsLLM=0)).mean()

    return or_df, {"adj_prob_LLM": pred_llm, "adj_prob_Mature": pred_mat,
                   "adj_prob_diff": pred_llm - pred_mat, "n": len(data)}


def run_sensitivity(llm, mat, outcome="AnsweredWithin30Days"):
    """Post-treatment variables included (score, views, comments) – sensitivity only."""
    data = build_model_data(llm, mat, outcome)
    post_vars = []
    for v in ["Score", "ViewCount", "CommentCount"]:
        if v in data.columns:
            data[v] = data[v].fillna(0)
            post_vars.append(v)
    if not post_vars:
        return None, None
    formula = (f"{outcome} ~ IsLLM + BodyLength + TitleLength + "
               f"LogReputation + HasCodeBlock + TagCount + "
               + " + ".join(post_vars) + " + C(Month)")
    try:
        model = smf.logit(formula, data=data).fit(cov_type="HC3", disp=False, maxiter=200)
    except Exception as e:
        print(f"  Sensitivity model failed: {e}")
        return None, None
    or_df = pd.DataFrame({
        "OR":    np.exp(model.params),
        "CI_lo": np.exp(model.conf_int()[0]),
        "CI_hi": np.exp(model.conf_int()[1]),
        "p":     model.pvalues,
    })
    return or_df, {"note": "sensitivity – post-treatment variables included"}


def compute_monthly_rates(df, outcome, eligible_col):
    sub = df[df[eligible_col]].copy()
    sub["YearMonth"] = sub["CreationDate"].dt.to_period("M")
    grp = sub.groupby("YearMonth")[outcome].agg(["sum","count"]).reset_index()
    grp.columns = ["YearMonth", "answered", "total"]
    grp["rate"]  = grp["answered"] / grp["total"]
    grp["ci_lo"] = grp.apply(lambda r: ci95_proportion(r.answered, r.total)[0], axis=1)
    grp["ci_hi"] = grp.apply(lambda r: ci95_proportion(r.answered, r.total)[1], axis=1)
    return grp


def plot_monthly_rates(llm, mat, outcome, eligible_col, label, fname):
    gl = compute_monthly_rates(llm, outcome, eligible_col)
    gm = compute_monthly_rates(mat, outcome, eligible_col)

    merged = gl.merge(gm, on="YearMonth", suffixes=("_llm","_mat"))
    merged["gap"] = merged["rate_llm"] - merged["rate_mat"]
    merged["ym"]  = merged["YearMonth"].dt.to_timestamp()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                   gridspec_kw={"height_ratios":[3,1]})

    x_l = gl["YearMonth"].dt.to_timestamp()
    x_m = gm["YearMonth"].dt.to_timestamp()
    ax1.plot(x_l, gl["rate"], color=PALETTE["LLM"],    lw=2, label="LLM")
    ax1.fill_between(x_l, gl["ci_lo"], gl["ci_hi"],    color=PALETTE["LLM"],    alpha=.15)
    ax1.plot(x_m, gm["rate"], color=PALETTE["Mature"], lw=2, label="Mature")
    ax1.fill_between(x_m, gm["ci_lo"], gm["ci_hi"],    color=PALETTE["Mature"], alpha=.15)
    ax1.set_ylabel(label)
    ax1.legend()
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    # bottom: gap
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
    combined["MonthIndex"] = (combined["CreationDate"].dt.to_period("M")
                              .apply(lambda p: p.ordinal))
    combined["MonthIndex"] -= combined["MonthIndex"].min()

    if "BodyLength" not in combined.columns:
        combined["BodyLength"] = combined.get("Body", pd.Series("", index=combined.index)).str.len()
    if "TitleLength" not in combined.columns:
        combined["TitleLength"] = combined.get("Title", pd.Series("", index=combined.index)).str.len()
    for col in ["LogReputation","HasCodeBlock","TagCount"]:
        if col not in combined.columns:
            combined[col] = 0

    combined = combined[combined[outcome].notna()].copy()
    formula = (f"{outcome} ~ IsLLM * MonthIndex + "
               f"BodyLength + TitleLength + LogReputation + HasCodeBlock + TagCount")
    try:
        model = smf.logit(formula, data=combined).fit(cov_type="HC3", disp=False, maxiter=200)
        interaction = {
            "coef": model.params.get("IsLLM:MonthIndex", np.nan),
            "OR":   np.exp(model.params.get("IsLLM:MonthIndex", np.nan)),
            "p":    model.pvalues.get("IsLLM:MonthIndex", np.nan),
            "CI_lo_OR": np.exp(model.conf_int().loc["IsLLM:MonthIndex", 0]) if "IsLLM:MonthIndex" in model.conf_int().index else np.nan,
            "CI_hi_OR": np.exp(model.conf_int().loc["IsLLM:MonthIndex", 1]) if "IsLLM:MonthIndex" in model.conf_int().index else np.nan,
        }
    except Exception as e:
        print(f"  Interaction model failed: {e}")
        interaction = {}
    return interaction



def run_community_reception(llm, mat):
    combined = pd.concat([llm, mat], ignore_index=True)
    combined["IsLLM"] = (combined["group"] == "LLM").astype(int)
    results = []
    raw_tests = []

    if "ClosedDate" in combined.columns:
        combined["IsClosed"] = combined["ClosedDate"].notna().astype(int)
        lc = combined[combined["IsLLM"]==1]["IsClosed"]
        mc = combined[combined["IsLLM"]==0]["IsClosed"]
        ct = np.array([[lc.sum(), len(lc)-lc.sum()],
                       [mc.sum(), len(mc)-mc.sum()]])
        rd = lc.mean() - mc.mean()
        if ct.min() == 0:
            print("  WARNING: Zero cell in closure-rate table; Fisher exact used.")
            _, p = stats.fisher_exact(ct)
            chi2, v = np.nan, np.nan
        else:
            chi2, p, dof, _ = chi2_contingency(ct, correction=False)
            v = cramers_v(chi2, len(combined), dof)
        results.append({"Metric":"Closure rate","LLM":lc.mean(),"Mature":mc.mean(),
                        "RiskDiff":rd,"EffectSize":v,"ES_name":"Cramér's V"})
        raw_tests.append(("Closure rate", chi2, p, v))

    if "Score" in combined.columns:
        l_s = combined[combined["IsLLM"]==1]["Score"].dropna()
        m_s = combined[combined["IsLLM"]==0]["Score"].dropna()
        u, p = mannwhitneyu(l_s, m_s, alternative="two-sided")
        rb = rank_biserial(u, len(l_s), len(m_s))
        results.append({"Metric":"Score (median)","LLM":l_s.median(),"Mature":m_s.median(),
                        "RiskDiff":l_s.median()-m_s.median(),"EffectSize":rb,"ES_name":"rank-biserial r"})
        raw_tests.append(("Score", u, p, rb))

    if "CommentCount" in combined.columns:
        l_c = combined[combined["IsLLM"]==1]["CommentCount"].dropna()
        m_c = combined[combined["IsLLM"]==0]["CommentCount"].dropna()
        u, p = mannwhitneyu(l_c, m_c, alternative="two-sided")
        rb = rank_biserial(u, len(l_c), len(m_c))
        results.append({"Metric":"Comment count (median)","LLM":l_c.median(),"Mature":m_c.median(),
                        "RiskDiff":l_c.median()-m_c.median(),"EffectSize":rb,"ES_name":"rank-biserial r"})
        raw_tests.append(("CommentCount", u, p, rb))

    if not raw_tests:
        return pd.DataFrame(), pd.DataFrame()

    p_values = [t[2] for t in raw_tests]
    reject, p_adj, _, _ = multipletests(p_values, method="holm")
    tests_df = pd.DataFrame([{
        "Test": t[0], "Statistic": t[1], "p_raw": t[2],
        "p_adj_Holm": pa, "Reject_H0": r, "EffectSize": t[3],
    } for t, pa, r in zip(raw_tests, p_adj, reject)])

    return pd.DataFrame(results), tests_df


def run_robustness(llm, mat):
    """Alternative windows 7/30 and detection-type subsets if columns exist."""
    notes = []

    def alt_window(df, extraction_date):
        age_days = (extraction_date - df["CreationDate"]).dt.total_seconds() / 86_400
        if "TimeToFirstAnswerHours" in df.columns:
            a7 = (df["TimeToFirstAnswerHours"].notna() & (df["TimeToFirstAnswerHours"] <= 7*24))
        else:
            a7 = df.get("AnswerCount", pd.Series(0, index=df.index)) > 0
        df["AnsweredWithin7Days"]  = np.where(age_days >= 7,  a7.astype(float), np.nan)
        df["Eligible7"] = age_days >= 7

        if "AcceptedAnswerCreationDate" in df.columns:
            ttacc = (pd.to_datetime(df["AcceptedAnswerCreationDate"], utc=True, errors="coerce")
                     - df["CreationDate"]).dt.total_seconds() / 3_600
            a30 = ttacc.notna() & (ttacc <= 30*24)
        elif "AcceptedAnswerId" in df.columns:
            a30 = df["AcceptedAnswerId"].notna()
        else:
            a30 = pd.Series(False, index=df.index)
        df["AcceptedWithin30Days"] = np.where(age_days >= 30, a30.astype(float), np.nan)
        df["Eligible30b"] = age_days >= 30
        return df

    robustness = {}

    if "DetectionType" in llm.columns:
        for dtype in llm["DetectionType"].unique():
            sub = llm[llm["DetectionType"]==dtype]
            n = sub["Eligible30"].sum() if "Eligible30" in sub.columns else 0
            robustness[f"LLM_subset_{dtype}_n"] = int(n)
        notes.append("Detection-type subsets computed; see robustness dict.")
    else:
        notes.append("No DetectionType column; skipping detection-subset robustness.")

    return robustness, notes


def plot_adj_prob(adj30, adj90):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, adj, title in zip(axes, [adj30, adj90],
                              ["Answered within 30d", "Accepted within 90d"]):
        if adj is None:
            ax.set_visible(False)
            continue
        groups = ["LLM", "Mature"]
        vals   = [adj["adj_prob_LLM"], adj["adj_prob_Mature"]]
        colors = [PALETTE["LLM"], PALETTE["Mature"]]
        bars = ax.bar(groups, vals, color=colors, width=.5)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.set_title(title)
        ax.set_ylabel("Adjusted predicted probability")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + .01,
                    f"{val:.1%}", ha="center", va="bottom", fontsize=10)
    fig.suptitle("Logistic regression: adjusted predicted probabilities", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_adj_prob.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'fig_adj_prob.png'}")


def plot_community_reception(llm, mat):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, label in zip(axes, ["Score","CommentCount"], ["Question Score","Comment Count"]):
        for grp, df in [("LLM", llm), ("Mature", mat)]:
            if col not in df.columns: continue
            vals = df[col].dropna().clip(upper=np.percentile(df[col].dropna(), 99))
            ax.hist(vals, bins=30, alpha=.6, color=PALETTE[grp], label=grp, density=True)
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.legend()
    fig.suptitle("Community reception distributions", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_community_reception.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'fig_community_reception.png'}")


def main():
    print("=" * 60)
    print("RQ2 ANALYSIS")
    print("=" * 60)

    print("\n[0] Loading data…")
    llm, mat = load_data()

    global EXTRACTION_DATE
    if EXTRACTION_DATE is None:
        max_date = max(llm["CreationDate"].max(), mat["CreationDate"].max())
        EXTRACTION_DATE = max_date + pd.Timedelta(days=1)
        print(f"  Inferred extraction date: {EXTRACTION_DATE.date()}")

    print("\n[2] Building windowed outcomes…")
    llm = build_windowed_outcomes(llm, EXTRACTION_DATE)
    mat = build_windowed_outcomes(mat, EXTRACTION_DATE)

    print(f"  LLM  eligible 30d: {llm['Eligible30'].sum():,}  "
          f"answered rate: {llm['AnsweredWithin30Days'].mean():.1%}")
    print(f"  Mat  eligible 30d: {mat['Eligible30'].sum():,}  "
          f"answered rate: {mat['AnsweredWithin30Days'].mean():.1%}")
    print(f"  LLM  eligible 90d: {llm['Eligible90'].sum():,}  "
          f"accepted rate: {llm['AcceptedWithin90Days'].mean():.1%}")
    print(f"  Mat  eligible 90d: {mat['Eligible90'].sum():,}  "
          f"accepted rate: {mat['AcceptedWithin90Days'].mean():.1%}")

    print("\n[3] Descriptive comparison…")
    rates_df, tests_df = run_descriptives(llm, mat)
    rates_df.to_csv(OUT_DIR / "table_descriptive_rates.csv", index=False)
    tests_df.to_csv(OUT_DIR / "table_descriptive_tests.csv", index=False)
    print(rates_df.to_string(index=False))
    print()
    print(tests_df.to_string(index=False))

    print("\n[5] Logistic regression…")
    or30, adj30 = run_logistic(llm, mat, "AnsweredWithin30Days")
    or90, adj90 = run_logistic(llm, mat, "AcceptedWithin90Days")
    sen30, _ = run_sensitivity(llm, mat, "AnsweredWithin30Days")

    if or30 is not None:
        or30.to_csv(OUT_DIR / "table_logistic_30d.csv")
        print(f"\n  30d model – IsLLM OR: {or30.loc['IsLLM','OR']:.3f} "
              f"[{or30.loc['IsLLM','CI_lo']:.3f}–{or30.loc['IsLLM','CI_hi']:.3f}], "
              f"p={or30.loc['IsLLM','p']:.4f}")
        print(f"  Adj. predicted prob: LLM={adj30['adj_prob_LLM']:.1%}  "
              f"Mature={adj30['adj_prob_Mature']:.1%}  "
              f"Diff={adj30['adj_prob_diff']:+.1%}")
    if or90 is not None:
        or90.to_csv(OUT_DIR / "table_logistic_90d.csv")
        print(f"\n  90d model – IsLLM OR: {or90.loc['IsLLM','OR']:.3f} "
              f"[{or90.loc['IsLLM','CI_lo']:.3f}–{or90.loc['IsLLM','CI_hi']:.3f}], "
              f"p={or90.loc['IsLLM','p']:.4f}")
    if sen30 is not None:
        sen30.to_csv(OUT_DIR / "table_logistic_30d_sensitivity.csv")

    plot_adj_prob(adj30, adj90)

    print("\n[6] Monthly trends…")
    gap30 = plot_monthly_rates(llm, mat, "AnsweredWithin30Days", "Eligible30",
                               "Answered rate (30-day window)", "fig_monthly_answered.png")
    gap90 = plot_monthly_rates(llm, mat, "AcceptedWithin90Days", "Eligible90",
                               "Accepted rate (90-day window)", "fig_monthly_accepted.png")
    gap30.to_csv(OUT_DIR / "data_gap_30d.csv", index=False)
    gap90.to_csv(OUT_DIR / "data_gap_90d.csv", index=False)

    print("\n  Interaction model (IsLLM × month):")
    ia30 = run_interaction_model(llm, mat, "AnsweredWithin30Days", "Eligible30")
    ia90 = run_interaction_model(llm, mat, "AcceptedWithin90Days", "Eligible90")
    for label, ia in [("30d", ia30), ("90d", ia90)]:
        if ia:
            direction = "narrowing ✓" if ia.get("coef",0) > 0 else "widening ✗"
            print(f"  {label}: coef={ia.get('coef',np.nan):.4f}  "
                  f"OR={ia.get('OR',np.nan):.4f}  p={ia.get('p',np.nan):.4f}  → {direction}")
    with open(OUT_DIR / "interaction_model_results.json", "w") as f:
        json.dump({"30d": ia30, "90d": ia90}, f, indent=2, default=str)

    print("\n[7] Community reception…")
    recep_df, recep_tests = run_community_reception(llm, mat)
    if not recep_df.empty:
        recep_df.to_csv(OUT_DIR / "table_community_reception.csv", index=False)
        recep_tests.to_csv(OUT_DIR / "table_community_reception_tests.csv", index=False)
        print(recep_df.to_string(index=False))
        print(recep_tests.to_string(index=False))
    plot_community_reception(llm, mat)

    print("\n[8] Robustness checks…")
    rob, notes = run_robustness(llm, mat)
    for n in notes:
        print(f"  {n}")
    with open(OUT_DIR / "robustness_notes.json", "w") as f:
        json.dump({"robustness": rob, "notes": notes}, f, indent=2)

    print("\n" + "=" * 60)
    print(f"All outputs written to {OUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()