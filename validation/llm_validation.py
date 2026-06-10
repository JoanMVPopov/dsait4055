"""
Manual validation of the LLM-question detection rules.

Workflow (three stages, run at different points in time):

  1. python llm_validation.py sample
       Draws a stratified sample from llm-data-with-rules.csv (produced
       by llm_detection.py) and writes two identical annotation files,
       one per annotator. Annotators fill the IsLLMRelated column with
       1 (LLM-related) or 0 (not), independently, without discussing.

  2. python llm_validation.py agreement
       After both files are filled in: computes Cohen's kappa and writes
       a disagreements file. The third team member fills FinalLabel for
       the disagreeing rows only.

  3. python llm_validation.py analyze
       Computes per-rule precision with Wilson 95% CIs, applies the
       precision threshold to the full dataset, and writes the cleaned
       dataset plus the rule decision table for the paper.

Annotators see only QuestionId, a stackoverflow.com link, the title and
a plain-text body excerpt. They do NOT see which rules matched -- this
avoids anchoring their judgment on the detection mechanism.
"""

import os
import sys

import numpy as np
import pandas as pd

from bs4 import BeautifulSoup

# ============================================================
# SECTION 1: CONFIGURATION
# ============================================================

RANDOM_SEED = 42

TOTAL_SAMPLE_SIZE = 250      # total questions to annotate
MIN_PER_STRATUM = 5          # floor per stratum (if available)
BODY_EXCERPT_CHARS = 1500    # how much body text annotators see

PRECISION_THRESHOLD = 0.70   # rules below this are dropped
MIN_LABELS_PER_RULE = 5      # rules with fewer labels => decided manually

ANNOTATION_DEFINITION = (
    "A question is LLM-related if it concerns developing with, using, "
    "deploying, or troubleshooting large language models or LLM-based "
    "tools/APIs (e.g. ChatGPT, GPT-x, Claude, Gemini, LangChain, RAG "
    "pipelines, embeddings for LLMs). Questions about classical ML, "
    "reinforcement learning (e.g. OpenAI Gym), rule-based chatbots, or "
    "unrelated uses of matching words (Gemini the zodiac/exchange, "
    "people named Claude) are NOT LLM-related."
)


# ============================================================
# SECTION 2: STAGE "sample" -- STRATIFIED SAMPLING + ANNOTATION FILES
# ============================================================

def assign_stratum(df):
    """
    Assign each question to exactly ONE stratum: the RAREST of its
    matched rules (fewest total matches in the dataset).

    Why rarest: strata must be disjoint for the sampling to be auditable,
    and assigning by rarest rule guarantees low-frequency rules still get
    represented instead of being absorbed by 'chatgpt'/'openai'.
    """
    df = df.copy()

    rule_counts = (
        df["MatchedRules"].str.split("|").explode().value_counts()
    )

    def rarest_rule(rules_str):
        rules = [r for r in rules_str.split("|") if r]
        if not rules:
            return "NO_RULE"
        return min(rules, key=lambda r: rule_counts.get(r, 0))

    df["Stratum"] = df["MatchedRules"].apply(rarest_rule)
    return df


def allocate_sample_sizes(stratum_sizes, total_budget, min_per_stratum):
    """
    Proportional allocation with a floor. Strata smaller than the floor
    contribute everything they have.
    """
    allocation = {}
    remaining_budget = total_budget

    # First pass: floors (or full stratum if tiny)
    for stratum, size in stratum_sizes.items():
        floor = min(min_per_stratum, size)
        allocation[stratum] = floor
        remaining_budget -= floor

    if remaining_budget <= 0:
        return allocation

    # Second pass: distribute the rest proportionally to remaining capacity
    capacity = {
        s: stratum_sizes[s] - allocation[s] for s in stratum_sizes
    }
    total_capacity = sum(capacity.values())

    if total_capacity == 0:
        return allocation

    for stratum in allocation:
        extra = int(round(remaining_budget * capacity[stratum] / total_capacity))
        allocation[stratum] = min(
            stratum_sizes[stratum],
            allocation[stratum] + extra
        )

    return allocation


def strip_html(text):
    if pd.isna(text):
        return ""
    return BeautifulSoup(str(text), "html.parser").get_text(" ")


def build_annotation_sample(df):
    df = assign_stratum(df)

    stratum_sizes = df["Stratum"].value_counts().to_dict()
    allocation = allocate_sample_sizes(
        stratum_sizes, TOTAL_SAMPLE_SIZE, MIN_PER_STRATUM
    )

    sampled_parts = []
    for stratum, n in allocation.items():
        if n == 0:
            continue
        stratum_df = df[df["Stratum"] == stratum]
        sampled_parts.append(
            stratum_df.sample(n=min(n, len(stratum_df)),
                              random_state=RANDOM_SEED)
        )

    sample = pd.concat(sampled_parts, ignore_index=True)

    # Shuffle so annotators don't see questions grouped by stratum
    sample = sample.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    return sample


def write_annotation_files(sample, output_path):
    annotator_view = pd.DataFrame({
        "QuestionId": sample["QuestionId"],
        "Link": "https://stackoverflow.com/q/" + sample["QuestionId"].astype(str),
        "Title": sample["Title"],
        "BodyExcerpt": sample["Body"].apply(strip_html).str[:BODY_EXCERPT_CHARS],
        "IsLLMRelated": "",   # annotator fills 1 or 0
        "Notes": "",
    })

    for annotator in ["A", "B"]:
        filename = os.path.join(output_path, f"annotation-{annotator}.csv")
        annotator_view.to_csv(filename, index=False)
        print(f"Wrote {filename}")

    # Key file: rules and strata, kept SEPARATE from what annotators see
    key = sample[["QuestionId", "MatchedRules", "Stratum"]]
    key_filename = os.path.join(output_path, "annotation-key.csv")
    key.to_csv(key_filename, index=False)
    print(f"Wrote {key_filename} (do not share with annotators)")

    definition_filename = os.path.join(output_path, "annotation-definition.txt")
    with open(definition_filename, "w") as f:
        f.write(ANNOTATION_DEFINITION + "\n")
    print(f"Wrote {definition_filename} (give this to both annotators)")


# ============================================================
# SECTION 3: STAGE "agreement" -- COHEN'S KAPPA + DISAGREEMENTS
# ============================================================

def load_annotations(output_path, annotator):
    filename = os.path.join(output_path, f"annotation-{annotator}.csv")
    df = pd.read_csv(filename)

    labels = pd.to_numeric(df["IsLLMRelated"], errors="coerce")
    invalid = labels.isna() | ~labels.isin([0, 1])

    if invalid.any():
        bad_ids = df.loc[invalid, "QuestionId"].tolist()
        raise ValueError(
            f"annotation-{annotator}.csv has {invalid.sum()} rows without a "
            f"valid 0/1 label. QuestionIds: {bad_ids[:10]}..."
        )

    df["IsLLMRelated"] = labels.astype(int)
    return df[["QuestionId", "IsLLMRelated"]]


def cohens_kappa(labels_a, labels_b):
    """Cohen's kappa for two binary label vectors (manual, no sklearn)."""
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)

    n = len(labels_a)
    observed_agreement = np.mean(labels_a == labels_b)

    p_a1 = labels_a.mean()
    p_b1 = labels_b.mean()
    expected_agreement = p_a1 * p_b1 + (1 - p_a1) * (1 - p_b1)

    if expected_agreement == 1.0:
        return 1.0

    return (observed_agreement - expected_agreement) / (1 - expected_agreement)


def run_agreement_stage(output_path):
    a = load_annotations(output_path, "A").rename(
        columns={"IsLLMRelated": "LabelA"})
    b = load_annotations(output_path, "B").rename(
        columns={"IsLLMRelated": "LabelB"})

    merged = a.merge(b, on="QuestionId", validate="one_to_one")

    kappa = cohens_kappa(merged["LabelA"], merged["LabelB"])
    agreement_rate = (merged["LabelA"] == merged["LabelB"]).mean()

    print(f"Labeled questions: {len(merged)}")
    print(f"Raw agreement:     {agreement_rate:.1%}")
    print(f"Cohen's kappa:     {kappa:.3f}")

    disagreements = merged[merged["LabelA"] != merged["LabelB"]].copy()
    disagreements["FinalLabel"] = ""   # third team member fills this

    filename = os.path.join(output_path, "disagreements.csv")
    disagreements.to_csv(filename, index=False)
    print(f"\nWrote {len(disagreements)} disagreements to {filename}")
    print("Have the third team member fill FinalLabel (0/1), then run "
          "the 'analyze' stage.")

    merged.to_csv(
        os.path.join(output_path, "merged-annotations.csv"), index=False
    )

    return kappa


# ============================================================
# SECTION 4: STAGE "analyze" -- PER-RULE PRECISION + FILTERING
# ============================================================

def wilson_interval(successes, n, z=1.96):
    """Wilson 95% CI for a proportion. Returns (low, high)."""
    if n == 0:
        return (np.nan, np.nan)

    p = successes / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = (z / denominator) * np.sqrt(
        p * (1 - p) / n + z**2 / (4 * n**2)
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def build_final_labels(output_path):
    """Merge both annotators' labels; resolve disagreements from the
    third member's FinalLabel column."""
    merged = pd.read_csv(
        os.path.join(output_path, "merged-annotations.csv")
    )

    merged["FinalLabel"] = np.where(
        merged["LabelA"] == merged["LabelB"], merged["LabelA"], np.nan
    )

    disagreements_file = os.path.join(output_path, "disagreements.csv")
    if os.path.exists(disagreements_file):
        resolved = pd.read_csv(disagreements_file)
        resolved["FinalLabel"] = pd.to_numeric(
            resolved["FinalLabel"], errors="coerce"
        )

        unresolved = resolved["FinalLabel"].isna().sum()
        if unresolved > 0:
            raise ValueError(
                f"{unresolved} disagreements in {disagreements_file} still "
                f"have no FinalLabel. Resolve them before analyzing."
            )

        resolution_map = dict(
            zip(resolved["QuestionId"], resolved["FinalLabel"])
        )
        merged["FinalLabel"] = merged.apply(
            lambda row: resolution_map.get(row["QuestionId"], row["FinalLabel"]),
            axis=1
        )

    merged["FinalLabel"] = merged["FinalLabel"].astype(int)
    return merged[["QuestionId", "FinalLabel"]]


def per_rule_precision(labels, key):
    """
    Precision per rule, computed over every LABELED question that the
    rule matched (a labeled question contributes to every rule it
    matches, not only its sampling stratum).

    Caveat for the paper: the sample is stratified, not simple-random,
    so per-rule estimates are valid but the pooled average is not a
    dataset-level precision estimate.
    """
    labeled = labels.merge(key, on="QuestionId", validate="one_to_one")

    exploded = (
        labeled
        .assign(Rule=labeled["MatchedRules"].str.split("|"))
        .explode("Rule")
    )

    rows = []
    for rule, group in exploded.groupby("Rule"):
        n = len(group)
        positives = int(group["FinalLabel"].sum())
        low, high = wilson_interval(positives, n)
        rows.append({
            "Rule": rule,
            "LabeledQuestions": n,
            "TruePositives": positives,
            "Precision": positives / n,
            "WilsonCI95Low": low,
            "WilsonCI95High": high,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("Precision", ascending=False)
        .reset_index(drop=True)
    )


def decide_rules(precision_table):
    """Apply the threshold. Rules with too few labels are flagged for a
    manual decision instead of being silently kept or dropped."""
    table = precision_table.copy()

    def decision(row):
        if row["LabeledQuestions"] < MIN_LABELS_PER_RULE:
            return "MANUAL_REVIEW"
        if row["Precision"] >= PRECISION_THRESHOLD:
            return "KEEP"
        return "DROP"

    table["Decision"] = table.apply(decision, axis=1)
    return table


def filter_dataset(df, rule_decisions):
    """
    Keep a question iff at least one of its matched rules is KEEP.
    MANUAL_REVIEW rules do not count as KEEP on their own -- questions
    matched ONLY by such rules are flagged separately for inspection.
    """
    kept_rules = set(
        rule_decisions.loc[rule_decisions["Decision"] == "KEEP", "Rule"]
    )
    review_rules = set(
        rule_decisions.loc[rule_decisions["Decision"] == "MANUAL_REVIEW", "Rule"]
    )

    def classify(rules_str):
        rules = set(r for r in rules_str.split("|") if r)
        if rules & kept_rules:
            return "KEEP"
        if rules & review_rules:
            return "REVIEW"
        return "DROP"

    df = df.copy()
    df["FilterDecision"] = df["MatchedRules"].apply(classify)
    return df


def sanity_check_pre_chatgpt(cleaned_df):
    """LLM volume before ChatGPT's launch (Nov 2022) should be small."""
    dates = pd.to_datetime(cleaned_df["CreationDate"], errors="coerce")
    pre = cleaned_df[dates < "2022-11-01"]

    print(f"\nSanity check: {len(pre)} cleaned questions predate Nov 2022 "
          f"({len(pre) / max(len(cleaned_df), 1):.1%} of cleaned set).")
    if len(pre) > 0:
        monthly = (
            dates[dates < "2022-11-01"].dt.to_period("M").value_counts().sort_index()
        )
        print(monthly.to_string())


def run_analyze_stage(df, output_path):
    labels = build_final_labels(output_path)
    key = pd.read_csv(os.path.join(output_path, "annotation-key.csv"))

    precision_table = per_rule_precision(labels, key)
    rule_decisions = decide_rules(precision_table)

    print("\nPer-rule precision and decisions:")
    print(rule_decisions.to_string(index=False))

    filtered = filter_dataset(df, rule_decisions)

    counts = filtered["FilterDecision"].value_counts()
    print(f"\nFilter outcome: {counts.to_dict()}")

    review = filtered[filtered["FilterDecision"] == "REVIEW"]
    if len(review) > 0:
        review_file = os.path.join(output_path, "questions-needing-review.csv")
        review[["QuestionId", "Title", "MatchedRules"]].to_csv(
            review_file, index=False
        )
        print(f"{len(review)} questions matched only low-evidence rules; "
              f"inspect {review_file} and decide KEEP/DROP per rule.")

    cleaned = filtered[filtered["FilterDecision"] == "KEEP"].copy()

    # Headline precision among labeled questions that survive filtering.
    # Report with the caveat that the sample is stratified, not random.
    surviving_ids = set(cleaned["QuestionId"])
    surviving_labels = labels[labels["QuestionId"].isin(surviving_ids)]
    if len(surviving_labels) > 0:
        positives = int(surviving_labels["FinalLabel"].sum())
        n = len(surviving_labels)
        low, high = wilson_interval(positives, n)
        print(f"\nPrecision among labeled questions in the CLEANED set: "
              f"{positives}/{n} = {positives / n:.1%} "
              f"(Wilson 95% CI {low:.1%}-{high:.1%}; stratified sample)")

    sanity_check_pre_chatgpt(cleaned)

    rule_decisions.to_csv(
        os.path.join(output_path, "rule-decision-table.csv"), index=False
    )
    cleaned.to_csv(
        os.path.join(output_path, "llm-data-cleaned.csv"), index=False
    )
    print(f"\nWrote rule-decision-table.csv and llm-data-cleaned.csv "
          f"to {output_path}")


# ============================================================
# SECTION 5: MAIN
# ============================================================

if __name__ == "__main__":
    CURRENT_PATH = os.path.dirname(__file__)
    PARENT_PATH = os.path.dirname(CURRENT_PATH)

    OUTPUT_PATH = os.path.join(PARENT_PATH, "outputs", "validation")
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    DATA_WITH_RULES = os.path.join(OUTPUT_PATH, "llm-data-with-rules.csv")

    stage = sys.argv[1] if len(sys.argv) > 1 else None

    if stage == "sample":
        df = pd.read_csv(DATA_WITH_RULES)
        df["MatchedRules"] = df["MatchedRules"].fillna("")

        sample = build_annotation_sample(df)
        write_annotation_files(sample, OUTPUT_PATH)
        print(f"\nSampled {len(sample)} questions across "
              f"{sample['Stratum'].nunique()} strata. Seed={RANDOM_SEED}.")

    elif stage == "agreement":
        run_agreement_stage(OUTPUT_PATH)

    elif stage == "analyze":
        df = pd.read_csv(DATA_WITH_RULES)
        df["MatchedRules"] = df["MatchedRules"].fillna("")
        run_analyze_stage(df, OUTPUT_PATH)

    else:
        print("Usage: python llm_validation.py [sample|agreement|analyze]")
