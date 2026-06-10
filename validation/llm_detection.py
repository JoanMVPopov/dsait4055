"""
Offline re-detection of LLM-related questions.

Recomputes, for every question in llm-combined-data.csv, WHICH detection
rule(s) caused its inclusion: each tag in the SEDE tag list is one rule,
and each keyword pattern is one rule.

The matching semantics deliberately mirror the SEDE query:
- tag rules  : exact tag membership (Tags column, e.g. "<python><gpt-4>")
- keyword    : case-insensitive SUBSTRING match on the RAW Title and the
               RAW (HTML) Body, exactly like SQL `LOWER(col) LIKE '%kw%'`.
               Note that ' llm ' keeps its literal surrounding spaces.

Output: a copy of the dataset with three new columns
- MatchedTagRules     : pipe-separated tag rules that fired
- MatchedKeywordRules : pipe-separated keyword rules that fired
- MatchedRules        : union of both (prefixed "tag:" / "kw:")

Run this once, before sampling for annotation:
    python llm_detection.py
"""

import os
import re

import pandas as pd

# ============================================================
# SECTION 1: RULE DEFINITIONS (copied verbatim from the SEDE query)
# ============================================================
# Keep the *full* original lists here, including the suspicious entries
# (openai-gym, chatbot, ...). The point is to measure each rule's
# precision empirically, not to pre-judge it.

LLM_TAG_RULES = [
    "openai-api", "chatgpt-api", "azure-openai", "langchain", "chatbot",
    "py-langchain", "langchain-js", "large-language-model",
    "gpt-3", "gpt-4", "gpt-4o", "gpt-4o-mini",
    "gpt-5", "chat-gpt-4", "rag",
    "vllm", "claude", "claude-code", "google-gemini", "gemini",
    "gemini-cli", "gemini-code-assist", "google-gemini-file-api",
    "google-gemini-context-caching",
    "openaiembeddings", "openai-assistants-api",
    "openai-agents", "openai-codex", "chatgpt-plugin",
    "chatgpt-function-call", "autogpt", "privategpt", "gpt-index",
    "gpt4all", "pygpt4all", "h2ogpt", "litellm",
    "langchain-agents", "langchain4j", "quarkus-langchain4j",
    "langchain-together", "llm-sql-generation", "promptfoo",
    "azure-promptflow", "artificial-intelligence", "agent", "prompt",
    "openai-gym",
]

# Each keyword is matched as a plain lowercase substring, like SQL LIKE.
# ' llm ' keeps its literal spaces, exactly as in the query.
LLM_KEYWORD_RULES = [
    "large language model",
    " llm ",
    "chatgpt",
    "openai",
    "langchain",
    "retrieval augmented generation",
    "gpt-3",
    "gpt-4",
    "claude",
    "gemini",
]


# ============================================================
# SECTION 2: MATCHING FUNCTIONS
# ============================================================

TAG_PATTERN = re.compile(r"<([^<>]+)>")


def parse_tags(tags_value):
    """'<python><gpt-4>' -> ['python', 'gpt-4']. NaN-safe."""
    if pd.isna(tags_value):
        return []
    return TAG_PATTERN.findall(str(tags_value))


def match_tag_rules(tags_value):
    """Return the list of tag rules that fire for this question."""
    question_tags = set(parse_tags(tags_value))
    return [tag for tag in LLM_TAG_RULES if tag in question_tags]


def match_keyword_rules(title_value, body_value):
    """
    Return the list of keyword rules that fire for this question.

    IMPORTANT: matches against the RAW body (HTML included), because that
    is what the SEDE query did. Stripping HTML first would change which
    questions match and break auditability.
    """
    title = "" if pd.isna(title_value) else str(title_value).lower()
    body = "" if pd.isna(body_value) else str(body_value).lower()

    return [
        keyword for keyword in LLM_KEYWORD_RULES
        if keyword in title or keyword in body
    ]


# ============================================================
# SECTION 3: APPLY DETECTION TO A DATAFRAME
# ============================================================

def apply_rule_detection(df):
    """
    Add per-rule attribution columns to the LLM question dataframe.
    Also recomputes IsLLMTagged / IsLLMKeywordDetected from scratch so
    they can be cross-checked against the values exported by SEDE.
    """
    df = df.copy()
    df = df.drop_duplicates(subset="QuestionId")

    tag_matches = df["Tags"].apply(match_tag_rules)
    keyword_matches = [
        match_keyword_rules(title, body)
        for title, body in zip(df["Title"], df["Body"])
    ]

    df["MatchedTagRules"] = ["|".join(m) for m in tag_matches]
    df["MatchedKeywordRules"] = ["|".join(m) for m in keyword_matches]
    df["MatchedRules"] = [
        "|".join([f"tag:{t}" for t in tags] + [f"kw:{k}" for k in kws])
        for tags, kws in zip(tag_matches, keyword_matches)
    ]

    df["RecomputedIsLLMTagged"] = [int(len(m) > 0) for m in tag_matches]
    df["RecomputedIsLLMKeywordDetected"] = [
        int(len(m) > 0) for m in keyword_matches
    ]

    return df


def report_detection_sanity(df):
    """
    Print cross-checks between SEDE flags and the offline recomputation.
    Disagreements usually mean a semantics mismatch worth investigating
    (collation, encoding, HTML escaping) before trusting per-rule results.
    """
    unmatched = (df["MatchedRules"] == "").sum()
    print(f"Questions matching NO rule offline: {unmatched}")
    if unmatched > 0:
        print("  -> inspect these rows; the offline matcher may diverge "
              "from the SQL semantics (e.g. HTML entities in Body).")

    for sede_col, recomputed_col in [
        ("IsLLMTagged", "RecomputedIsLLMTagged"),
        ("IsLLMKeywordDetected", "RecomputedIsLLMKeywordDetected"),
    ]:
        if sede_col in df.columns:
            disagree = (
                df[sede_col].fillna(0).astype(int)
                != df[recomputed_col]
            ).sum()
            print(f"Disagreement {sede_col} vs offline: {disagree} rows")


def rule_frequency_table(df):
    """One row per rule: how many questions it matched, and how many it
    matched EXCLUSIVELY (no other rule fired). Exclusive counts show how
    much data is at stake if a rule is dropped."""
    rows = []
    exploded = df["MatchedRules"].str.split("|")

    all_rules = (
        [f"tag:{t}" for t in LLM_TAG_RULES]
        + [f"kw:{k}" for k in LLM_KEYWORD_RULES]
    )

    for rule in all_rules:
        mask = exploded.apply(lambda rules: rule in rules)
        exclusive_mask = exploded.apply(
            lambda rules: rules == [rule]
        )
        rows.append({
            "Rule": rule,
            "QuestionsMatched": int(mask.sum()),
            "QuestionsMatchedExclusively": int(exclusive_mask.sum()),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("QuestionsMatched", ascending=False)
        .reset_index(drop=True)
    )


# ============================================================
# SECTION 4: MAIN
# ============================================================

if __name__ == "__main__":
    CURRENT_PATH = os.path.dirname(__file__)
    PARENT_PATH = os.path.dirname(CURRENT_PATH)
    DATA_PATH = os.path.join(PARENT_PATH, "data")

    LLM_DATA_FILENAME = os.path.join(DATA_PATH, "llm", "llm-combined-data.csv")

    OUTPUT_PATH = os.path.join(PARENT_PATH, "outputs", "validation")
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    llm_df = pd.read_csv(LLM_DATA_FILENAME)
    llm_df = apply_rule_detection(llm_df)

    report_detection_sanity(llm_df)

    rule_table = rule_frequency_table(llm_df)
    print("\nRule frequency table (top 20):")
    print(rule_table.head(20).to_string(index=False))

    llm_df.to_csv(
        os.path.join(OUTPUT_PATH, "llm-data-with-rules.csv"),
        index=False
    )
    rule_table.to_csv(
        os.path.join(OUTPUT_PATH, "rule-frequency-table.csv"),
        index=False
    )

    print(f"\nWrote outputs to {OUTPUT_PATH}")
