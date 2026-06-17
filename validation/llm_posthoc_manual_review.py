"""
Post-hoc manual adjudication of the MANUAL_REVIEW rules left by
`llm_validation.py analyze`.

Rules matched by fewer than MIN_LABELS_PER_RULE sampled questions cannot
be auto-decided by precision, so `analyze` flags them MANUAL_REVIEW and
excludes the questions those rules SOLELY gate (written to
`questions-needing-review.csv`). A question matched by at least one KEEP
rule is unaffected -- only questions whose every match is a MANUAL_REVIEW
(or DROP) rule fall through.

This script lets the team record an explicit KEEP / DROP decision per such
rule and folds it back into the cleaned dataset, WITHOUT modifying the
frozen validation pipeline (`llm_validation.py`). It reuses that module's
`filter_dataset`, so the filtering logic stays in exactly one place.

The unit of decision is the RULE, not the individual question: you are
vouching that a low-evidence detection rule is (un)trustworthy, which is
the same rule-based logic the precision threshold uses -- not hand-picking
questions. For the rules here that happens to be one question each, but the
rule framing is what is defensible in the paper.

Workflow:

  1. python llm_posthoc_manual_review.py
       If `manual-rule-decisions.csv` is missing or incomplete, writes a
       template listing every rule that currently gates a review question
       (with an example title) and exits without changing anything.

  2. Fill the `Decision` column with KEEP or DROP for each rule, in
       `outputs/validation/manual-rule-decisions.csv`.

  3. python llm_posthoc_manual_review.py
       Applies the decisions and rewrites, in `outputs/validation/`:
         - rule-decision-table.csv      (MANUAL_REVIEW -> KEEP/DROP,
                                          ManualOverride flag set)
         - llm-data-cleaned.csv         (now includes the KEEP'd questions)
         - questions-needing-review.csv (remaining undecided rows; empty
                                          when everything is resolved)

Idempotent: re-running after a successful apply finds nothing left to
adjudicate. If you re-run `analyze` (which resets the table), just run this
script again -- the saved decisions file is re-applied automatically.
"""

import os
import sys

import pandas as pd

from llm_validation import filter_dataset


VALID_DECISIONS = {"KEEP", "DROP"}


def blocking_review_rules(filtered, rule_decisions):
    """The MANUAL_REVIEW rules that gate at least one REVIEW question.

    A REVIEW question is one with no KEEP rule (see filter_dataset). The
    rules worth adjudicating are the MANUAL_REVIEW rules appearing on those
    rows -- the other MANUAL_REVIEW rules don't change cleaned-set
    membership and are left untouched.
    """
    review = filtered[filtered["FilterDecision"] == "REVIEW"]
    manual_rules = set(
        rule_decisions.loc[
            rule_decisions["Decision"] == "MANUAL_REVIEW", "Rule"
        ]
    )

    rules_on_review_rows = set()
    for rules_str in review["MatchedRules"]:
        rules_on_review_rows.update(r for r in rules_str.split("|") if r)

    return sorted(rules_on_review_rows & manual_rules)


def write_decisions_template(blocking, filtered, rule_decisions, path):
    """Write a fill-in template: one row per blocking rule, with context."""
    review = filtered[filtered["FilterDecision"] == "REVIEW"]
    prec = rule_decisions.set_index("Rule")

    rows = []
    for rule in blocking:
        # An example question that this rule gates, to aid the decision.
        gated = review[
            review["MatchedRules"].apply(
                lambda s: rule in s.split("|")
            )
        ]
        example = gated.iloc[0] if len(gated) else None
        rows.append({
            "Rule": rule,
            "LabeledQuestions": int(prec.loc[rule, "LabeledQuestions"]),
            "Precision": prec.loc[rule, "Precision"],
            "QuestionsGated": len(gated),
            "ExampleQuestionId": "" if example is None else example["QuestionId"],
            "ExampleTitle": "" if example is None else example["Title"],
            "Decision": "",   # team fills KEEP or DROP
        })

    pd.DataFrame(rows).to_csv(path, index=False)


def load_decisions(path, blocking):
    """Read and validate the filled decisions file against the rules that
    actually need a decision. Returns {rule: KEEP|DROP}."""
    decisions_df = pd.read_csv(path)
    decisions_df["Decision"] = (
        decisions_df["Decision"].astype(str).str.strip().str.upper()
    )

    decided = {
        row["Rule"]: row["Decision"]
        for _, row in decisions_df.iterrows()
        if row["Decision"] in VALID_DECISIONS
    }

    invalid = decisions_df[
        ~decisions_df["Decision"].isin(VALID_DECISIONS)
        & decisions_df["Rule"].isin(blocking)
    ]
    if len(invalid):
        raise ValueError(
            "These blocking rules have no valid KEEP/DROP decision in "
            f"{os.path.basename(path)}:\n"
            + "\n".join(f"  {r}" for r in invalid["Rule"])
        )

    missing = [r for r in blocking if r not in decided]
    if missing:
        raise ValueError(
            f"{os.path.basename(path)} is missing rows for these blocking "
            "rules (re-run with no args to regenerate the template):\n"
            + "\n".join(f"  {r}" for r in missing)
        )

    return decided


def main(output_path):
    table_path = os.path.join(output_path, "rule-decision-table.csv")
    data_path = os.path.join(output_path, "llm-data-with-rules.csv")
    decisions_path = os.path.join(output_path, "manual-rule-decisions.csv")
    cleaned_path = os.path.join(output_path, "llm-data-cleaned.csv")
    review_path = os.path.join(output_path, "questions-needing-review.csv")

    if not os.path.exists(table_path):
        raise FileNotFoundError(
            f"{table_path} not found. Run `python llm_validation.py analyze` "
            "first."
        )

    df = pd.read_csv(data_path)
    df["MatchedRules"] = df["MatchedRules"].fillna("")
    rule_decisions = pd.read_csv(table_path)

    filtered = filter_dataset(df, rule_decisions)
    blocking = blocking_review_rules(filtered, rule_decisions)

    if not blocking:
        print("No MANUAL_REVIEW rules are gating any question -- nothing to "
              "adjudicate. (Either already resolved, or analyze produced no "
              "review questions.)")
        return

    if not os.path.exists(decisions_path):
        write_decisions_template(blocking, filtered, rule_decisions,
                                 decisions_path)
        print(f"Wrote decisions template to {decisions_path}")
        print(f"{len(blocking)} rule(s) need a manual KEEP/DROP decision:")
        for r in blocking:
            print(f"  {r}")
        print("\nFill the Decision column (KEEP or DROP), then re-run this "
              "script.")
        return

    decided = load_decisions(decisions_path, blocking)

    # Apply overrides to the decision table, flagging them as manual.
    table = rule_decisions.copy()
    if "ManualOverride" not in table.columns:
        table["ManualOverride"] = False
    for rule, decision in decided.items():
        mask = table["Rule"] == rule
        table.loc[mask, "Decision"] = decision
        table.loc[mask, "ManualOverride"] = True

    # Re-filter with the resolved table; this reproduces analyze's logic.
    refiltered = filter_dataset(df, table)
    cleaned = refiltered[refiltered["FilterDecision"] == "KEEP"].copy()
    review = refiltered[refiltered["FilterDecision"] == "REVIEW"].copy()

    before = len(filter_dataset(df, rule_decisions).query(
        "FilterDecision == 'KEEP'"))
    added = len(cleaned) - before

    table.to_csv(table_path, index=False)
    cleaned.to_csv(cleaned_path, index=False)
    review[["QuestionId", "Title", "MatchedRules"]].to_csv(
        review_path, index=False
    )

    kept = [r for r, d in decided.items() if d == "KEEP"]
    dropped = [r for r, d in decided.items() if d == "DROP"]
    print("Applied manual decisions:")
    for r in kept:
        print(f"  KEEP  {r}")
    for r in dropped:
        print(f"  DROP  {r}")
    print(f"\nCleaned set: {before} -> {len(cleaned)} questions "
          f"({added:+d} from manual KEEPs).")
    print(f"Remaining questions needing review: {len(review)}")
    print(f"\nRewrote rule-decision-table.csv, llm-data-cleaned.csv and "
          f"questions-needing-review.csv in {output_path}")


if __name__ == "__main__":
    CURRENT_PATH = os.path.dirname(__file__)
    PARENT_PATH = os.path.dirname(CURRENT_PATH)
    OUTPUT_PATH = os.path.join(PARENT_PATH, "outputs", "validation")
    main(OUTPUT_PATH)
