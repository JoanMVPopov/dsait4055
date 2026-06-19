"""
RQ2 section 7: community reception (the friction remnant).

Beyond whether questions get *answered*, how does the community *receive*
LLM questions vs mature ones? Same two-group machinery as the descriptive
comparison, on three reception outcomes:

  - IsClosed (binary)        -> chi-square + risk difference + Cramer's V
  - Score (skewed)           -> Mann-Whitney U + rank-biserial
  - CommentCount (skewed)    -> Mann-Whitney U + rank-biserial

Holm-Bonferroni across this family. Reuses compare_binary /
compare_continuous from rq2_descriptive so the stats stay identical.

Caveats for the writeup:
  - Closure is a CONSERVATIVE lower bound: SEDE hides deleted posts, so
    truly-closed-then-deleted questions are invisible.
  - Score and CommentCount accumulate AFTER posting; LLM questions are
    younger on average, so both are mildly right-censored. Descriptive
    only - no causal reading.

Run:
    python rq2_reception.py
"""

import os

import pandas as pd

from statsmodels.stats.multitest import multipletests

from rq2_prepare import load_or_build_rq2_frame, PARENT_PATH
from rq2_descriptive import (
    compare_binary, compare_continuous, format_for_display,
)


def run_reception(df):
    results = [
        compare_binary(df, "IsClosed"),
        compare_continuous(df, "ScoreNum"),
        compare_continuous(df, "CommentCountNum"),
    ]
    table = pd.DataFrame(results)

    reject, p_adjusted, _, _ = multipletests(
        table["PValue"].values, method="holm"
    )
    table["PValue_HolmAdjusted"] = p_adjusted
    table["Significant_05"] = reject

    return table


if __name__ == "__main__":
    frame = load_or_build_rq2_frame()
    table = run_reception(frame)

    print("\n=== RQ2 community reception (LLM vs mature) ===")
    print(format_for_display(table).to_string(index=False))

    out_path = os.path.join(PARENT_PATH, "outputs", "rq2")
    os.makedirs(out_path, exist_ok=True)
    table.to_csv(
        os.path.join(out_path, "rq2_reception_comparison.csv"), index=False
    )
    print(f"\nWrote rq2_reception_comparison.csv to {out_path}")
