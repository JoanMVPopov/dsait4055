"""
RQ2 shared preparation.

Builds the analysis dataframe used by every RQ2 script:
  - loads the cleaned LLM set (raw fallback) and the mature sample,
  - computes CENSORING-FIXED windowed outcomes (the core of RQ2 sec 2),
  - computes PRE-POSTING features for the regression (sec 5),
  - stacks both into one frame with an IsLLM indicator.

The windowing is the important part. "Did the question get an answer?"
is right-censored: a question posted shortly before the data snapshot
has had less time to attract an answer. Because LLM questions cluster
late in the window, naive answer rates are biased downward for LLM
questions - in the exact direction of our hypothesis. We therefore:
  - define answered/accepted WITHIN a fixed number of days, and
  - EXCLUDE questions too young to have been observed for that long.

Run standalone to sanity-check the build:
    python rq2_prepare.py
"""

import os
import re
import warnings

import numpy as np
import pandas as pd

# ============================================================
# SECTION 1: CONFIGURATION AND PATHS
# ============================================================

CURRENT_PATH = os.path.dirname(__file__)
PARENT_PATH = os.path.dirname(CURRENT_PATH)
DATA_PATH = os.path.join(PARENT_PATH, "data")

CLEANED_LLM_FILENAME = os.path.join(
    PARENT_PATH, "outputs", "validation", "llm-data-cleaned.csv"
)
RAW_LLM_FILENAME = os.path.join(DATA_PATH, "llm", "llm-combined-data.csv")
MATURE_FILENAME = os.path.join(DATA_PATH, "mature", "mature-combined-data.csv")

# Cached analysis frame. Built once (streams the 3.6 GB mature file); the
# descriptive/regression scripts read this instead of re-streaming.
RQ2_FRAME_CACHE = os.path.join(
    PARENT_PATH, "outputs", "rq2", "rq2-analysis-frame.csv"
)

# Date the SEDE data was extracted. Every question's observation window
# ends here, so it controls which questions are "too young" to observe.
# Accepted forms:
#   - a single date string, e.g. "2026-06-10"  (both datasets pulled same day)
#   - a dict {"LLM": "2026-06-10", "Mature": "2026-06-13"} if pulled on
#     different days (the LLM and mature files here have different mtimes)
#   - None -> inferred PER DATASET as the latest activity in that dataset
#     (a safe lower bound) and printed. SET IT EXPLICITLY for the paper.
SNAPSHOT_DATE = None

# Rows per chunk when streaming the large mature CSV, so Body never sits
# in memory all at once. Lower it if the process is OOM-killed; raise it
# if you have RAM to spare. The small LLM file is streamed the same way.
CHUNK_SIZE = 100_000

# Observation windows, in days.
ANSWER_WINDOW_DAYS = 30
ACCEPT_WINDOW_DAYS = 90

ANALYSIS_START = "2022-01"
ANALYSIS_END = "2026-05"

# Columns expected in BOTH datasets (same SEDE schema).
REQUIRED_COLUMNS = [
    "QuestionId", "CreationDate", "FirstAnswerDate",
    "AcceptedAnswerCreationDate", "OwnerReputation", "Title", "Body",
    "Tags", "Score", "ViewCount", "CommentCount", "ClosedDate",
]


# ============================================================
# SECTION 2: LOADING
# ============================================================

def _llm_source():
    """Path to the LLM dataset (cleaned preferred) + whether it is the
    validated set. The file is small but streamed the same way as mature."""
    if os.path.exists(CLEANED_LLM_FILENAME):
        print(f"LLM: using cleaned dataset ({CLEANED_LLM_FILENAME})")
        return CLEANED_LLM_FILENAME, True
    warnings.warn(
        "Cleaned LLM dataset not found - falling back to the RAW export. "
        "Final paper numbers must use the validated set."
    )
    return RAW_LLM_FILENAME, False


def _mature_source():
    if not os.path.exists(MATURE_FILENAME):
        raise FileNotFoundError(
            f"Mature per-question sample not found at {MATURE_FILENAME}. "
            f"This is the extract-mature-sample-per-month output (full "
            f"post fields), not the monthly-counts aggregate."
        )
    return MATURE_FILENAME


def _check_columns(path, label):
    """Verify the required columns exist before streaming the whole file."""
    header = pd.read_csv(path, nrows=0)
    missing = [c for c in REQUIRED_COLUMNS if c not in header.columns]
    if missing:
        raise KeyError(
            f"{label} dataset is missing required columns: {missing}"
        )


# ============================================================
# SECTION 3: FEATURE HELPERS
# ============================================================

_TAG_HTML = re.compile(r"<[^>]+>")
_TAG_FIELD = re.compile(r"<([^<>]+)>")
# A code BLOCK on SO is <pre ...><code>...</code></pre>. Match the whole
# <pre>...</pre> span (DOTALL: blocks span newlines). Inline code is a
# bare <code> NOT wrapped in <pre> and is left in the prose.
_CODE_BLOCK_HTML = re.compile(r"<pre\b.*?</pre>", re.IGNORECASE | re.DOTALL)


def body_features(body):
    """Decompose a question Body into prose vs code, in one pass.

    Returns (prose_len, code_len, has_inline, has_block, has_any) where:
      - code_len  = chars of text inside <pre>...</pre> code BLOCKS,
      - prose_len = chars of everything else (prose + inline <code>),
        i.e. the body with code blocks removed and HTML tags stripped.
    The two lengths are disjoint and additive, so they are not
    mechanically correlated (BodyLength does NOT contain CodeLength).

    Code-type flags use raw tag counts. Each <pre> wraps exactly one
    <code>, so inline = (#<code>) - (#<pre>). Counting "<pre" rather than
    the literal "<pre><code>" also catches syntax-highlighted blocks
    (<pre class="lang-..."><code>), which a bare "<code>" match would have
    miscounted as inline.
    """
    if pd.isna(body):
        return (0, 0, 0, 0, 0)
    s = str(body)

    blocks = _CODE_BLOCK_HTML.findall(s)
    code_len = sum(len(_TAG_HTML.sub(" ", b)) for b in blocks)
    prose_len = len(_TAG_HTML.sub(" ", _CODE_BLOCK_HTML.sub(" ", s)))

    n_code = s.count("<code")
    n_pre = s.count("<pre")
    has_block = int(n_pre > 0)
    has_inline = int((n_code - n_pre) > 0)
    has_any = int(n_code > 0)

    return (prose_len, code_len, has_inline, has_block, has_any)


def count_tags(tags_value):
    if pd.isna(tags_value):
        return 0
    return len(_TAG_FIELD.findall(str(tags_value)))


# ============================================================
# SECTION 4: PER-DATASET PREPARATION
# ============================================================

def _parse_dates(df):
    for column in ["CreationDate", "FirstAnswerDate",
                   "AcceptedAnswerCreationDate", "ClosedDate"]:
        df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def _windowed_outcomes(df, snapshot_date):
    """
    Censoring-fixed binary outcomes. A question is ELIGIBLE for a given
    window only if its full window fits before the snapshot; otherwise
    the outcome is NaN (dropped per analysis, not counted as a failure).
    """
    observed_days = (snapshot_date - df["CreationDate"]).dt.days

    # Time from posting to first answer / acceptance (days), NaT-safe.
    days_to_first = (df["FirstAnswerDate"] - df["CreationDate"]).dt.days
    days_to_accept = (
        df["AcceptedAnswerCreationDate"] - df["CreationDate"]
    ).dt.days

    # A valid within-window answer must arrive AFTER posting: 0 <= gap <= window.
    # The >= 0 guard rejects answers timestamped before the question, which
    # happens for a handful of merged-duplicate / migrated posts (SEDE's
    # MIN(answer date) can predate CreationDate). Without it, a negative gap
    # trivially satisfies "<= window" and counts as a spurious fast answer.

    # Answered within window
    eligible_answer = observed_days >= ANSWER_WINDOW_DAYS
    answered = (
        df["FirstAnswerDate"].notna()
        & (days_to_first >= 0) & (days_to_first <= ANSWER_WINDOW_DAYS)
    ).astype(float)
    df[f"AnsweredWithin{ANSWER_WINDOW_DAYS}Days"] = answered.where(
        eligible_answer, np.nan
    )

    # Accepted within window
    eligible_accept = observed_days >= ACCEPT_WINDOW_DAYS
    accepted = (
        df["AcceptedAnswerCreationDate"].notna()
        & (days_to_accept >= 0) & (days_to_accept <= ACCEPT_WINDOW_DAYS)
    ).astype(float)
    df[f"AcceptedWithin{ACCEPT_WINDOW_DAYS}Days"] = accepted.where(
        eligible_accept, np.nan
    )

    # Continuous: time to first answer in hours (answered questions only).
    # Descriptive use only - conditions on being answered (selection bias);
    # survival analysis is the unbiased version.
    hours_to_first = (
        (df["FirstAnswerDate"] - df["CreationDate"]).dt.total_seconds() / 3600
    )
    # Keep only valid (non-negative) times; negatives are corrupt merged/
    # migrated timestamps (see the >= 0 guard above) and would otherwise rank
    # as the "fastest" answers. NaN => excluded from the time descriptive,
    # consistent with the survival code dropping non-positive durations.
    df["TimeToFirstAnswerHours"] = hours_to_first.where(
        df["FirstAnswerDate"].notna() & (hours_to_first >= 0), np.nan
    )

    df["ObservedDays"] = observed_days
    return df


def _features(df):
    df["Month"] = df["CreationDate"].dt.to_period("M").astype(str)

    # One pass over Body -> prose/code lengths + code-type flags.
    feats = df["Body"].apply(body_features)
    df["BodyLength"] = [f[0] for f in feats]   # prose only (excl. code blocks)
    df["CodeLength"] = [f[1] for f in feats]   # text inside code blocks
    df["LogBodyLength"] = np.log1p(df["BodyLength"])
    df["LogCodeLength"] = np.log1p(df["CodeLength"])
    df["TitleLength"] = df["Title"].fillna("").str.len()
    df["HasInlineCode"] = [f[2] for f in feats]
    df["HasCodeBlock"] = [f[3] for f in feats]
    df["HasAnyCode"] = [f[4] for f in feats]   # derived; descriptive use only
    df["TagCount"] = df["Tags"].apply(count_tags)

    # Owner reputation: missing => deleted/community user. Flag it and keep
    # the raw numeric; the median impute is DEFERRED to after all chunks are
    # concatenated, so the median is per-DATASET, not per-chunk.
    reputation = pd.to_numeric(df["OwnerReputation"], errors="coerce")
    df["ReputationMissing"] = reputation.isna().astype(int)
    df["RepNumeric"] = reputation

    # Community reception extras (used by the friction subsection later).
    df["IsClosed"] = df["ClosedDate"].notna().astype(int)
    df["ScoreNum"] = pd.to_numeric(df["Score"], errors="coerce")
    df["CommentCountNum"] = pd.to_numeric(df["CommentCount"], errors="coerce")
    df["LogViewCount"] = np.log1p(
        pd.to_numeric(df["ViewCount"], errors="coerce").fillna(0)
    )

    return df


# Heavy text columns, dropped right after features are computed so the slim
# frame accumulated across chunks never holds Body.
_DROP_AFTER_FEATURES = ["Body", "Title", "Tags"]


def prepare_features(chunk, is_llm):
    """Snapshot-INDEPENDENT preparation, safe to run per chunk: parse dates,
    compute posting-time features, tag IsLLM/Dataset, and drop the heavy text
    columns. Deliberately does NOT dedup, window-filter, impute reputation,
    or compute outcomes - those are global/per-dataset and run once on the
    concatenated slim frame (see _load_prepared)."""
    df = _parse_dates(chunk)
    df = _features(df)
    df["IsLLM"] = int(is_llm)
    df["Dataset"] = "LLM" if is_llm else "Mature"
    return df.drop(columns=_DROP_AFTER_FEATURES, errors="ignore")


# ============================================================
# SECTION 5: BUILD THE COMBINED FRAME
# ============================================================

def _infer_snapshot_date(df):
    candidates = []
    for column in ["CreationDate", "FirstAnswerDate",
                   "AcceptedAnswerCreationDate"]:
        parsed = pd.to_datetime(df[column], errors="coerce")
        candidates.append(parsed.max())
    return max(c for c in candidates if pd.notna(c))


def _snapshot_for(df, label):
    """Resolve the snapshot date for one dataset from SNAPSHOT_DATE
    (single value, per-dataset dict, or None=infer)."""
    configured = SNAPSHOT_DATE
    if isinstance(configured, dict):
        configured = configured.get(label)
    if configured is not None:
        snapshot = pd.Timestamp(configured)
        print(f"Snapshot {label} (configured): {snapshot.date()}")
    else:
        snapshot = _infer_snapshot_date(df)
        print(f"Snapshot {label} (inferred): {snapshot.date()} "
              f"- set SNAPSHOT_DATE explicitly for the paper.")
    return snapshot


def _load_prepared(path, is_llm, label):
    """Stream a CSV in chunks (Body kept out of memory), then run the global
    steps once: dedup, per-dataset reputation impute, window restriction,
    snapshot resolution, and censoring-fixed outcomes."""
    _check_columns(path, label)
    needed = set(REQUIRED_COLUMNS)

    parts = []
    n_read = 0
    for chunk in pd.read_csv(path, usecols=lambda c: c in needed,
                             chunksize=CHUNK_SIZE):
        n_read += len(chunk)
        parts.append(prepare_features(chunk, is_llm))
    slim = pd.concat(parts, ignore_index=True)
    print(f"{label}: streamed {n_read} rows (chunksize {CHUNK_SIZE}).")

    # --- global steps (must NOT be per-chunk) ---
    slim = slim.drop_duplicates(subset="QuestionId").copy()

    # Per-dataset median reputation impute (deferred from prepare_features).
    rep = slim["RepNumeric"]
    slim["LogOwnerReputation"] = np.log1p(rep.fillna(rep.median()))
    slim = slim.drop(columns=["RepNumeric"])

    # Window restriction. Upper bound is the first day after the last
    # analysis month, so the whole of ANALYSIS_END is included.
    n_unparsed = slim["CreationDate"].isna().sum()
    if n_unparsed:
        print(f"  NOTE: dropping {n_unparsed} {label} rows with "
              f"unparseable CreationDate.")
    in_window = (
        (slim["CreationDate"] >= pd.Timestamp(ANALYSIS_START))
        & (slim["CreationDate"] < pd.Timestamp(ANALYSIS_END)
           + pd.offsets.MonthBegin(1))
    )
    slim = slim[in_window].copy()

    # Per-dataset snapshot + censoring-fixed windowed outcomes.
    snapshot = _snapshot_for(slim, label)
    slim = _windowed_outcomes(slim, snapshot)

    return slim, snapshot


def build_rq2_frame():
    llm_path, used_cleaned = _llm_source()
    mature_path = _mature_source()

    llm, llm_snapshot = _load_prepared(llm_path, is_llm=True, label="LLM")
    mature, mature_snapshot = _load_prepared(mature_path, is_llm=False,
                                             label="Mature")

    # A question can match both populations (e.g. tagged python AND langchain).
    # Treat LLM membership as definitive and remove the overlap from the mature
    # side, so no QuestionId appears in both groups (no contradictory IsLLM
    # labels, no double-counting in the comparison).
    overlap = mature["QuestionId"].isin(set(llm["QuestionId"]))
    n_overlap = int(overlap.sum())
    if n_overlap:
        print(f"Overlap: {n_overlap} QuestionIds in both sets - removed from "
              f"the mature side (LLM membership treated as definitive).")
    mature = mature[~overlap].copy()

    combined = pd.concat([llm, mature], ignore_index=True)

    keep = [
        "QuestionId", "Dataset", "IsLLM", "Month", "CreationDate",
        "ObservedDays",
        f"AnsweredWithin{ANSWER_WINDOW_DAYS}Days",
        f"AcceptedWithin{ACCEPT_WINDOW_DAYS}Days",
        "TimeToFirstAnswerHours",
        "BodyLength", "LogBodyLength", "CodeLength", "LogCodeLength",
        "TitleLength",
        "HasInlineCode", "HasCodeBlock", "HasAnyCode",
        "TagCount", "LogOwnerReputation", "ReputationMissing",
        "IsClosed", "ScoreNum", "CommentCountNum", "LogViewCount",
    ]
    combined = combined[keep]

    combined.attrs["used_cleaned"] = used_cleaned
    combined.attrs["snapshot_date"] = {
        "LLM": llm_snapshot, "Mature": mature_snapshot,
    }
    return combined


def load_or_build_rq2_frame(use_cache=True):
    """Return the RQ2 analysis frame, reusing the cached CSV when present.

    Building re-streams the 3.6 GB mature file (~minutes); the cache makes
    the descriptive/regression scripts start instantly. After changing any
    prepare logic, delete outputs/rq2/rq2-analysis-frame.csv (or pass
    use_cache=False) to force a rebuild - the cache is NOT auto-invalidated.
    """
    if use_cache and os.path.exists(RQ2_FRAME_CACHE):
        print(f"RQ2 frame: loading cached {RQ2_FRAME_CACHE} "
              f"(delete it to force a rebuild).")
        # Month is a period-like string ('2023-07'); keep it as text so
        # C(Month) treats it categorically.
        return pd.read_csv(RQ2_FRAME_CACHE, dtype={"Month": str})

    frame = build_rq2_frame()
    os.makedirs(os.path.dirname(RQ2_FRAME_CACHE), exist_ok=True)
    frame.to_csv(RQ2_FRAME_CACHE, index=False)
    print(f"RQ2 frame: built and cached to {RQ2_FRAME_CACHE}.")
    return frame


def print_build_report(df):
    ans = f"AnsweredWithin{ANSWER_WINDOW_DAYS}Days"
    acc = f"AcceptedWithin{ACCEPT_WINDOW_DAYS}Days"

    print("\n--- RQ2 build report ---")
    print(f"Rows total: {len(df)}")
    print(df["Dataset"].value_counts().to_string())

    print(f"\nEligible for {ans} (full window observed):")
    print(df.groupby("Dataset")[ans].apply(lambda s: s.notna().sum()).to_string())
    print(f"Dropped as too young: "
          f"{df[ans].isna().sum()} of {len(df)}")

    print(f"\nEligible for {acc}:")
    print(df.groupby("Dataset")[acc].apply(lambda s: s.notna().sum()).to_string())

    print("\nFeature sanity (means by group):")
    print(df.groupby("Dataset")[
        ["BodyLength", "CodeLength", "TitleLength",
         "HasInlineCode", "HasCodeBlock", "HasAnyCode", "TagCount",
         "LogOwnerReputation"]
    ].mean().round(2).to_string())

    # Does the prose/code split matter? If LLM questions carry much more
    # code, raw total length would proxy "amount of code" rather than
    # writing effort - this is the number that decides BodyLength vs total.
    print("\nMissing-reputation rate by group "
          "(for the imputation robustness note):")
    print(df.groupby("Dataset")["ReputationMissing"].mean().round(3).to_string())


if __name__ == "__main__":
    frame = build_rq2_frame()
    print_build_report(frame)

    os.makedirs(os.path.dirname(RQ2_FRAME_CACHE), exist_ok=True)
    frame.to_csv(RQ2_FRAME_CACHE, index=False)
    print(f"\nWrote {RQ2_FRAME_CACHE}")
