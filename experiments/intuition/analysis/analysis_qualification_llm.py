"""Analyze LLM qualification-task outputs at the trial level.

This script evaluates whether model responses match the expected answer for the
qualification prompt. For the area qualification, the expected answer is
`scenario_mec` (Utilitarian Sum). For the volume qualification, it is
`scenario_nbs` (Nash Product).
"""

from __future__ import annotations

import argparse

import pandas as pd
from scipy.stats import binomtest


def _validate_columns(df: pd.DataFrame, chart_type: str) -> None:
    """Validate required columns for qualification analysis.

    Args:
        df: Input LLM output dataframe.
        chart_type: Qualification type (`area` or `volume`).
    """

    required = {"scenario_hash", "question", "response"}
    if chart_type == "area":
        required.add("scenario_mec")
    else:
        required.add("scenario_nbs")
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
            + ". Provide a qualification output CSV from new_model_experiment.py."
        )


def _dedupe_by_prompt(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Dedupe rows by `(scenario_hash, question)` and count conflicts.

    Args:
        df: Input rows.

    Returns:
        Deduped dataframe and number of conflicting prompt keys.
    """

    key_columns = ["scenario_hash", "question"]
    response_nunique = df.groupby(key_columns, dropna=False)["response"].nunique(
        dropna=False
    )
    conflicting_prompts = int((response_nunique > 1).sum())
    deduped = df.drop_duplicates(subset=key_columns, keep="last").copy()
    return deduped, conflicting_prompts


def _infer_num_options(df: pd.DataFrame) -> int:
    """Infer number of options from probability columns.

    Args:
        df: Input dataframe.

    Returns:
        Number of choice options; falls back to 3 if unavailable.
    """

    probability_columns = [
        col for col in df.columns if str(col).startswith("probability: ")
    ]
    if probability_columns:
        return len(probability_columns)
    return 3


def _print_accuracy(label: str, correct: int, total: int, chance_rate: float) -> None:
    """Print accuracy, exact CI, and one-sided binomial test against chance.

    Args:
        label: Label for the subset.
        correct: Number correct.
        total: Total trials in subset.
        chance_rate: Chance baseline.
    """

    if total == 0:
        print(f"{label}: no trials")
        return
    test = binomtest(k=correct, n=total, p=chance_rate, alternative="greater")
    ci = test.proportion_ci(confidence_level=0.95)
    accuracy = correct / total
    print(
        f"{label}: {correct}/{total} ({accuracy:.1%}) "
        f"95% CI [{ci.low:.3f}, {ci.high:.3f}], p={test.pvalue:.6f}"
    )


def _prepare_scored_rows(
    analysis_df: pd.DataFrame, expected_column: str
) -> pd.DataFrame:
    """Return scored rows with normalized response and expected columns.

    Args:
        analysis_df: Input dataframe to score.
        expected_column: Column containing expected answers.

    Returns:
        Filtered dataframe containing non-null response/expected rows.
    """

    scored_df = analysis_df.copy()
    scored_df["response"] = scored_df["response"].astype("string").str.strip()
    scored_df["expected"] = scored_df[expected_column].astype("string").str.strip()
    return scored_df[scored_df["response"].notna() & scored_df["expected"].notna()]


def _print_accuracy_by_agreement(
    scored_df: pd.DataFrame,
    *,
    chance_rate: float,
) -> None:
    """Print accuracy split by agreement/disagreement scenarios.

    Args:
        scored_df: Scored rows with `response`, `expected`, `scenario_nbs`,
            and `scenario_mec`.
        chance_rate: Chance baseline for one-sided binomial tests.
    """

    agree_mask = scored_df["scenario_nbs"] == scored_df["scenario_mec"]
    disagree_mask = ~agree_mask
    agree_total = int(agree_mask.sum())
    disagree_total = int(disagree_mask.sum())
    agree_correct = int(
        (
            scored_df.loc[agree_mask, "response"]
            == scored_df.loc[agree_mask, "expected"]
        ).sum()
    )
    disagree_correct = int(
        (
            scored_df.loc[disagree_mask, "response"]
            == scored_df.loc[disagree_mask, "expected"]
        ).sum()
    )
    _print_accuracy(
        "Agreement-scenario accuracy",
        agree_correct,
        agree_total,
        chance_rate,
    )
    _print_accuracy(
        "Disagreement-scenario accuracy",
        disagree_correct,
        disagree_total,
        chance_rate,
    )


def analyze_qualification_file(
    file_path: str,
    *,
    chart_type: str,
    dedupe_prompts: bool,
) -> None:
    """Analyze a qualification-task output file.

    Args:
        file_path: Path to the LLM output CSV.
        chart_type: Qualification type (`area` or `volume`).
        dedupe_prompts: Whether to dedupe by `(scenario_hash, question)`.
    """

    raw_df = pd.read_csv(file_path)
    _validate_columns(raw_df, chart_type)

    print(f"Loaded rows: {len(raw_df)}")
    analysis_df = raw_df
    if dedupe_prompts:
        analysis_df, conflicting_prompts = _dedupe_by_prompt(raw_df)
        print(f"Unique prompts (scenario_hash, question): {len(analysis_df)}")
        print(
            "Prompts with conflicting responses before dedupe: "
            f"{conflicting_prompts}"
        )
    else:
        print("Dedupe disabled: using all rows as trials.")

    expected_column = "scenario_mec" if chart_type == "area" else "scenario_nbs"
    comparison_df = _prepare_scored_rows(analysis_df, expected_column)

    total_trials = len(comparison_df)
    correct_trials = int((comparison_df["response"] == comparison_df["expected"]).sum())
    num_options = _infer_num_options(analysis_df)
    chance_rate = 1.0 / float(num_options)

    print(f"Chart type: {chart_type}")
    print(f"Expected answer column: {expected_column}")
    print(f"Chance baseline: 1/{num_options} = {chance_rate:.3f}")
    _print_accuracy("Overall accuracy", correct_trials, total_trials, chance_rate)

    if {"scenario_nbs", "scenario_mec"}.issubset(comparison_df.columns):
        _print_accuracy_by_agreement(comparison_df, chance_rate=chance_rate)

    print("Response counts:")
    response_counts = comparison_df["response"].value_counts(dropna=False)
    for label, count in response_counts.items():
        print(f"  {label}: {int(count)}")


def main() -> None:
    """Parse CLI arguments and run qualification analysis."""

    parser = argparse.ArgumentParser(
        description="Analyze LLM qualification-task CSV output.",
    )
    parser.add_argument("--file", required=True, help="Path to LLM output CSV.")
    parser.add_argument(
        "--chart-type",
        choices=["area", "volume"],
        default="area",
        help="Which qualification answer key to use.",
    )
    parser.add_argument(
        "--no-dedupe-prompts",
        action="store_true",
        help="Use all rows as trials (default dedupes by scenario_hash/question).",
    )
    args = parser.parse_args()
    analyze_qualification_file(
        args.file,
        chart_type=args.chart_type,
        dedupe_prompts=not args.no_dedupe_prompts,
    )


if __name__ == "__main__":
    main()
