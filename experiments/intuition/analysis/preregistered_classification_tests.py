"""Recompute preregistered classification-based exact binomial tests.

This script reproduces the participant-level classification tests described in
``pre-registration.md`` for the within-participant human study. It is designed
to work directly from the wide MTurk/JATOS export plus the qualification answer
key, without requiring pandas, scipy, or the plotting environment used in the
main analysis repository.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass
class Summary:
    """Container for preregistered classification-test results.

    Parameters:
        workers: Number of participants in the subset.
        random_rate: Aggregate disagree-trial random rate ``R``.
        p_guess: Derived guessing probability ``(1 - R) / 2``.
        disagree_k: Threshold used for Nash / IE / Random classification.
        disagree_fpr: Threshold-implied false positive rate for disagreement.
        nash_classified: Number of participants classified as Nash-consistent.
        ie_classified: Number of participants classified as IE-consistent.
        random_classified: Number of participants classified as Random-consistent.
        disagree_indeterminate: Number of disagreement indeterminate participants.
        disagree_p_log10: Base-10 log of the exact binomial tail probability for
            the preregistered disagreement test.
        agree_k: Threshold used for Principled / Decoy classification.
        agree_fpr: Threshold-implied false positive rate for agreement.
        principled_classified: Number of participants classified as Principled.
        decoy_classified: Number of participants classified as Decoy.
        agree_indeterminate: Number of agreement indeterminate participants.
        agree_p_log10: Base-10 log of the exact binomial tail probability for
            the preregistered agreement test.
    """

    workers: int
    random_rate: float
    p_guess: float
    disagree_k: int
    disagree_fpr: float
    nash_classified: int
    ie_classified: int
    random_classified: int
    disagree_indeterminate: int
    disagree_p_log10: float
    agree_k: int
    agree_fpr: float
    principled_classified: int
    decoy_classified: int
    agree_indeterminate: int
    agree_p_log10: float


def load_qualification_answers(path: str) -> dict[str, str]:
    """Load expected qualification answers.

    Parameters:
        path: Path to the JSON answer key.

    Returns:
        A mapping from ``Answer.<field>`` column names to expected answers.
    """

    with open(path, "r", encoding="utf-8") as handle:
        answers = json.load(handle)
    return {f"Answer.{key}": str(value) for key, value in answers.items()}


def load_rows(path: str) -> tuple[list[str], list[dict[str, str]]]:
    """Load the wide export rows.

    Parameters:
        path: Path to the CSV export.

    Returns:
        A pair of header names and CSV row dictionaries.
    """

    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames or [], list(reader)


def passes_area_qualification(
    row: dict[str, str], answers: dict[str, str], columns: list[str]
) -> bool:
    """Return whether a row passes the area qualification filter.

    Parameters:
        row: One wide CSV row.
        answers: Expected qualification answers.
        columns: Qualification columns to check.

    Returns:
        True if every checked answer matches the answer key.
    """

    for column in columns:
        expected = answers.get(column)
        if expected is None:
            continue
        if str(row.get(column, "")).strip() != expected.strip():
            return False
    return True


def explode_rows(
    rows: list[dict[str, str]], num_scenarios: int
) -> list[dict[str, str | bool]]:
    """Expand wide participant rows into one record per scenario response.

    Parameters:
        rows: Wide CSV rows.
        num_scenarios: Number of scenarios shown per participant.

    Returns:
        A flat list of participant-scenario records.
    """

    exploded: list[dict[str, str | bool]] = []
    for row in rows:
        worker = row["WorkerId"]
        for index in range(1, num_scenarios + 1):
            scenario_ie = row.get(f"Input.scenario_{index}_ie", "") or row.get(
                f"Input.scenario_{index}_mec", ""
            )
            scenario_nbs = row.get(f"Input.scenario_{index}_nbs", "")
            attention_response = row.get(f"Answer.q_question-{index}_attn", "")
            attention_answer = row.get(f"Answer.q_question-{index}_attn_answer", "")
            exploded.append(
                {
                    "WorkerId": worker,
                    "alpha": row.get(f"Input.scenario_{index}_alpha_bin", ""),
                    "scenario_ie": scenario_ie,
                    "scenario_nbs": scenario_nbs,
                    "response": row.get(f"Answer.q_question-{index}", ""),
                    "passed_attention": attention_response == attention_answer,
                }
            )
    return exploded


def filter_by_attention(
    rows: list[dict[str, str | bool]], attention_threshold: float
) -> list[dict[str, str | bool]]:
    """Filter rows by worker-level attention accuracy.

    Parameters:
        rows: Exploded participant-scenario rows.
        attention_threshold: Minimum worker-level attention pass rate.

    Returns:
        The subset of rows from workers meeting the threshold.
    """

    worker_attention: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        worker_attention[str(row["WorkerId"])].append(bool(row["passed_attention"]))
    keep_workers = {
        worker
        for worker, values in worker_attention.items()
        if (sum(values) / len(values)) >= attention_threshold
    }
    return [row for row in rows if str(row["WorkerId"]) in keep_workers]


def logsumexp(log_values: list[float]) -> float:
    """Return ``log(sum(exp(log_values)))`` stably.

    Parameters:
        log_values: Log-scale values to aggregate.

    Returns:
        The log of the summed probabilities.
    """

    maximum = max(log_values)
    if math.isinf(maximum):
        return maximum
    return maximum + math.log(sum(math.exp(value - maximum) for value in log_values))


def binomial_tail_log_probability(
    num_trials: int, success_probability: float, threshold: int
) -> float:
    """Return ``log(P(X >= threshold))`` for a binomial random variable.

    Parameters:
        num_trials: Number of Bernoulli trials.
        success_probability: Per-trial success probability.
        threshold: Inclusive success threshold.

    Returns:
        The natural log of the upper-tail probability.
    """

    if threshold <= 0:
        return 0.0
    if threshold > num_trials:
        return float("-inf")
    if success_probability <= 0.0:
        return float("-inf")
    if success_probability >= 1.0:
        return 0.0
    log_terms = []
    for successes in range(threshold, num_trials + 1):
        log_terms.append(
            math.lgamma(num_trials + 1)
            - math.lgamma(successes + 1)
            - math.lgamma(num_trials - successes + 1)
            + successes * math.log(success_probability)
            + (num_trials - successes) * math.log1p(-success_probability)
        )
    return logsumexp(log_terms)


def binomial_tail_probability(
    num_trials: int, success_probability: float, threshold: int
) -> float:
    """Return ``P(X >= threshold)`` for a binomial random variable.

    Parameters:
        num_trials: Number of Bernoulli trials.
        success_probability: Per-trial success probability.
        threshold: Inclusive success threshold.

    Returns:
        The upper-tail probability.
    """

    return math.exp(
        binomial_tail_log_probability(num_trials, success_probability, threshold)
    )


def determine_classification_threshold(
    num_trials: int, success_probability: float, alpha: float
) -> tuple[int, float]:
    """Find the smallest threshold whose upper-tail probability is below alpha.

    Parameters:
        num_trials: Number of trials used for the classification.
        success_probability: Null-model success probability.
        alpha: Desired false positive rate cutoff.

    Returns:
        A pair of the threshold ``k`` and its binomial tail probability.
    """

    for threshold in range(0, num_trials + 1):
        tail_probability = binomial_tail_probability(
            num_trials, success_probability, threshold
        )
        if tail_probability < alpha:
            return threshold, tail_probability
    return num_trials + 1, 0.0


def rounded_median(counts: list[int]) -> int:
    """Return the rounded median count.

    Parameters:
        counts: Integer counts.

    Returns:
        The median, rounded to the nearest integer.
    """

    ordered = sorted(counts)
    midpoint = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[midpoint]
    return int(round((ordered[midpoint - 1] + ordered[midpoint]) / 2))


def summarize_subset(rows: list[dict[str, str | bool]], alpha: float) -> Summary:
    """Compute preregistered classification-test summaries for one subset.

    Parameters:
        rows: Exploded participant-scenario rows after filtering.
        alpha: False positive rate used when deriving classification thresholds.

    Returns:
        A ``Summary`` with disagreement and agreement classification results.
    """

    by_worker_disagree: dict[str, Counter[str]] = defaultdict(Counter)
    by_worker_agree: dict[str, Counter[str]] = defaultdict(Counter)
    disagree_trial_count = 0

    for row in rows:
        worker = str(row["WorkerId"])
        scenario_ie = str(row["scenario_ie"])
        scenario_nbs = str(row["scenario_nbs"])
        response = str(row["response"])
        if scenario_ie == scenario_nbs:
            if response == scenario_nbs:
                by_worker_agree[worker]["shared"] += 1
            else:
                by_worker_agree[worker]["decoy"] += 1
            continue
        disagree_trial_count += 1
        if response == scenario_nbs:
            by_worker_disagree[worker]["nash"] += 1
        elif response == scenario_ie:
            by_worker_disagree[worker]["ie"] += 1
        else:
            by_worker_disagree[worker]["random"] += 1

    workers = sorted(set(by_worker_disagree) | set(by_worker_agree))
    random_count = sum(counter["random"] for counter in by_worker_disagree.values())
    random_rate = random_count / disagree_trial_count
    p_guess = (1.0 - random_rate) / 2.0

    disagree_counts = [
        counter["nash"] + counter["ie"] + counter["random"]
        for counter in by_worker_disagree.values()
    ]
    agree_counts = [
        counter["shared"] + counter["decoy"] for counter in by_worker_agree.values()
    ]
    disagree_trials_per_worker = rounded_median(disagree_counts)
    agree_trials_per_worker = rounded_median(agree_counts)

    disagree_k, disagree_fpr = determine_classification_threshold(
        disagree_trials_per_worker, p_guess, alpha
    )
    agree_k, agree_fpr = determine_classification_threshold(
        agree_trials_per_worker, 1.0 / 3.0, alpha
    )

    disagree_classes: Counter[str] = Counter()
    agree_classes: Counter[str] = Counter()
    for worker in workers:
        disagree_counter = by_worker_disagree[worker]
        agree_counter = by_worker_agree[worker]
        if disagree_counter["nash"] >= disagree_k:
            disagree_classes["Nash"] += 1
        elif disagree_counter["ie"] >= disagree_k:
            disagree_classes["IE"] += 1
        elif disagree_counter["random"] >= disagree_k:
            disagree_classes["Random"] += 1
        else:
            disagree_classes["Indeterminate"] += 1

        if agree_counter["shared"] >= agree_k:
            agree_classes["Principled"] += 1
        elif agree_counter["decoy"] >= agree_k:
            agree_classes["Decoy"] += 1
        else:
            agree_classes["Indeterminate"] += 1

    disagree_p_log10 = binomial_tail_log_probability(
        len(workers), disagree_fpr, disagree_classes["Nash"]
    ) / math.log(10.0)
    agree_p_log10 = binomial_tail_log_probability(
        len(workers), agree_fpr, agree_classes["Principled"]
    ) / math.log(10.0)

    return Summary(
        workers=len(workers),
        random_rate=random_rate,
        p_guess=p_guess,
        disagree_k=disagree_k,
        disagree_fpr=disagree_fpr,
        nash_classified=disagree_classes["Nash"],
        ie_classified=disagree_classes["IE"],
        random_classified=disagree_classes["Random"],
        disagree_indeterminate=disagree_classes["Indeterminate"],
        disagree_p_log10=disagree_p_log10,
        agree_k=agree_k,
        agree_fpr=agree_fpr,
        principled_classified=agree_classes["Principled"],
        decoy_classified=agree_classes["Decoy"],
        agree_indeterminate=agree_classes["Indeterminate"],
        agree_p_log10=agree_p_log10,
    )


def format_probability_from_log10(log10_probability: float) -> str:
    """Format a tail probability from its base-10 logarithm.

    Parameters:
        log10_probability: Base-10 log probability.

    Returns:
        A scientific-notation string.
    """

    if log10_probability == float("-inf"):
        return "0"
    exponent = math.floor(log10_probability)
    mantissa = 10 ** (log10_probability - exponent)
    if mantissa >= 9.995:
        mantissa = 1.0
        exponent += 1
    return f"{mantissa:.2f}e{exponent}"


def print_summary(label: str, summary: Summary) -> None:
    """Print one human-readable summary row.

    Parameters:
        label: Subset label, such as ``overall`` or one alpha bin.
        summary: The computed summary statistics.

    Returns:
        None.
    """

    print(
        "\t".join(
            [
                label,
                str(summary.workers),
                f"{summary.random_rate:.4f}",
                str(summary.disagree_k),
                f"{summary.disagree_fpr:.6f}",
                str(summary.nash_classified),
                str(summary.ie_classified),
                str(summary.random_classified),
                str(summary.disagree_indeterminate),
                format_probability_from_log10(summary.disagree_p_log10),
                str(summary.agree_k),
                f"{summary.agree_fpr:.6f}",
                str(summary.principled_classified),
                str(summary.decoy_classified),
                str(summary.agree_indeterminate),
                format_probability_from_log10(summary.agree_p_log10),
            ]
        )
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments for the script.
    """

    parser = argparse.ArgumentParser(
        description="Compute preregistered classification exact binomial tests."
    )
    parser.add_argument("--csv-path", required=True, help="Path to the wide CSV file.")
    parser.add_argument(
        "--qualification-answers",
        required=True,
        help="Path to the qualification answer key JSON.",
    )
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=18,
        help="Number of scenarios shown per participant (default: 18).",
    )
    parser.add_argument(
        "--false-positive-alpha",
        type=float,
        default=0.05,
        help="Alpha used when deriving classification thresholds (default: 0.05).",
    )
    parser.add_argument(
        "--attention-threshold",
        type=float,
        default=1.0,
        help="Minimum worker-level attention pass rate (default: 1.0).",
    )
    parser.add_argument(
        "--skip-qualification",
        action="store_true",
        help="Do not apply the area qualification filter.",
    )
    parser.add_argument(
        "--skip-attention",
        action="store_true",
        help="Do not apply the attention filter.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the preregistered classification-test summary script."""

    args = parse_args()
    header, rows = load_rows(args.csv_path)
    if not args.skip_qualification:
        answers = load_qualification_answers(args.qualification_answers)
        qualification_columns = [
            column for column in header if "q_question-stacked" in column
        ]
        rows = [
            row
            for row in rows
            if passes_area_qualification(row, answers, qualification_columns)
        ]
    exploded = explode_rows(rows, args.num_scenarios)
    if not args.skip_attention:
        exploded = filter_by_attention(exploded, args.attention_threshold)

    print(
        "\t".join(
            [
                "label",
                "workers",
                "random_rate",
                "disagree_k",
                "disagree_fpr",
                "nash_classified",
                "ie_classified",
                "random_classified",
                "disagree_indeterminate",
                "disagree_p",
                "agree_k",
                "agree_fpr",
                "principled_classified",
                "decoy_classified",
                "agree_indeterminate",
                "agree_p",
            ]
        )
    )
    print_summary("overall", summarize_subset(exploded, args.false_positive_alpha))

    by_alpha: dict[str, list[dict[str, str | bool]]] = defaultdict(list)
    for row in exploded:
        by_alpha[str(row["alpha"])].append(row)
    for alpha_label in sorted(by_alpha, key=float):
        print_summary(
            alpha_label,
            summarize_subset(by_alpha[alpha_label], args.false_positive_alpha),
        )


if __name__ == "__main__":
    main()
