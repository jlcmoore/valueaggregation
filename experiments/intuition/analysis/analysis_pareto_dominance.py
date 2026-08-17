"""Audit Pareto dominance and human choices in IE-Nash disagreements."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
from scipy.stats import wilcoxon

# isort: split
from common.qualification_utils import (
    load_qualification_answers,
    qualification_passed_row,
)
from common.shared_analysis import mturk_explode_df

import value_aggregation as aggregation

DEFAULT_SCENARIO_FILE = (
    "data/scenarios/maximize=True_num-agents=3_belief-steps=1_"
    "belief-range=3,3_action-steps=101_action-range=1,101_"
    "action-function-log=False_prevent-ties=True_"
    "agg-functions=['fehr','nash']_disagrees-only=False_"
    "num-scenarios=18_sample-size=150.csv"
)
DEFAULT_RESPONSE_FILE = (
    "data/results/mturk/jatos_data_condition=best compromise-area.csv"
)


@dataclass(frozen=True)
class ScenarioAudit:
    """Store dominance properties for one alpha-specific scenario.

    Parameters:
        alpha: IE inequality-aversion parameter used for the scenario.
        scenario_json: Encoded game state used to identify the scenario.
        ie_dominated: Whether another proposal Pareto dominates the IE proposal.
    """

    alpha: float
    scenario_json: str
    ie_dominated: bool


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments and return the configured namespace."""
    parser = argparse.ArgumentParser(
        description=(
            "Count Pareto-dominated IE disagreement options and compare human "
            "Nash and IE choices within dominance strata."
        )
    )
    parser.add_argument("--scenario-file", default=DEFAULT_SCENARIO_FILE)
    parser.add_argument("--response-file", default=DEFAULT_RESPONSE_FILE)
    parser.add_argument("--num-scenarios", type=int, default=18)
    parser.add_argument("--attention-threshold", type=float, default=1.0)
    parser.add_argument(
        "--qualification-type",
        choices=("area", "volume", "any"),
        default="area",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="Optional directory for the two CSV summary tables.",
    )
    return parser.parse_args()


def pareto_dominates(candidate: Sequence[float], target: Sequence[float]) -> bool:
    """Return whether candidate weakly improves every value and strictly one.

    Parameters:
        candidate: Outcomes of the potentially dominating proposal.
        target: Outcomes of the proposal being evaluated.

    Returns:
        True when candidate Pareto dominates target, otherwise False.
    """
    weakly_better = all(
        candidate_value >= target_value
        for candidate_value, target_value in zip(candidate, target)
    )
    strictly_better = any(
        candidate_value > target_value
        for candidate_value, target_value in zip(candidate, target)
    )
    return weakly_better and strictly_better


def proposal_outcomes(scenario_json: str) -> dict[str, tuple[float, ...]]:
    """Decode proposal outcomes from a serialized game state.

    Parameters:
        scenario_json: Encoded game state from the scenario CSV.

    Returns:
        A mapping from proposal name to its ordered outcome tuple.
    """
    game_state = aggregation.decode_gameState(scenario_json)
    return {
        str(proposal): tuple(float(value) for value in values)
        for proposal, values in game_state.vote_to_outcomes().items()
    }


def audit_scenarios(scenario_frame: pd.DataFrame, num_scenarios: int) -> pd.DataFrame:
    """Return one dominance record per unique alpha-specific disagreement.

    Parameters:
        scenario_frame: Wide scenario dataframe used for the experiment.
        num_scenarios: Number of scenario slots in each participant row.

    Returns:
        Dataframe with alpha, serialized scenario, and IE dominance status.
    """
    records: dict[tuple[float, str], ScenarioAudit] = {}
    for _, row in scenario_frame.iterrows():
        for scenario_number in range(1, num_scenarios + 1):
            prefix = f"scenario_{scenario_number}_"
            ie_choice = str(row[prefix + "ie"])
            nash_choice = str(row[prefix + "nbs"])
            if ie_choice == nash_choice:
                continue

            alpha = float(row[prefix + "alpha_bin"])
            scenario_json = str(row[prefix + "json"])
            outcomes = proposal_outcomes(scenario_json)
            ie_outcomes = outcomes[ie_choice]
            alternatives = [
                values for proposal, values in outcomes.items() if proposal != ie_choice
            ]
            records[(alpha, scenario_json)] = ScenarioAudit(
                alpha=alpha,
                scenario_json=scenario_json,
                ie_dominated=any(
                    pareto_dominates(values, ie_outcomes) for values in alternatives
                ),
            )

    return pd.DataFrame(
        {
            "alpha": record.alpha,
            "scenario_json": record.scenario_json,
            "ie_dominated": record.ie_dominated,
        }
        for record in records.values()
    )


def qualification_columns(
    frame: pd.DataFrame,
    answers: Mapping[str, object],
    qualification_type: str,
) -> list[str]:
    """Return answer columns used for the requested qualification condition.

    Parameters:
        frame: Wide response dataframe.
        answers: Expected qualification answers keyed by column.
        qualification_type: One of area, volume, or any.

    Returns:
        Qualification columns present in the response dataframe.
    """
    if qualification_type == "area":
        marker = "q_question-stacked"
    elif qualification_type == "volume":
        marker = "q_question-3D"
    else:
        marker = "q_question-"
    return [
        column for column in answers if marker in column and column in frame.columns
    ]


def filter_human_responses(
    response_frame: pd.DataFrame,
    num_scenarios: int,
    attention_threshold: float,
    qualification_type: str,
) -> pd.DataFrame:
    """Apply the main qualification and worker-level attention filters.

    Parameters:
        response_frame: Wide human response export.
        num_scenarios: Number of scenario slots in each participant row.
        attention_threshold: Minimum worker-level attention-check accuracy.
        qualification_type: Qualification condition to evaluate.

    Returns:
        Filtered long dataframe with one row per participant and scenario.
    """
    answers = load_qualification_answers()
    columns = qualification_columns(response_frame, answers, qualification_type)
    qualification_passes = response_frame.apply(
        qualification_passed_row,
        axis=1,
        args=(columns, answers),
    )
    qualified = response_frame[qualification_passes].copy()
    long_frame = mturk_explode_df(qualified, num_scenarios)
    long_frame["passed_attention"] = (
        long_frame["attention-response"] == long_frame["attention-answer"]
    )
    attention_rates = long_frame.groupby("WorkerId")["passed_attention"].mean()
    passed_workers = attention_rates[attention_rates >= attention_threshold].index
    return long_frame[long_frame["WorkerId"].isin(passed_workers)].copy()


def scenario_count_table(audit_frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize unique dominated and undominated disagreements by alpha.

    Parameters:
        audit_frame: Unique-scenario dominance audit.

    Returns:
        Dataframe with one row per alpha and dominance stratum.
    """
    table = (
        audit_frame.groupby(["alpha", "ie_dominated"])
        .size()
        .rename("unique_scenarios")
        .reset_index()
    )
    alpha_values = sorted(audit_frame["alpha"].unique())
    complete_index = pd.MultiIndex.from_product(
        [alpha_values, [False, True]], names=["alpha", "ie_dominated"]
    )
    return (
        table.set_index(["alpha", "ie_dominated"])
        .reindex(complete_index, fill_value=0)
        .reset_index()
    )


def wilcoxon_summary(group: pd.DataFrame) -> pd.Series:
    """Summarize choices and a one-sided participant-level Wilcoxon test.

    Parameters:
        group: Trial presentations from one alpha and dominance stratum.

    Returns:
        Series containing choice rates, participant counts, W, and p.
    """
    worker_counts = group.groupby("WorkerId")[["choose_nash", "choose_ie"]].sum()
    differences = worker_counts["choose_nash"] - worker_counts["choose_ie"]
    nonzero_differences = differences[differences != 0]
    if nonzero_differences.empty:
        statistic = float("nan")
        p_value = float("nan")
    else:
        result = wilcoxon(
            nonzero_differences,
            alternative="greater",
            zero_method="wilcox",
        )
        statistic = float(result.statistic)
        p_value = float(result.pvalue)

    return pd.Series(
        {
            "presentations": len(group),
            "participants": len(worker_counts),
            "nonzero_participants": len(nonzero_differences),
            "nash_rate": group["choose_nash"].mean(),
            "ie_rate": group["choose_ie"].mean(),
            "other_rate": group["choose_other"].mean(),
            "wilcoxon_w": statistic,
            "wilcoxon_p": p_value,
        }
    )


def response_test_table(
    filtered_responses: pd.DataFrame, audit_frame: pd.DataFrame
) -> pd.DataFrame:
    """Create dominance-stratified response summaries and Wilcoxon tests.

    Parameters:
        filtered_responses: Filtered long-form human responses.
        audit_frame: Unique-scenario dominance audit.

    Returns:
        Dataframe with one row per observed alpha and dominance stratum.
    """
    disagreements = filtered_responses[
        filtered_responses["scenario_ie"] != filtered_responses["scenario_nbs"]
    ].copy()
    disagreements["alpha"] = disagreements["scenario_alpha_bin"].astype(float)
    audit_lookup = audit_frame.rename(columns={"scenario_json": "audit_scenario_json"})
    disagreements = disagreements.merge(
        audit_lookup,
        left_on=["alpha", "scenario_json"],
        right_on=["alpha", "audit_scenario_json"],
        how="left",
        validate="many_to_one",
    )
    if disagreements["ie_dominated"].isna().any():
        raise ValueError("Some disagreement responses lacked a dominance audit.")

    disagreements["choose_nash"] = (
        disagreements["response"] == disagreements["scenario_nbs"]
    )
    disagreements["choose_ie"] = (
        disagreements["response"] == disagreements["scenario_ie"]
    )
    disagreements["choose_other"] = ~(
        disagreements["choose_nash"] | disagreements["choose_ie"]
    )
    return (
        disagreements.groupby(["alpha", "ie_dominated"], sort=True)
        .apply(wilcoxon_summary)
        .reset_index()
    )


def write_outputs(
    count_table: pd.DataFrame,
    test_table: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Write reproducible scenario-count and choice-test CSV tables.

    Parameters:
        count_table: Unique scenario counts by alpha and dominance.
        test_table: Human choice summaries and significance tests.
        output_directory: Destination directory for both CSV files.

    Returns:
        None.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    count_table.to_csv(
        output_directory / "pareto_dominance_scenario_counts.csv", index=False
    )
    test_table.to_csv(
        output_directory / "pareto_dominance_choice_tests.csv", index=False
    )


def main() -> None:
    """Run the dominance audit, print summaries, and optionally write CSVs."""
    args = parse_args()
    scenario_frame = pd.read_csv(args.scenario_file)
    response_frame = pd.read_csv(args.response_file)
    audit_frame = audit_scenarios(scenario_frame, args.num_scenarios)
    filtered_responses = filter_human_responses(
        response_frame,
        args.num_scenarios,
        args.attention_threshold,
        args.qualification_type,
    )
    count_table = scenario_count_table(audit_frame)
    test_table = response_test_table(filtered_responses, audit_frame)

    print(f"Filtered participants: {filtered_responses['WorkerId'].nunique()}")
    print("\nUnique disagreement scenarios")
    print(count_table.to_string(index=False))
    print("\nDominance-stratified human choices and tests")
    print(test_table.to_string(index=False))

    if args.output_directory is not None:
        write_outputs(count_table, test_table, args.output_directory)


if __name__ == "__main__":
    main()
