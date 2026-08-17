"""Prepare MTurk or LLM CSVs in-memory and run the brms analysis."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import tempfile
from typing import Dict, Iterable, List, Tuple

from common.qualification_utils import (
    load_qualification_answers,
    qualification_columns_in_header,
    qualification_passed_row,
)

SCENARIO_ORDER = ["wait-times", "medical-costs", "life-expectancy", "travel-times"]
LLM_REQUIRED_COLUMNS = {
    "scenario_hash",
    "scenario_nbs",
    "response",
    "question",
}


def parse_args() -> tuple[argparse.Namespace, List[str]]:
    """Parse arguments and preserve extra flags for the R script."""
    parser = argparse.ArgumentParser(
        description="Run brms analysis from MTurk or LLM CSVs."
    )
    parser.add_argument(
        "--data",
        action="append",
        required=True,
        help="Raw MTurk CSV path(s), can be repeated or comma-separated.",
    )
    parser.add_argument(
        "--analysis",
        required=True,
        choices=["disagree", "agree"],
        help="Analysis mode passed through to the R script.",
    )
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=None,
        help="Number of scenarios per HIT; inferred if omitted.",
    )
    parser.add_argument(
        "--scenario-type",
        default="unknown",
        help="Fallback scenario_type label when mapping is unavailable.",
    )
    parser.add_argument(
        "--keep-combined",
        action="store_true",
        help="Keep the combined CSV file for inspection.",
    )
    parser.add_argument(
        "--combined-path",
        default=None,
        help="Optional path for the combined CSV.",
    )
    parser.add_argument(
        "--qualification-status",
        default="any",
        choices=["any", "pass", "fail"],
        help="Filter human rows by qualificationPassed (pass/fail).",
    )
    options, extra_args = parser.parse_known_args()
    return options, extra_args


def expand_paths(paths: Iterable[str]) -> List[str]:
    """Expand comma-separated data arguments into a flat list."""
    expanded: List[str] = []
    for entry in paths:
        if os.path.exists(entry):
            expanded.append(entry)
            continue
        expanded.extend([part.strip() for part in entry.split(",") if part.strip()])
    return expanded


def infer_num_scenarios(fieldnames: Iterable[str]) -> int:
    """Infer the number of scenarios from input column names."""
    pattern = re.compile(r"^Input\.scenario_(\d+)_hash$")
    indices = set()
    for field in fieldnames:
        match = pattern.match(field)
        if match:
            indices.add(int(match.group(1)))
    if not indices:
        raise ValueError("Could not infer number of scenarios from input columns.")
    return max(indices)


def parse_condition_from_path(path: str) -> str:
    """Extract the condition label from the input filename."""
    match = re.search(r"chart_type=([^_./]+)", path)
    if match:
        return match.group(1)
    return "unknown"


def scenario_type_from_index(index: int, fallback: str) -> str:
    """Map scenario index to scenario type name."""
    mapping = {idx + 1: name for idx, name in enumerate(SCENARIO_ORDER)}
    return mapping.get(index, fallback)


def extract_alpha_columns(
    fieldnames: Iterable[str], index: int
) -> List[Tuple[str, float]]:
    """Collect fehr alpha columns for a given scenario index."""
    pattern = re.compile(rf"^Input\.scenario_{index}_fehr_alpha_(\d+_\d+)$")
    results: List[Tuple[str, float]] = []
    for field in fieldnames:
        match = pattern.match(field)
        if not match:
            continue
        alpha_str = match.group(1).replace("_", ".")
        alpha_value = float(alpha_str)
        results.append((field, alpha_value))
    return sorted(results, key=lambda item: item[1])


def extract_llm_alpha_columns(fieldnames: Iterable[str]) -> List[Tuple[str, float]]:
    """Collect fehr alpha columns for LLM inputs."""

    pattern = re.compile(r"^scenario_fehr_alpha_(\d+_\d+)$")
    results: List[Tuple[str, float]] = []
    for field in fieldnames:
        match = pattern.match(field)
        if not match:
            continue
        alpha_str = match.group(1).replace("_", ".")
        alpha_value = float(alpha_str)
        results.append((field, alpha_value))
    return sorted(results, key=lambda item: item[1])


def determine_choice_label(
    response: str,
    nash_action: str,
    ie_action: str,
    labels: Dict[str, str],
) -> Tuple[str, str]:
    """Map a response action to a choice label and agreement status."""
    if nash_action == ie_action:
        if response == nash_action:
            return labels["nash_ie"], "agree"
        return labels["decoy"], "agree"
    if response == nash_action:
        return labels["nash"], "disagree"
    if response == ie_action:
        return labels["ie"], "disagree"
    return labels["decoy"], "disagree"


def build_rows(
    row: Dict[str, str],
    num_scenarios: int,
    condition: str,
    fallback_scenario_type: str,
    fieldnames: Iterable[str],
    qualification_status: str,
    qual_columns: List[str],
    qual_answers: Dict[str, object],
) -> List[Dict[str, str]]:
    """Expand a single MTurk assignment row into brms-ready rows."""
    rows: List[Dict[str, str]] = []
    participant_id = str(row.get("WorkerId", "")).strip()
    labels = {
        "nash": "Nash",
        "ie": "IE",
        "decoy": "Decoy",
        "nash_ie": "Nash/IE",
    }
    if qualification_status != "any":
        passed = qualification_passed_row(row, qual_columns, qual_answers)
        if qualification_status == "pass" and not passed:
            return []
        if qualification_status == "fail" and passed:
            return []

    for index in range(1, num_scenarios + 1):
        response = str(row.get(f"Answer.q_question-{index}", "")).strip()
        if not response:
            continue
        nash_action = str(row.get(f"Input.scenario_{index}_nbs", "")).strip()
        scenario_id = str(row.get(f"Input.scenario_{index}_hash", "")).strip()

        if not nash_action:
            continue

        alpha_columns = extract_alpha_columns(fieldnames, index)
        if not alpha_columns:
            raise ValueError(
                "Missing fehr_alpha columns for scenario index "
                f"{index}. Ensure scenario_{index}_fehr_alpha_* is present."
            )

        scenario_type = scenario_type_from_index(index, fallback_scenario_type)

        for alpha_column, alpha_value in alpha_columns:
            ie_alpha_action = str(row.get(alpha_column, "")).strip()
            if not ie_alpha_action:
                continue
            choice_label, agreement = determine_choice_label(
                response=response,
                nash_action=nash_action,
                ie_action=ie_alpha_action,
                labels=labels,
            )
            rows.append(
                {
                    "choice": choice_label,
                    "condition": condition,
                    "scenario_type": scenario_type,
                    "alpha": alpha_value,
                    "participant_id": participant_id,
                    "scenario_id": scenario_id,
                    "agreement": agreement,
                }
            )
    return rows


def detect_input_type(fieldnames: Iterable[str]) -> str:
    """Detect whether the input is human or LLM formatted."""

    fieldset = set(fieldnames)
    if any(name.startswith("Input.scenario_") for name in fieldset):
        return "human"
    if LLM_REQUIRED_COLUMNS.issubset(fieldset):
        return "llm"
    return "unknown"


def build_rows_from_llm(
    row: Dict[str, str],
    *,
    condition: str,
    fallback_scenario_type: str,
    fieldnames: Iterable[str],
    qualification_status: str,
) -> List[Dict[str, str]]:
    """Expand a single LLM row into brms-ready rows."""
    if qualification_status != "any":
        raise ValueError("Qualification filtering is only supported for human data.")

    response = str(row.get("response", "")).strip()
    if not response:
        return []

    nash_action = str(row.get("scenario_nbs", "")).strip()
    scenario_id = str(row.get("scenario_hash", "")).strip()
    if not nash_action or not scenario_id:
        return []

    question_raw = row.get("question", "")
    try:
        question_index = int(float(question_raw))
    except (TypeError, ValueError):
        question_index = 0

    scenario_type = scenario_type_from_index(question_index, fallback_scenario_type)

    alpha_columns = extract_llm_alpha_columns(fieldnames)
    if not alpha_columns:
        raise ValueError(
            "Missing scenario_fehr_alpha_* columns for LLM input. "
            "Ensure the LLM output includes Fehr alpha columns."
        )

    labels = {
        "nash": "Nash",
        "ie": "IE",
        "decoy": "Decoy",
        "nash_ie": "Nash/IE",
    }
    participant_id = str(row.get("participant_id", "")).strip()
    if not participant_id:
        participant_id = f"llm_{scenario_id}_{question_index}"

    rows: List[Dict[str, str]] = []
    for alpha_column, alpha_value in alpha_columns:
        ie_action = str(row.get(alpha_column, "")).strip()
        if not ie_action:
            continue
        choice_label, agreement = determine_choice_label(
            response=response,
            nash_action=nash_action,
            ie_action=ie_action,
            labels=labels,
        )
        rows.append(
            {
                "choice": choice_label,
                "condition": condition,
                "scenario_type": scenario_type,
                "alpha": alpha_value,
                "participant_id": participant_id,
                "scenario_id": scenario_id,
                "agreement": agreement,
            }
        )
    return rows


def write_combined_csv(
    paths: List[str],
    combined_path: str,
    num_scenarios: int | None,
    scenario_type: str,
    qualification_status: str,
) -> None:
    """Write a combined brms-ready CSV from MTurk or LLM files."""
    fieldnames_out = [
        "choice",
        "condition",
        "scenario_type",
        "alpha",
        "participant_id",
        "scenario_id",
        "agreement",
    ]
    qual_answers = load_qualification_answers()
    with open(combined_path, "w", newline="", encoding="utf-8") as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=fieldnames_out)
        writer.writeheader()

        for path in paths:
            with open(path, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    raise ValueError(f"CSV file has no header: {path}")
                condition = parse_condition_from_path(path)
                input_type = detect_input_type(reader.fieldnames)
                qual_columns: List[str] = []
                if input_type == "human" and qualification_status != "any":
                    qual_columns, _ = qualification_columns_in_header(
                        reader.fieldnames, qual_answers
                    )
                    if not qual_columns:
                        raise ValueError(
                            "Missing qualification answer columns; "
                            "cannot recompute qualification status."
                        )
                if input_type == "human":
                    scenarios = num_scenarios or infer_num_scenarios(reader.fieldnames)
                    for row in reader:
                        for out_row in build_rows(
                            row=row,
                            num_scenarios=scenarios,
                            condition=condition,
                            fallback_scenario_type=scenario_type,
                            fieldnames=reader.fieldnames,
                            qualification_status=qualification_status,
                            qual_columns=qual_columns,
                            qual_answers=qual_answers,
                        ):
                            writer.writerow(out_row)
                elif input_type == "llm":
                    for row in reader:
                        for out_row in build_rows_from_llm(
                            row=row,
                            condition=condition,
                            fallback_scenario_type=scenario_type,
                            fieldnames=reader.fieldnames,
                            qualification_status=qualification_status,
                        ):
                            writer.writerow(out_row)
                else:
                    raise ValueError(
                        f"Unrecognized input format for {path}; "
                        "expected human or LLM CSV."
                    )


def infer_persuader_type(paths: Iterable[str]) -> str:
    """Infer whether inputs are human, LLM, or mixed."""
    input_types: set[str] = set()
    unknown_paths: List[str] = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"CSV file has no header: {path}")
            input_type = detect_input_type(reader.fieldnames)
            if input_type == "unknown":
                unknown_paths.append(path)
            input_types.add(input_type)

    if input_types == {"human"}:
        return "human"
    if input_types == {"llm"}:
        return "llm"
    if "unknown" in input_types:
        hint = ""
        if unknown_paths:
            hint = (
                " Unrecognized inputs: "
                + ", ".join(unknown_paths)
                + ". brms_run.py expects results CSVs with responses "
                "(e.g., data/results/mturk, data/results/jatos, data/results/llm), "
                "not scenario definition files."
            )
        raise ValueError(f"Could not infer persuader type from input data.{hint}")
    return "mixed"


def has_flag(extra_args: List[str], flag: str) -> bool:
    """Check whether a flag is already present in extra args."""
    flag_prefix = f"{flag}="
    return any(arg == flag or arg.startswith(flag_prefix) for arg in extra_args)


def run_rscript(
    combined_path: str,
    analysis: str,
    extra_args: List[str],
    plot_file: str | None,
    fit_file: str | None,
) -> None:
    """Invoke the R analysis script with the combined data."""
    command = [
        "Rscript",
        "mixed_effects_analysis.R",
        "--data",
        combined_path,
        "--analysis",
        analysis,
    ]
    if plot_file and not has_flag(extra_args, "--plot-file"):
        command.extend(["--plot-file", plot_file])
    if fit_file and not has_flag(extra_args, "--fit-file"):
        command.extend(["--fit-file", fit_file])
    command.extend(extra_args)
    subprocess.run(command, check=True)


def main() -> None:
    """Entry point for combining raw MTurk data and running brms."""
    options, extra_args = parse_args()
    data_paths = expand_paths(options.data)
    if not data_paths:
        raise ValueError("No input data paths provided.")

    persuader_type = infer_persuader_type(data_paths)
    plot_file = f"alpha_probabilities_{options.analysis}_{persuader_type}.csv"
    fit_file = f"brms_fit_{options.analysis}_{persuader_type}.rds"

    if options.combined_path:
        combined_path = options.combined_path
    else:
        temp_dir = tempfile.mkdtemp(prefix="brms_data_")
        combined_path = os.path.join(temp_dir, "combined.csv")

    write_combined_csv(
        paths=data_paths,
        combined_path=combined_path,
        num_scenarios=options.num_scenarios,
        scenario_type=options.scenario_type,
        qualification_status=options.qualification_status,
    )
    run_rscript(combined_path, options.analysis, extra_args, plot_file, fit_file)

    if not options.keep_combined and not options.combined_path:
        os.remove(combined_path)
        os.rmdir(os.path.dirname(combined_path))


if __name__ == "__main__":
    main()
