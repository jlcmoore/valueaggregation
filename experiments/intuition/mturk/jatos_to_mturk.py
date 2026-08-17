"""Convert JATOS JSONL results into MTurk-style CSV output."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from typing import Dict, Iterable, List, Set, Tuple


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert JATOS JSONL results to MTurk-style CSV."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to JATOS results JSONL file.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output CSV path. Defaults to data/results/mturk/<input>.csv.",
    )
    parser.add_argument(
        "--scenario-file",
        default="",
        help="Scenario CSV file to compute coverage.",
    )
    parser.add_argument(
        "--desired-per-row",
        type=int,
        default=3,
        help="Target number of unique raters per scenario row.",
    )
    parser.add_argument(
        "--demographics-file",
        default="",
        help="Prolific demographics CSV export to join on PROLIFIC_PID.",
    )
    parser.add_argument(
        "--batch-session-file",
        default="",
        help="Existing JATOS batch session JSON to rebuild queues for.",
    )
    parser.add_argument(
        "--batch-session-output",
        default="",
        help="Output path for rebuilt batch session JSON.",
    )
    parser.add_argument(
        "--no-split-by-condition",
        action="store_true",
        help="Write a single CSV instead of splitting by condition.",
    )
    parser.add_argument(
        "--include-failed-qualification",
        action="store_true",
        help="Include rows where qualificationPassed is false.",
    )
    parser.add_argument(
        "--include-failed-attention",
        action="store_true",
        help="Include rows where attentionPassed is false.",
    )
    parser.add_argument(
        "--submission-cost",
        type=float,
        default=7.0,
        help="Base cost per submitted response before multiplier (default: 7.0).",
    )
    parser.add_argument(
        "--cost-multiplier",
        type=float,
        default=1.2,
        help="Cost multiplier applied to base submission cost (default: 1.2).",
    )
    parser.add_argument(
        "--target-included",
        type=int,
        default=0,
        help=(
            "Target number of included responses. If 0, defaults to "
            "desired-per-row * total scenario rows."
        ),
    )
    parser.add_argument(
        "--qualification-rule",
        choices=["stored", "area_first_three"],
        default="stored",
        help=(
            "Qualification rule for the main coverage/cost summary: "
            "'stored' uses qualificationPassed, "
            "'area_first_three' recomputes from q_question-stacked-1/2/3."
        ),
    )
    return parser.parse_args()


def iter_jatos_rows(path: str) -> Iterable[Dict[str, object]]:
    """Yield parsed JATOS rows that include both row and answers."""
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError:
                print(
                    f"Skipping line {line_number}: invalid JSON.",
                    file=sys.stderr,
                )
                continue
            if not isinstance(data, dict):
                continue
            row = data.get("row")
            answers = data.get("answers")
            if not isinstance(row, dict) or not isinstance(answers, dict):
                continue
            yield data


def load_demographics(
    path: str,
) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """Load demographics keyed by Prolific participant id."""
    demographics: Dict[str, Dict[str, str]] = {}
    columns: List[str] = []
    if not path:
        return demographics, columns
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        for row in reader:
            participant_id = (row.get("Participant id") or "").strip()
            if participant_id:
                demographics[participant_id] = row
    return demographics, columns


def count_scenario_rows(path: str) -> int:
    """Count the number of data rows in a scenario CSV file."""
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        row_count = -1
        for row_count, _row in enumerate(reader):
            pass
    if row_count < 0:
        return 0
    return row_count


def count_export_rows(path: str) -> int:
    """Infer number of rows from JATOS export rowIndex values."""
    max_index = -1
    for data in iter_jatos_rows(path):
        row_index = data.get("rowIndex")
        if row_index is None:
            continue
        try:
            row_index_int = int(row_index)
        except (TypeError, ValueError):
            continue
        if row_index_int > max_index:
            max_index = row_index_int
    return max_index + 1 if max_index >= 0 else 0


def compute_coverage(
    input_path: str,
    scenario_path: str,
    desired_per_row: int,
    require_qualification: bool,
    require_attention: bool,
    conditions: List[str] | None = None,
) -> Dict[str, float]:
    """Compute coverage statistics for scenario rows."""
    seen: Dict[int, Set[str]] = {}
    seen_by_condition: Dict[str, Dict[int, Set[str]]] = {}
    for data in iter_jatos_rows(input_path):
        if require_qualification and not data.get("qualificationPassed", False):
            continue
        if require_attention and not data.get("attentionPassed", False):
            continue
        condition = data.get("assignedCondition", "")
        if not condition:
            answers = data.get("answers", {})
            if isinstance(answers, dict):
                condition = answers.get("condition_name", "")
        if not isinstance(condition, str):
            condition = ""
        condition = condition.strip()
        row_index = data.get("rowIndex")
        if row_index is None:
            continue
        try:
            row_index_int = int(row_index)
        except (TypeError, ValueError):
            continue
        meta = data.get("meta", {})
        worker_id = ""
        if isinstance(meta, dict):
            worker_id = str(meta.get("workerId", "")).strip()
        if worker_id == "":
            continue
        seen.setdefault(row_index_int, set()).add(worker_id)
        if condition:
            by_cond = seen_by_condition.setdefault(condition, {})
            by_cond.setdefault(row_index_int, set()).add(worker_id)

    total_rows = count_scenario_rows(scenario_path)
    unique_counts = [len(workers) for workers in seen.values()]
    num_with_any = len(unique_counts)
    num_with_target = sum(1 for count in unique_counts if count >= desired_per_row)
    avg_count = sum(unique_counts) / num_with_any if num_with_any else 0.0
    min_count = min(unique_counts) if unique_counts else 0.0
    max_count = max(unique_counts) if unique_counts else 0.0
    target_rate = num_with_target / total_rows if total_rows else 0.0
    missing_overall = (
        sum(max(0, desired_per_row - count) for count in unique_counts)
        + max(0, total_rows - num_with_any) * desired_per_row
    )

    stats = {
        "total_rows": float(total_rows),
        "rows_with_any": float(num_with_any),
        "rows_with_target": float(num_with_target),
        "target_rate": target_rate,
        "avg_per_row_with_any": avg_count,
        "min_per_row_with_any": min_count,
        "max_per_row_with_any": max_count,
        "missing_slots": float(missing_overall),
    }
    condition_list = conditions or sorted(seen_by_condition.keys())
    for condition in condition_list:
        rows = seen_by_condition.get(condition, {})
        counts = [len(workers) for workers in rows.values()]
        rows_with_any = len(counts)
        rows_with_target = sum(1 for count in counts if count >= desired_per_row)
        avg_count = sum(counts) / rows_with_any if rows_with_any else 0.0
        min_count = min(counts) if counts else 0.0
        max_count = max(counts) if counts else 0.0
        target_rate = rows_with_target / total_rows if total_rows else 0.0
        missing_slots = (
            sum(max(0, desired_per_row - count) for count in counts)
            + max(0, total_rows - rows_with_any) * desired_per_row
        )
        stats[f"condition.{condition}.total_rows"] = float(total_rows)
        stats[f"condition.{condition}.rows_with_any"] = float(rows_with_any)
        stats[f"condition.{condition}.rows_with_target"] = float(rows_with_target)
        stats[f"condition.{condition}.target_rate"] = target_rate
        stats[f"condition.{condition}.avg_per_row_with_any"] = avg_count
        stats[f"condition.{condition}.min_per_row_with_any"] = min_count
        stats[f"condition.{condition}.max_per_row_with_any"] = max_count
        stats[f"condition.{condition}.missing_slots"] = float(missing_slots)
    return stats


def compute_coverage_by_rule(
    input_path: str,
    scenario_path: str,
    desired_per_row: int,
    attention_threshold: float,
    use_area_first_three_qualification: bool = False,
    qualification_answers_path: str = "mturk/qualification_answers.json",
    conditions: List[str] | None = None,
) -> Dict[str, float]:
    """Compute coverage statistics using a custom qualification rule."""
    expected_answers: Dict[str, object] = {}
    if use_area_first_three_qualification:
        expected_answers = load_raw_qualification_answers(qualification_answers_path)

    seen: Dict[int, Set[str]] = {}
    seen_by_condition: Dict[str, Dict[int, Set[str]]] = {}
    for data in iter_jatos_rows(input_path):
        if not submission_passes_rule(
            data,
            expected_answers,
            use_area_first_three_qualification,
            attention_threshold,
        ):
            continue

        condition = get_condition(data)
        row_index = data.get("rowIndex")
        if row_index is None:
            continue
        try:
            row_index_int = int(row_index)
        except (TypeError, ValueError):
            continue
        meta = data.get("meta", {})
        worker_id = ""
        if isinstance(meta, dict):
            worker_id = str(meta.get("workerId", "")).strip()
        if not worker_id:
            continue
        seen.setdefault(row_index_int, set()).add(worker_id)
        if condition:
            seen_by_condition.setdefault(condition, {}).setdefault(
                row_index_int, set()
            ).add(worker_id)

    total_rows = count_scenario_rows(scenario_path)
    if total_rows == 0:
        total_rows = count_export_rows(input_path)
    unique_counts = [len(workers) for workers in seen.values()]
    num_with_any = len(unique_counts)
    num_with_target = sum(1 for count in unique_counts if count >= desired_per_row)
    avg_count = sum(unique_counts) / num_with_any if num_with_any else 0.0
    min_count = min(unique_counts) if unique_counts else 0.0
    max_count = max(unique_counts) if unique_counts else 0.0
    target_rate = num_with_target / total_rows if total_rows else 0.0
    missing_overall = (
        sum(max(0, desired_per_row - count) for count in unique_counts)
        + max(0, total_rows - num_with_any) * desired_per_row
    )

    stats = {
        "total_rows": float(total_rows),
        "rows_with_any": float(num_with_any),
        "rows_with_target": float(num_with_target),
        "target_rate": target_rate,
        "avg_per_row_with_any": avg_count,
        "min_per_row_with_any": min_count,
        "max_per_row_with_any": max_count,
        "missing_slots": float(missing_overall),
    }
    condition_list = conditions or sorted(seen_by_condition.keys())
    for condition in condition_list:
        rows = seen_by_condition.get(condition, {})
        counts = [len(workers) for workers in rows.values()]
        rows_with_any = len(counts)
        rows_with_target = sum(1 for count in counts if count >= desired_per_row)
        avg_count = sum(counts) / rows_with_any if rows_with_any else 0.0
        min_count = min(counts) if counts else 0.0
        max_count = max(counts) if counts else 0.0
        target_rate = rows_with_target / total_rows if total_rows else 0.0
        missing_slots = (
            sum(max(0, desired_per_row - count) for count in counts)
            + max(0, total_rows - rows_with_any) * desired_per_row
        )
        stats[f"condition.{condition}.total_rows"] = float(total_rows)
        stats[f"condition.{condition}.rows_with_any"] = float(rows_with_any)
        stats[f"condition.{condition}.rows_with_target"] = float(rows_with_target)
        stats[f"condition.{condition}.target_rate"] = target_rate
        stats[f"condition.{condition}.avg_per_row_with_any"] = avg_count
        stats[f"condition.{condition}.min_per_row_with_any"] = min_count
        stats[f"condition.{condition}.max_per_row_with_any"] = max_count
        stats[f"condition.{condition}.missing_slots"] = float(missing_slots)
    return stats


def submission_passes_rule(
    data: Dict[str, object],
    expected_answers: Dict[str, object],
    use_area_first_three_qualification: bool,
    attention_threshold: float,
) -> bool:
    """Return whether one submission passes the requested qual+attention rule."""
    answers = data.get("answers", {})
    if not isinstance(answers, dict):
        return False
    if use_area_first_three_qualification:
        qual_pass = area_first_three_qualification_passed(answers, expected_answers)
    else:
        qual_pass = bool(data.get("qualificationPassed", False))
    passed, total = attention_counts_from_answers(answers)
    attention_rate = (passed / total) if total > 0 else 0.0
    return qual_pass and attention_rate >= attention_threshold


def compute_missing_slots_by_rule(
    input_path: str,
    scenario_path: str,
    desired_per_row: int,
    attention_threshold: float,
    use_area_first_three_qualification: bool = False,
    qualification_answers_path: str = "mturk/qualification_answers.json",
    conditions: List[str] | None = None,
) -> Dict[str, float]:
    """Compute remaining per-row coverage slots under a qual+attention rule."""
    expected_answers: Dict[str, object] = {}
    if use_area_first_three_qualification:
        expected_answers = load_raw_qualification_answers(qualification_answers_path)

    seen: Dict[int, Set[str]] = {}
    seen_by_condition: Dict[str, Dict[int, Set[str]]] = {}
    for data in iter_jatos_rows(input_path):
        if not submission_passes_rule(
            data,
            expected_answers,
            use_area_first_three_qualification,
            attention_threshold,
        ):
            continue

        row_index = data.get("rowIndex")
        if row_index is None:
            continue
        try:
            row_index_int = int(row_index)
        except (TypeError, ValueError):
            continue

        meta = data.get("meta", {})
        worker_id = ""
        if isinstance(meta, dict):
            worker_id = str(meta.get("workerId", "")).strip()
        if not worker_id:
            continue

        condition = get_condition(data)
        seen.setdefault(row_index_int, set()).add(worker_id)
        if condition:
            seen_by_condition.setdefault(condition, {}).setdefault(
                row_index_int, set()
            ).add(worker_id)

    total_rows = count_scenario_rows(scenario_path)
    if total_rows == 0:
        total_rows = count_export_rows(input_path)

    unique_counts = [len(workers) for workers in seen.values()]
    missing_overall = (
        sum(max(0, desired_per_row - count) for count in unique_counts)
        + max(0, total_rows - len(unique_counts)) * desired_per_row
    )
    stats: Dict[str, float] = {"missing_slots": float(missing_overall)}

    condition_list = conditions or sorted(seen_by_condition.keys())
    for condition in condition_list:
        rows = seen_by_condition.get(condition, {})
        counts = [len(workers) for workers in rows.values()]
        missing_slots = (
            sum(max(0, desired_per_row - count) for count in counts)
            + max(0, total_rows - len(counts)) * desired_per_row
        )
        stats[f"condition.{condition}.missing_slots"] = float(missing_slots)
    return stats


def get_condition(data: Dict[str, object]) -> str:
    """Extract the condition name from a JATOS result row."""
    condition = data.get("assignedCondition", "")
    if not condition:
        answers = data.get("answers", {})
        if isinstance(answers, dict):
            condition = answers.get("condition_name", "")
    if not condition:
        dependent_measure = str(data.get("dependent_measure", "") or "").strip()
        chart_type = str(data.get("chart_type", "") or "").strip()
        if dependent_measure and chart_type:
            condition = f"{dependent_measure}-{chart_type}"
    if not condition:
        answers = data.get("answers", {})
        if isinstance(answers, dict):
            dependent_measure = str(answers.get("dependent_measure", "") or "").strip()
            chart_type = str(answers.get("chart_type", "") or "").strip()
            if dependent_measure and chart_type:
                condition = f"{dependent_measure}-{chart_type}"
    if not isinstance(condition, str):
        return ""
    return condition.strip()


def get_flag_state(data: Dict[str, object], key: str) -> str:
    """Normalize a boolean flag into pass/fail/missing."""
    value = data.get(key)
    if value is None:
        return "missing"
    return "pass" if bool(value) else "fail"


def load_raw_qualification_answers(
    path: str = "mturk/qualification_answers.json",
) -> Dict[str, object]:
    """Load expected qualification answers keyed by raw question ids."""
    with open(path, "r", encoding="utf-8") as handle:
        answers = json.load(handle)
    if not isinstance(answers, dict):
        raise ValueError("qualification_answers.json did not contain an object.")
    return answers


def area_first_three_qualification_passed(
    answers: Dict[str, object],
    expected_answers: Dict[str, object],
) -> bool:
    """Return whether the first three area qualification answers are correct."""
    required = (
        "q_question-stacked-1",
        "q_question-stacked-2",
        "q_question-stacked-3",
    )
    for question_key in required:
        if question_key not in expected_answers:
            return False
        observed = str(answers.get(question_key, "")).strip()
        expected = str(expected_answers[question_key]).strip()
        if observed != expected:
            return False
    return True


def get_prolific_pid(data: Dict[str, object]) -> str:
    """Extract the Prolific participant id from a JATOS result row."""
    prolific = data.get("prolific", {})
    if not isinstance(prolific, dict):
        return ""
    return str(prolific.get("PROLIFIC_PID", "")).strip()


def load_batch_session(path: str) -> Dict[str, object]:
    """Load a batch session JSON file."""
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def detect_conditions(batch_session: Dict[str, object]) -> List[str]:
    """Detect condition names from batch session keys."""
    conditions: Set[str] = set()
    for key in batch_session.keys():
        for prefix in ("done_", "queue_", "inProgress_"):
            if key.startswith(prefix):
                conditions.add(key[len(prefix) :])
    return sorted(conditions)


def filter_conditions(conditions: Iterable[str]) -> List[str]:
    """Prefer full conditions; drop chart-only ones when full exist."""
    filtered = [cond for cond in conditions if cond]
    full_conditions = [cond for cond in filtered if "-" in cond]
    if full_conditions:
        return sorted(set(full_conditions))
    return sorted(set(filtered))


def alternate_condition_key(condition: str) -> str:
    """Return the reversed chart/dependent-measure condition key when possible."""
    if "-" not in condition:
        return condition
    left, right = condition.split("-", 1)
    return f"{right}-{left}"


def build_done_from_export(
    input_path: str,
    require_qualification: bool,
    require_attention: bool,
    use_area_first_three_qualification: bool = False,
    qualification_answers_path: str = "mturk/qualification_answers.json",
) -> Dict[str, Dict[str, List[str]]]:
    """Build done lists per condition from the JATOS export."""
    done: Dict[str, Dict[str, List[str]]] = {}
    seen: Dict[str, Dict[str, Set[str]]] = {}
    expected_answers: Dict[str, object] = {}
    if use_area_first_three_qualification:
        expected_answers = load_raw_qualification_answers(qualification_answers_path)
    for data in iter_jatos_rows(input_path):
        if use_area_first_three_qualification:
            if not submission_passes_rule(
                data,
                expected_answers,
                True,
                1.0 if require_attention else 0.0,
            ):
                continue
        else:
            if require_qualification and not data.get("qualificationPassed", False):
                continue
            if require_attention and not data.get("attentionPassed", False):
                continue
        condition = get_condition(data)
        if not condition:
            continue
        row_index = data.get("rowIndex")
        if row_index is None:
            continue
        try:
            row_index_int = int(row_index)
        except (TypeError, ValueError):
            continue
        meta = data.get("meta", {})
        worker_id = ""
        if isinstance(meta, dict):
            worker_id = str(meta.get("workerId", "")).strip()
        if worker_id == "":
            continue
        row_key = str(row_index_int)
        seen.setdefault(condition, {}).setdefault(row_key, set())
        if worker_id in seen[condition][row_key]:
            continue
        seen[condition][row_key].add(worker_id)
        done.setdefault(condition, {}).setdefault(row_key, []).append(worker_id)
    return done


def build_batch_session_from_export(
    input_path: str,
    scenario_path: str,
    desired_per_row: int,
    require_qualification: bool,
    require_attention: bool,
    batch_session: Dict[str, object],
    use_area_first_three_qualification: bool = False,
) -> Dict[str, object]:
    """Rebuild batch session queues from the JATOS export."""
    conditions = detect_conditions(batch_session)
    done = build_done_from_export(
        input_path,
        require_qualification=require_qualification,
        require_attention=require_attention,
        use_area_first_three_qualification=use_area_first_three_qualification,
    )
    if not conditions:
        conditions = sorted(done.keys())
    total_rows = count_scenario_rows(scenario_path) if scenario_path else 0
    if total_rows == 0:
        total_rows = count_export_rows(input_path)

    rebuilt: Dict[str, object] = {}
    for key, value in batch_session.items():
        if any(key.startswith(prefix) for prefix in ("done_", "queue_", "inProgress_")):
            continue
        rebuilt[key] = value

    for condition in conditions:
        cond_done = done.get(condition, {})
        if not cond_done:
            cond_done = done.get(alternate_condition_key(condition), {})
        queue: List[int] = []
        for row_index in range(total_rows):
            count = len(cond_done.get(str(row_index), []))
            missing = max(0, desired_per_row - count)
            if missing:
                queue.extend([row_index] * missing)
        rebuilt[f"done_{condition}"] = cond_done
        rebuilt[f"queue_{condition}"] = queue
        rebuilt[f"inProgress_{condition}"] = {}

    return rebuilt


def age_bin(age_value: str) -> str:
    """Bucket an age string into a labeled bin."""
    try:
        age = float(age_value)
    except (TypeError, ValueError):
        return "Unknown"
    if age < 18:
        return "Under 18"
    if age < 25:
        return "18-24"
    if age < 35:
        return "25-34"
    if age < 45:
        return "35-44"
    if age < 55:
        return "45-54"
    if age < 65:
        return "55-64"
    return "65+"


def compute_demographics_breakdown(
    input_path: str,
    demographics: Dict[str, Dict[str, str]],
    require_qualification: bool,
    require_attention: bool,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute demographic breakdowns by condition."""
    by_condition: Dict[str, Set[str]] = {}
    for data in iter_jatos_rows(input_path):
        if require_qualification and not data.get("qualificationPassed", False):
            continue
        if require_attention and not data.get("attentionPassed", False):
            continue
        condition = get_condition(data)
        if not condition:
            continue
        prolific_pid = get_prolific_pid(data)
        if not prolific_pid:
            continue
        if prolific_pid not in demographics:
            continue
        by_condition.setdefault(condition, set()).add(prolific_pid)

    breakdowns: Dict[str, Dict[str, Dict[str, float]]] = {}
    for condition, participant_ids in sorted(by_condition.items()):
        total = len(participant_ids)
        if total == 0:
            continue
        age_counts: Dict[str, int] = {}
        sex_counts: Dict[str, int] = {}
        race_counts: Dict[str, int] = {}
        for pid in participant_ids:
            demo = demographics.get(pid, {})
            age_label = age_bin(demo.get("Age", ""))
            sex_label = (demo.get("Sex") or "Unknown").strip() or "Unknown"
            race_label = (demo.get("Ethnicity simplified") or "Unknown").strip()
            if not race_label:
                race_label = "Unknown"
            age_counts[age_label] = age_counts.get(age_label, 0) + 1
            sex_counts[sex_label] = sex_counts.get(sex_label, 0) + 1
            race_counts[race_label] = race_counts.get(race_label, 0) + 1

        breakdowns[condition] = {
            "age": {k: v / total for k, v in sorted(age_counts.items())},
            "sex": {k: v / total for k, v in sorted(sex_counts.items())},
            "race": {k: v / total for k, v in sorted(race_counts.items())},
        }
    return breakdowns


def compute_quality_stats(
    input_path: str,
) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Compute qualification/attention pass-fail-missing counts overall and by condition."""
    totals: Dict[str, Dict[str, Dict[str, int]]] = {}
    for data in iter_jatos_rows(input_path):
        qual_state = get_flag_state(data, "qualificationPassed")
        attn_state = get_flag_state(data, "attentionPassed")
        if qual_state == "pass" and attn_state == "pass":
            pass_both_state = "pass"
        elif qual_state == "missing" or attn_state == "missing":
            pass_both_state = "missing"
        else:
            pass_both_state = "fail"

        for key, state in (
            ("qualification", qual_state),
            ("attention", attn_state),
            ("pass_both", pass_both_state),
        ):
            overall_bucket = totals.setdefault("__all__", {}).setdefault(key, {})
            overall_bucket[state] = overall_bucket.get(state, 0) + 1
            condition = get_condition(data)
            if condition:
                bucket = totals.setdefault(condition, {}).setdefault(key, {})
                bucket[state] = bucket.get(state, 0) + 1
    return totals


def compute_quality_stats_by_rule(
    input_path: str,
    use_area_first_three_qualification: bool = False,
    qualification_answers_path: str = "mturk/qualification_answers.json",
) -> Dict[str, Dict[str, Dict[str, int]]]:
    """Compute submission-level qualification/attention stats under a rule."""
    totals: Dict[str, Dict[str, Dict[str, int]]] = {}
    expected_answers: Dict[str, object] = {}
    if use_area_first_three_qualification:
        expected_answers = load_raw_qualification_answers(qualification_answers_path)

    for data in iter_jatos_rows(input_path):
        answers = data.get("answers", {})
        if not isinstance(answers, dict):
            continue
        if use_area_first_three_qualification:
            qual_state = (
                "pass"
                if area_first_three_qualification_passed(answers, expected_answers)
                else "fail"
            )
        else:
            qual_state = get_flag_state(data, "qualificationPassed")

        attn_state = get_flag_state(data, "attentionPassed")
        if qual_state == "pass" and attn_state == "pass":
            pass_both_state = "pass"
        elif qual_state == "missing" or attn_state == "missing":
            pass_both_state = "missing"
        else:
            pass_both_state = "fail"

        for key, state in (
            ("qualification", qual_state),
            ("attention", attn_state),
            ("pass_both", pass_both_state),
        ):
            overall_bucket = totals.setdefault("__all__", {}).setdefault(key, {})
            overall_bucket[state] = overall_bucket.get(state, 0) + 1
            condition = get_condition(data)
            if condition:
                bucket = totals.setdefault(condition, {}).setdefault(key, {})
                bucket[state] = bucket.get(state, 0) + 1
    return totals


def format_quality_line(
    label: str,
    bucket: Dict[str, int],
) -> str:
    """Format qualification/attention stats for printing."""
    return (
        f"{label}: pass={bucket.get('pass', 0)}, "
        f"fail={bucket.get('fail', 0)}, "
        f"missing={bucket.get('missing', 0)}."
    )


def attention_counts_from_answers(answers: Dict[str, object]) -> Tuple[int, int]:
    """Return number of passed and total attention checks in one answers dict."""
    passed = 0
    total = 0
    for key, expected_value in answers.items():
        if not re.match(r"^q_question-\d+_attn_answer$", key):
            continue
        observed_key = key.replace("_attn_answer", "_attn")
        if observed_key not in answers:
            continue
        total += 1
        observed_value = str(answers.get(observed_key, "")).strip()
        expected_text = str(expected_value).strip()
        if observed_value == expected_text:
            passed += 1
    return passed, total


def compute_worker_inclusion_stats(
    input_path: str,
    use_area_first_three_qualification: bool = False,
    qualification_answers_path: str = "mturk/qualification_answers.json",
) -> Dict[str, Dict[str, float]]:
    """Compute worker-level inclusion rates by condition and overall.

    A worker is counted as included at a threshold if:
    1) they pass qualification, and
    2) their attention accuracy is at least the threshold.
    """
    expected_answers: Dict[str, object] = {}
    if use_area_first_three_qualification:
        expected_answers = load_raw_qualification_answers(qualification_answers_path)

    buckets: Dict[str, Dict[str, Dict[str, float]]] = {"__all__": {}}
    for data in iter_jatos_rows(input_path):
        meta = data.get("meta", {})
        if not isinstance(meta, dict):
            continue
        worker_id = str(meta.get("workerId", "")).strip()
        if not worker_id:
            continue

        answers = data.get("answers", {})
        if not isinstance(answers, dict):
            continue

        condition = get_condition(data)
        bucket_names = ["__all__"]
        if condition:
            bucket_names.append(condition)

        passed, total = attention_counts_from_answers(answers)
        if use_area_first_three_qualification:
            qual_pass = area_first_three_qualification_passed(answers, expected_answers)
        else:
            qual_pass = bool(data.get("qualificationPassed", False))

        for bucket_name in bucket_names:
            worker_entry = buckets.setdefault(bucket_name, {}).setdefault(
                worker_id,
                {"qual_pass": False, "attention_passed": 0.0, "attention_total": 0.0},
            )
            worker_entry["qual_pass"] = bool(worker_entry["qual_pass"]) or qual_pass
            worker_entry["attention_passed"] = float(
                worker_entry["attention_passed"]
            ) + float(passed)
            worker_entry["attention_total"] = float(
                worker_entry["attention_total"]
            ) + float(total)

    output: Dict[str, Dict[str, float]] = {}
    for bucket_name, workers in buckets.items():
        total_workers = len(workers)
        qual_pass_count = 0
        pass_100 = 0
        pass_075 = 0
        for worker in workers.values():
            qual_pass = bool(worker["qual_pass"])
            attn_total = float(worker["attention_total"])
            attn_rate = (
                float(worker["attention_passed"]) / attn_total
                if attn_total > 0
                else 0.0
            )
            if qual_pass:
                qual_pass_count += 1
            if qual_pass and attn_rate >= 1.0:
                pass_100 += 1
            if qual_pass and attn_rate >= 0.75:
                pass_075 += 1
        output[bucket_name] = {
            "total_workers": float(total_workers),
            "qual_pass_workers": float(qual_pass_count),
            "pass_100_workers": float(pass_100),
            "pass_075_workers": float(pass_075),
            "pass_100_rate": (pass_100 / total_workers) if total_workers else 0.0,
            "pass_075_rate": (pass_075 / total_workers) if total_workers else 0.0,
        }
    return output


def format_inclusion_line(label: str, stats: Dict[str, float]) -> str:
    """Format worker inclusion summary for two attention thresholds."""
    total_workers = int(stats.get("total_workers", 0))
    qual_pass_workers = int(stats.get("qual_pass_workers", 0))
    pass_100_workers = int(stats.get("pass_100_workers", 0))
    pass_075_workers = int(stats.get("pass_075_workers", 0))
    pass_100_rate = stats.get("pass_100_rate", 0.0)
    pass_075_rate = stats.get("pass_075_rate", 0.0)
    return (
        f"{label}: workers={total_workers}, qual-pass={qual_pass_workers}, "
        f"qual+attn>=1.00={pass_100_workers} ({pass_100_rate:.2%}), "
        f"qual+attn>=0.75={pass_075_workers} ({pass_075_rate:.2%})."
    )


def cost_projection(
    inclusion_stats: Dict[str, float],
    remaining_100: int,
    remaining_075: int,
    unit_cost: float,
) -> Dict[str, float]:
    """Compute cost-so-far and projected submissions/cost for two thresholds."""
    submitted = int(inclusion_stats.get("total_workers", 0))
    pass_100_rate = float(inclusion_stats.get("pass_100_rate", 0.0))
    pass_075_rate = float(inclusion_stats.get("pass_075_rate", 0.0))

    needed_submissions_100 = (
        (remaining_100 / pass_100_rate)
        if remaining_100 > 0 and pass_100_rate > 0
        else 0.0
    )
    needed_submissions_075 = (
        (remaining_075 / pass_075_rate)
        if remaining_075 > 0 and pass_075_rate > 0
        else 0.0
    )

    return {
        "unit_cost": unit_cost,
        "submitted": float(submitted),
        "cost_so_far": submitted * unit_cost,
        "remaining_100": float(remaining_100),
        "remaining_075": float(remaining_075),
        "needed_submissions_100": needed_submissions_100,
        "needed_submissions_075": needed_submissions_075,
        "cost_remaining_100": needed_submissions_100 * unit_cost,
        "cost_remaining_075": needed_submissions_075 * unit_cost,
    }


def format_cost_projection_line(label: str, projection: Dict[str, float]) -> str:
    """Format one-line cost projection output."""
    return (
        f"{label}: submitted={int(projection.get('submitted', 0))}, "
        f"cost_so_far=${projection.get('cost_so_far', 0.0):,.2f}; "
        f"attn>=1.00 remaining_slots={int(projection.get('remaining_100', 0))}, "
        f"projected_submissions={projection.get('needed_submissions_100', 0.0):.1f}, "
        f"projected_cost=${projection.get('cost_remaining_100', 0.0):,.2f}; "
        f"attn>=0.75 remaining_slots={int(projection.get('remaining_075', 0))}, "
        f"projected_submissions={projection.get('needed_submissions_075', 0.0):.1f}, "
        f"projected_cost=${projection.get('cost_remaining_075', 0.0):,.2f}."
    )


def collect_keys(path: str) -> Tuple[List[str], List[str], List[str], int]:
    """Collect input, answer, and extra top-level keys across all JATOS rows."""
    input_keys: Set[str] = set()
    answer_keys: Set[str] = set()
    extra_keys: Set[str] = set()
    row_count = 0
    for data in iter_jatos_rows(path):
        row_count += 1
        row = data.get("row", {})
        answers = data.get("answers", {})
        input_keys.update(row.keys())
        answer_keys.update(answers.keys())
        for key in data.keys():
            if key not in {"row", "answers"}:
                extra_keys.add(key)
    return sorted(input_keys), sorted(answer_keys), sorted(extra_keys), row_count


def build_output_row(
    data: Dict[str, object],
    input_keys: List[str],
    answer_keys: List[str],
    extra_keys: List[str],
    demographics: Dict[str, Dict[str, str]],
    demographics_columns: List[str],
) -> Dict[str, object]:
    """Build a single MTurk-style CSV row from a JATOS entry."""
    output: Dict[str, object] = {}
    meta = data.get("meta", {})
    worker_id = ""
    if isinstance(meta, dict):
        worker_id = str(meta.get("workerId", "")).strip()
    output["WorkerId"] = worker_id

    prolific_pid = get_prolific_pid(data)
    if demographics and prolific_pid in demographics:
        demo_row = demographics[prolific_pid]
        for key in demographics_columns:
            output[f"Demographics.{key}"] = demo_row.get(key, "")

    for key in extra_keys:
        value = data.get(key, "")
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=True, sort_keys=True)
        output[key] = value

    row = data.get("row", {})
    answers = data.get("answers", {})
    for key in input_keys:
        value = ""
        if isinstance(row, dict):
            value = row.get(key, "")
        output[f"Input.{key}"] = value
    for key in answer_keys:
        value = ""
        if isinstance(answers, dict):
            value = answers.get(key, "")
        output[f"Answer.{key}"] = value
    return output


def output_path(input_path: str, output_override: str) -> str:
    """Resolve the output path based on inputs."""
    if output_override:
        return output_override
    base = os.path.splitext(os.path.basename(input_path))[0]
    parent = os.path.dirname(input_path)
    if parent:
        return os.path.join(parent, f"{base}.csv")
    return os.path.join("data", "results", "mturk", f"{base}.csv")


def batch_session_output_path(input_path: str, output_override: str) -> str:
    """Resolve the batch session output path based on inputs."""
    if output_override:
        return output_override
    base = os.path.splitext(os.path.basename(input_path))[0]
    parent = os.path.dirname(input_path)
    if parent:
        return os.path.join(parent, f"{base}_batch_session_rebuilt.json")
    return os.path.join(
        "data",
        "results",
        "mturk",
        f"{base}_batch_session_rebuilt.json",
    )


def write_csv(
    input_path: str,
    output_csv: str,
    input_keys: List[str],
    answer_keys: List[str],
    extra_keys: List[str],
    demographics: Dict[str, Dict[str, str]],
    demographics_columns: List[str],
    condition_filter: str | None = None,
) -> int:
    """Write MTurk-style CSV rows to disk."""
    fieldnames = ["WorkerId"]
    if demographics_columns:
        fieldnames.extend([f"Demographics.{key}" for key in demographics_columns])
    fieldnames.extend(extra_keys)
    fieldnames.extend([f"Input.{key}" for key in input_keys])
    fieldnames.extend([f"Answer.{key}" for key in answer_keys])

    row_count = 0
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for data in iter_jatos_rows(input_path):
            if condition_filter:
                if get_condition(data) != condition_filter:
                    continue
            writer.writerow(
                build_output_row(
                    data,
                    input_keys,
                    answer_keys,
                    extra_keys,
                    demographics,
                    demographics_columns,
                )
            )
            row_count += 1
    return row_count


def main() -> None:
    """Run the JATOS to MTurk conversion."""
    args = parse_args()
    output_csv_path = output_path(args.input, args.output)
    input_keys, answer_keys, extra_keys, row_count = collect_keys(args.input)
    demographics, demographics_columns = load_demographics(args.demographics_file)
    if row_count == 0:
        print("No JATOS rows with answers found.", file=sys.stderr)
        return
    if not args.no_split_by_condition:
        conditions: Set[str] = set()
        for data in iter_jatos_rows(args.input):
            condition = get_condition(data)
            if condition:
                conditions.add(condition)
        conditions_list = filter_conditions(conditions)
        if not conditions_list:
            print("No conditions found to split output.", file=sys.stderr)
        for condition in conditions_list:
            split_path = (
                f"{os.path.splitext(output_csv_path)[0]}_condition={condition}.csv"
            )
            written = write_csv(
                args.input,
                split_path,
                input_keys,
                answer_keys,
                extra_keys,
                demographics,
                demographics_columns,
                condition_filter=condition,
            )
            print(f"Wrote {written} rows to {split_path}.")
    else:
        written = write_csv(
            args.input,
            output_csv_path,
            input_keys,
            answer_keys,
            extra_keys,
            demographics,
            demographics_columns,
        )
        print(f"Wrote {written} rows to {output_csv_path}.")
    if args.scenario_file:
        use_area_first_three_qualification = (
            args.qualification_rule == "area_first_three"
        )
        require_qualification = (
            not args.include_failed_qualification
            and not use_area_first_three_qualification
        )
        require_attention = not args.include_failed_attention
        if use_area_first_three_qualification:
            stats = compute_coverage_by_rule(
                args.input,
                args.scenario_file,
                args.desired_per_row,
                attention_threshold=1.0,
                use_area_first_three_qualification=True,
                conditions=conditions_list if not args.no_split_by_condition else None,
            )
            quality_stats = compute_quality_stats_by_rule(
                args.input,
                use_area_first_three_qualification=True,
            )
            inclusion_stats = compute_worker_inclusion_stats(
                args.input,
                use_area_first_three_qualification=True,
            )
        else:
            stats = compute_coverage(
                args.input,
                args.scenario_file,
                args.desired_per_row,
                require_qualification,
                require_attention,
                conditions=conditions_list if not args.no_split_by_condition else None,
            )
            quality_stats = compute_quality_stats(args.input)
            inclusion_stats = compute_worker_inclusion_stats(args.input)
        inclusion_stats_area_first_three = compute_worker_inclusion_stats(
            args.input,
            use_area_first_three_qualification=True,
        )
        condition_keys = [
            key.split(".", 1)[1].split(".", 1)[0]
            for key in stats
            if key.startswith("condition.") and key.endswith(".rows_with_any")
        ]
        condition_names = sorted(set(condition_keys))
        missing_flag_100 = compute_missing_slots_by_rule(
            args.input,
            args.scenario_file,
            args.desired_per_row,
            attention_threshold=1.0,
            use_area_first_three_qualification=False,
            conditions=condition_names if condition_names else None,
        )
        missing_flag_075 = compute_missing_slots_by_rule(
            args.input,
            args.scenario_file,
            args.desired_per_row,
            attention_threshold=0.75,
            use_area_first_three_qualification=False,
            conditions=condition_names if condition_names else None,
        )
        missing_area_100 = compute_missing_slots_by_rule(
            args.input,
            args.scenario_file,
            args.desired_per_row,
            attention_threshold=1.0,
            use_area_first_three_qualification=True,
            conditions=condition_names if condition_names else None,
        )
        missing_area_075 = compute_missing_slots_by_rule(
            args.input,
            args.scenario_file,
            args.desired_per_row,
            attention_threshold=0.75,
            use_area_first_three_qualification=True,
            conditions=condition_names if condition_names else None,
        )
        main_missing_100 = (
            missing_area_100 if use_area_first_three_qualification else missing_flag_100
        )
        main_missing_075 = (
            missing_area_075 if use_area_first_three_qualification else missing_flag_075
        )
        unit_cost = args.submission_cost * args.cost_multiplier
        overall_quality = quality_stats.get("__all__", {})
        main_projection_label = (
            "Cost projection (area qual first 3)"
            if use_area_first_three_qualification
            else "Cost projection (qualificationPassed flag)"
        )

        print()
        print("=== OVERALL SUMMARY ===")
        print("Qualification rule for main summary: " f"{args.qualification_rule}.")
        print(
            "Coverage "
            f"(desired {args.desired_per_row} per row): "
            f"{int(stats['rows_with_target'])}/{int(stats['total_rows'])} "
            f"rows ({stats['target_rate']:.2%})."
        )
        print(
            "Per-row counts with any ratings: "
            f"avg={stats['avg_per_row_with_any']:.2f}, "
            f"min={int(stats['min_per_row_with_any'])}, "
            f"max={int(stats['max_per_row_with_any'])}."
        )
        print(
            "Remaining assignments to reach target: " f"{int(stats['missing_slots'])}."
        )
        print(
            format_quality_line(
                "Qualification", overall_quality.get("qualification", {})
            )
        )
        print(format_quality_line("Attention", overall_quality.get("attention", {})))
        print(
            format_quality_line(
                "Pass both (qual+attention)",
                overall_quality.get("pass_both", {}),
            )
        )
        print(
            format_inclusion_line(
                "Included for study (worker-level)",
                inclusion_stats.get("__all__", {}),
            )
        )
        if not use_area_first_three_qualification:
            print(
                format_inclusion_line(
                    "Included for study (worker-level, area qual first 3)",
                    inclusion_stats_area_first_three.get("__all__", {}),
                )
            )
        print(
            "Cost assumptions: "
            f"base=${args.submission_cost:.2f}, "
            f"multiplier={args.cost_multiplier:.2f}, "
            f"effective=${unit_cost:.2f} per submission."
        )
        print(
            format_cost_projection_line(
                main_projection_label,
                cost_projection(
                    inclusion_stats.get("__all__", {}),
                    remaining_100=int(main_missing_100.get("missing_slots", 0)),
                    remaining_075=int(main_missing_075.get("missing_slots", 0)),
                    unit_cost=unit_cost,
                ),
            )
        )
        if not use_area_first_three_qualification:
            print(
                format_cost_projection_line(
                    "Cost projection (area qual first 3)",
                    cost_projection(
                        inclusion_stats_area_first_three.get("__all__", {}),
                        remaining_100=int(missing_area_100.get("missing_slots", 0)),
                        remaining_075=int(missing_area_075.get("missing_slots", 0)),
                        unit_cost=unit_cost,
                    ),
                )
            )

        print()
        print("=== PER-CONDITION SUMMARY ===")
        for condition in condition_names:
            prefix = f"condition.{condition}."
            print(f"[{condition}]")
            print(
                "Coverage "
                f"(desired {args.desired_per_row} per row): "
                f"{int(stats[prefix + 'rows_with_target'])}/"
                f"{int(stats[prefix + 'total_rows'])} "
                f"rows ({stats[prefix + 'target_rate']:.2%})."
            )
            print(
                "Per-row counts with any ratings: "
                f"avg={stats[prefix + 'avg_per_row_with_any']:.2f}, "
                f"min={int(stats[prefix + 'min_per_row_with_any'])}, "
                f"max={int(stats[prefix + 'max_per_row_with_any'])}."
            )
            print(
                "Remaining assignments to reach target: "
                f"{int(stats[prefix + 'missing_slots'])}."
            )
            cond_quality = quality_stats.get(condition, {})
            print(
                format_quality_line(
                    "Qualification",
                    cond_quality.get("qualification", {}),
                )
            )
            print(
                format_quality_line(
                    "Attention",
                    cond_quality.get("attention", {}),
                )
            )
            print(
                format_quality_line(
                    "Pass both (qual+attention)",
                    cond_quality.get("pass_both", {}),
                )
            )
            print(
                format_inclusion_line(
                    "Included for study (worker-level)",
                    inclusion_stats.get(condition, {}),
                )
            )
            if not use_area_first_three_qualification:
                print(
                    format_inclusion_line(
                        "Included for study (worker-level, area qual first 3)",
                        inclusion_stats_area_first_three.get(condition, {}),
                    )
                )
            print(
                format_cost_projection_line(
                    main_projection_label,
                    cost_projection(
                        inclusion_stats.get(condition, {}),
                        remaining_100=int(
                            main_missing_100.get(f"{prefix}missing_slots", 0)
                        ),
                        remaining_075=int(
                            main_missing_075.get(f"{prefix}missing_slots", 0)
                        ),
                        unit_cost=unit_cost,
                    ),
                )
            )
            if not use_area_first_three_qualification:
                print(
                    format_cost_projection_line(
                        "Cost projection (area qual first 3)",
                        cost_projection(
                            inclusion_stats_area_first_three.get(condition, {}),
                            remaining_100=int(
                                missing_area_100.get(f"{prefix}missing_slots", 0)
                            ),
                            remaining_075=int(
                                missing_area_075.get(f"{prefix}missing_slots", 0)
                            ),
                            unit_cost=unit_cost,
                        ),
                    )
                )
            print()
        if demographics:
            breakdowns = compute_demographics_breakdown(
                args.input,
                demographics,
                require_qualification,
                require_attention,
            )
            for condition, detail in sorted(breakdowns.items()):
                print(f"Demographics {condition}:")
                for category in ("age", "sex", "race"):
                    entries = ", ".join(
                        f"{label}={pct:.2%}" for label, pct in detail[category].items()
                    )
                    print(f"{category}: {entries}")
                print()
    if args.batch_session_file:
        use_area_first_three_qualification = (
            args.qualification_rule == "area_first_three"
        )
        require_qualification = (
            not args.include_failed_qualification
            and not use_area_first_three_qualification
        )
        require_attention = not args.include_failed_attention
        batch_session = load_batch_session(args.batch_session_file)
        rebuilt = build_batch_session_from_export(
            args.input,
            args.scenario_file,
            args.desired_per_row,
            require_qualification,
            require_attention,
            batch_session,
            use_area_first_three_qualification=use_area_first_three_qualification,
        )
        batch_output_path = batch_session_output_path(
            args.input, args.batch_session_output
        )
        batch_output_dir = os.path.dirname(batch_output_path)
        if batch_output_dir:
            os.makedirs(batch_output_dir, exist_ok=True)
        with open(batch_output_path, "w", encoding="utf-8") as handle:
            json.dump(rebuilt, handle, indent=2, sort_keys=True)
        print(f"Wrote rebuilt batch session to {batch_output_path}.")


if __name__ == "__main__":
    main()
