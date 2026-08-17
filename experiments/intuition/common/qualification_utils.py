"""Utilities for recomputing qualification pass/fail from answer columns."""

from __future__ import annotations

import json
from typing import Iterable, Mapping, Sequence, Tuple


def load_qualification_answers(path: str = "mturk/qualification_answers.json") -> dict:
    """Load expected qualification answers keyed by Answer.<field>."""
    with open(path, "r", encoding="utf-8") as handle:
        answers = json.load(handle)
    return {f"Answer.{key}": value for key, value in answers.items()}


def qualification_columns_in_header(
    header: Iterable[str], answers: Mapping[str, object]
) -> Tuple[list[str], list[str]]:
    """Return qualification columns present and missing."""
    expected = set(answers.keys())
    present = [col for col in expected if col in header]
    missing = sorted(expected - set(present))
    return present, missing


def qualification_passed_row(
    row: Mapping[str, object], columns: Sequence[str], answers: Mapping[str, object]
) -> bool:
    """Check qualification answers for a CSV row."""
    if not columns:
        return False
    for column in columns:
        if str(row.get(column, "")) != str(answers[column]):
            return False
    return True


def qualification_passed_group(
    group, columns: Sequence[str], answers: Mapping[str, object]
) -> bool:
    """Check qualification answers for a worker group."""
    if not columns:
        return False
    for column in columns:
        values = group[column].dropna().astype(str).unique()
        if len(values) != 1 or values[0] != str(answers[column]):
            return False
    return True
