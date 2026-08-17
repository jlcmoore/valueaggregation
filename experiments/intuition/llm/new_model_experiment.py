"""
Run the intuition experiment using LiteLLM batch completion.

It uses LiteLLM batch completion and can sample multiple responses per question.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence

import litellm
import pandas as pd
from common.utils import filename_to_variables
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)
from tqdm import tqdm

import value_aggregation as pm

logging.basicConfig(level=logging.INFO)
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)

INPUT_DIR = "./data/scenarios"
OUTPUT_DIR = "./data/results/llm"

OPTIONS = string.ascii_uppercase

ORDER = [
    "wait-times",
    "life-expectancy",
    "medical-costs",
    "travel-times",
    "emergency-response-time",
    "hospital-readmission-rates",
    "chronic-disease-prevalence",
    "communicable-disease-spread",
    "mental-health-access",
    "infant-mortality",
    "overdose-incidence",
    "mental-health-waitlist-length",
    "missed-work",
    "specialist-referral-delay",
    "primary-care-access",
    "medication-adherence",
    "post-surgery-recovery-time",
    "preventable-hospitalizations",
]

SYSTEM = "Follow the given examples and answer the questions."

HIT_DIRECTIONS = (
    "TASK: In this task we assess how to choose between different views.\n\n"
    "Tell us which of the options specified is the best for the given situation.\n\n"
    "Whether one option is better than another is up to you. It might be that "
    "multiple parties have to accept a slightly worse outcome for themselves in "
    "order to best balance the desires of the group.\n\n"
    "In this version of the task, groups prefer higher outcomes. This means that "
    "higher outcomes are better.\n\n"
    "SCENARIO: Your local health department is looking for advice on a project.\n\n"
    "To figure out what to do, the health department has researched how the "
    "proposals will affect different groups. We will not get into the specifics of "
    "the proposals, but none is perfect; some groups prefer some proposals over "
    "others.\n\n"
    "TIP: If you find yourself stuck between different proposals, please go with "
    "your gut or intuition.\n\n"
    "Answer with only the proposal letter."
)

QUAL_DIRECTIONS = (
    "This is a qualification task. Your answers will be compared to our ground "
    "truth answers.\n\n"
    "TASK: In this task we assess how well you can judge various aggregated "
    "properties of groups.\n\n"
    "These charts will show you the numeric outcomes on a few proposals for a few "
    "groups.\n\n"
    "Answer with only the proposal letter."
)

COMPROMISE_Q = (
    "Which proposal is best in this situation? " "Answer with only the proposal letter."
)
LITELLM_API_ERRORS = (
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
    TimeoutError,
)

EXAMPLE_DATA = [
    [
        {"utility": 20, "agent": "A", "credence": 10, "action": "one"},
        {"utility": 10, "agent": "B", "credence": 10, "action": "one"},
    ],
    [
        {"utility": 20, "agent": "A", "credence": 10, "action": "one"},
        {"utility": 0, "agent": "A", "credence": 10, "action": "two"},
        {"utility": 0, "agent": "B", "credence": 10, "action": "one"},
        {"utility": 10, "agent": "B", "credence": 10, "action": "two"},
    ],
    [
        {"utility": 20, "agent": "A", "credence": 10, "action": "one"},
        {"utility": 5, "agent": "A", "credence": 10, "action": "two"},
        {"utility": 5, "agent": "B", "credence": 10, "action": "one"},
        {"utility": 10, "agent": "B", "credence": 10, "action": "two"},
    ],
    [
        {"utility": 1000.0, "agent": "A", "credence": 30, "action": "one"},
        {"utility": 1000.0, "agent": "B", "credence": 10, "action": "one"},
        {"utility": 1000.0, "agent": "C", "credence": 20, "action": "one"},
        {"utility": 1.0, "agent": "A", "credence": 30, "action": "two"},
        {"utility": 1.0, "agent": "B", "credence": 10, "action": "two"},
        {"utility": 1.0, "agent": "C", "credence": 20, "action": "two"},
        {"utility": 1.0, "agent": "A", "credence": 30, "action": "three"},
        {"utility": 1.0, "agent": "B", "credence": 10, "action": "three"},
        {"utility": 10000.0, "agent": "C", "credence": 20, "action": "three"},
    ],
]


@dataclass(frozen=True)
class ModelConfig:
    """Model configuration for LiteLLM calls."""

    name: str
    model_id: str
    reasoning_effort: Optional[str] = None
    extra_params: Optional[Mapping[str, object]] = None


@dataclass(frozen=True)
class RequestItem:
    """Prepared request payload for a single scenario question."""

    row_id: int
    participant_id: str
    scenario_hash: str
    question: int
    proposals: List[str]
    messages: List[Mapping[str, object]]


@dataclass(frozen=True)
class SampledRequest:
    """Represents a single sampled request for an item."""

    item_index: int
    sample_index: int
    messages: List[Mapping[str, object]]


@dataclass(frozen=True)
class CostEstimate:
    """Aggregate cost estimate for a model run."""

    model_id: str
    requests: int
    prompt_tokens: Optional[int]
    assumed_completion_tokens: Optional[int]
    prompt_cost_usd: Optional[float]
    completion_cost_usd: Optional[float]
    total_cost_usd: Optional[float]


def build_contexts(maximize: bool) -> dict[str, dict[str, str]]:
    """Return context strings keyed by scenario category."""

    valence_min = "decrease" if maximize else "increase"
    valence_max = "increase" if maximize else "decrease"
    return {
        "wait-times": {
            "context": (
                f"will {valence_min} the average number of days a group member must "
                "wait for an appointment"
            ),
            "unit": "days",
        },
        "life-expectancy": {
            "context": (
                f"will {valence_max} the average number of years a group member will "
                "live"
            ),
            "unit": "years",
        },
        "medical-costs": {
            "context": (
                f"will {valence_min} the average cost of a medical visit for each "
                "group"
            ),
            "unit": "dollars",
        },
        "travel-times": {
            "context": (
                f"will {valence_min} the average number of minutes a group member must "
                "travel for an appointment"
            ),
            "unit": "minutes",
        },
        "emergency-response-time": {
            "context": (f"will {valence_min} the average ambulance response time"),
            "unit": "minutes",
        },
        "hospital-readmission-rates": {
            "context": (
                f"will {valence_min} the 30-day readmission rate at the local hospital"
            ),
            "unit": "percent",
        },
        "chronic-disease-prevalence": {
            "context": (
                f"will {valence_min} the number of adults with a preventable "
                "chronic disease (like diabetes)"
            ),
            "unit": "cases per 1,000 people",
        },
        "communicable-disease-spread": {
            "context": (
                f"will {valence_min} the average number of people infected per "
                "disease outbreak"
            ),
            "unit": "people",
        },
        "mental-health-access": {
            "context": (
                f"will {valence_min} the average number of people per mental health "
                "professional"
            ),
            "unit": "people",
        },
        "infant-mortality": {
            "context": (
                f"will {valence_min} the number of infant deaths per 1,000 live births"
            ),
            "unit": "deaths per 1,000 births",
        },
        "overdose-incidence": {
            "context": (
                f"will {valence_min} the number of overdose events per 100,000 people"
            ),
            "unit": "cases per 100,000 people",
        },
        "mental-health-waitlist-length": {
            "context": (
                f"will {valence_min} the average number of days to the first therapy "
                "appointment"
            ),
            "unit": "days",
        },
        "missed-work": {
            "context": (
                f"will {valence_min} the average number of work days missed due to "
                "illness"
            ),
            "unit": "days",
        },
        "specialist-referral-delay": {
            "context": (
                f"will {valence_min} the average number of days between referral and "
                "specialist visit"
            ),
            "unit": "days",
        },
        "primary-care-access": {
            "context": (
                f"will {valence_min} the average number of days to a primary care "
                "appointment"
            ),
            "unit": "days",
        },
        "medication-adherence": {
            "context": (
                f"will {valence_max} the share of patients who take prescribed "
                "medications as directed"
            ),
            "unit": "percent",
        },
        "post-surgery-recovery-time": {
            "context": (
                f"will {valence_min} the average number of days needed for recovery "
                "after surgery"
            ),
            "unit": "days",
        },
        "preventable-hospitalizations": {
            "context": (
                f"will {valence_min} the number of hospital admissions that could "
                "have been prevented"
            ),
            "unit": "cases per 100,000 people",
        },
    }


def explode_df_with_fehr(
    df: pd.DataFrame,
    num_scenarios: int,
) -> pd.DataFrame:
    """Explode scenario columns and keep Fehr alpha columns without recomputing."""

    required_col_names = {
        "scenario_{0}_hash": "scenario_hash",
        "scenario_{0}_json": "scenario_json",
        "scenario_{0}_nbs": "scenario_nbs",
        "scenario_{0}_mec": "scenario_mec",
        "scenario_{0}_mft": "scenario_mft",
    }
    optional_col_names = {
        "scenario_{0}_ie": "scenario_ie",
        "scenario_{0}_alpha_bin": "scenario_alpha_bin",
    }
    frames: List[pd.DataFrame] = []
    for index in range(1, num_scenarios + 1):
        num_cols = {
            key.format(index): value for key, value in required_col_names.items()
        }
        for key, value in optional_col_names.items():
            source_col = key.format(index)
            if source_col in df.columns:
                num_cols[source_col] = value
        fehr_prefix = f"scenario_{index}_fehr_alpha_"
        fehr_cols = [col for col in df.columns if col.startswith(fehr_prefix)]
        renamed_fehr = {
            col: col.replace(fehr_prefix, "scenario_fehr_alpha_") for col in fehr_cols
        }
        selected_cols = list(num_cols.keys()) + fehr_cols
        selected = df[selected_cols].rename(columns=num_cols | renamed_fehr)
        if "scenario_ie" not in selected.columns and "scenario_mec" in selected.columns:
            selected["scenario_ie"] = selected["scenario_mec"]
        elif "scenario_ie" in selected.columns and "scenario_mec" in selected.columns:
            selected["scenario_ie"] = selected["scenario_ie"].where(
                selected["scenario_ie"].notna(),
                selected["scenario_mec"],
            )
        if "scenario_alpha_bin" not in selected.columns:
            selected["scenario_alpha_bin"] = pd.NA
        selected["question"] = index
        selected["participant_id"] = selected.index.map(lambda row: f"llm_{int(row)}")
        frames.append(selected)
    return pd.concat(frames).reset_index(drop=True)


def mec_qual_description(game_state: pm.VoteGameState) -> str:
    """Return the area chart description for a qualification task."""

    sums = ""
    for proposal in game_state.getVotes():
        output = f"For proposal {proposal}, the total area is "
        total = 0.0
        total_calculation = ""
        for index, group in enumerate(game_state.getAgents()):
            credence = game_state.getAgentCredence(group)
            utility = game_state.getUtility(group, proposal)
            format_spec = ".1f" if credence < 1 else ".0f"
            output += f'{utility} * {credence:{format_spec}} (for group "{group}") '
            total_calculation += f"{credence * utility}"
            if index < game_state.numAgents() - 1:
                total_calculation += " + "
                output += "+ "
            total += credence * utility
        sums += f"{output}= {total_calculation} = {total:.1f}.\n\n"
    return (
        "The total area for a proposal is calculated by summing up the product "
        "of group size and proposed outcome for each group.\n\n"
        f"{sums}"
    )


def qual_question(game_state: pm.VoteGameState, chart_type: str) -> str:
    """Return the qualification question for a chart type."""

    question = f"Which of the proposal(s) takes up the largest total {chart_type}?\n"
    proposals = make_question_list(game_state)
    return question + proposals


def make_question_description(
    game_state: pm.VoteGameState,
    context: Optional[str],
    unit: Optional[str],
) -> str:
    """Return the descriptive paragraph for a scenario."""

    groups = game_state.getAgents()
    group_description = (
        f"In this scenario, there are {len(groups)} groups:\n"
        if len(groups) > 1
        else f"In this scenario, there is {len(groups)} group:\n"
    )
    group_list = ""
    for index, group in enumerate(groups):
        text = f"- group {group} with {game_state.getAgentCredence(group)}"
        text += " people in it" if game_state.getAgentCredence(group) > 1 else " person"
        if index < len(groups) - 2:
            text += ", "
        elif index < len(groups) - 1:
            text += ", and "
        else:
            text += "."
        group_list += text + "\n"
    group_list += "\n"

    proposals = game_state.getVotes()
    proposals_description = (
        f"There are {len(proposals)} proposals"
        if len(proposals) > 1
        else f"There is {len(proposals)} proposal"
    )
    if context is not None:
        proposals_description += f", each of which {context} by"
    proposals_description += ":\n"

    proposals_list = ""
    for proposal in proposals:
        scenario_text = f"- proposal {proposal}: "
        if unit is None:
            scenario_text += "with an outcome of "
        for index, group in enumerate(groups):
            utility = game_state.getAgentOutcomes(group)[proposal]
            text = f"{utility}"
            if unit is not None:
                text += f" {unit}"
            text += f" for group {group}"
            if index < len(groups) - 2:
                text += ", "
            elif index < len(groups) - 1:
                text += ", and "
            else:
                text += "."
            scenario_text += text
        proposals_list += scenario_text + "\n"
    proposals_list += "\n"

    return group_description + group_list + proposals_description + proposals_list


def make_question_list(game_state: pm.VoteGameState) -> str:
    """Return formatted proposal list with lettered options."""

    proposals = [f"Proposal {proposal}" for proposal in game_state.getVotes()]
    proposal_lines = [
        f"- ({OPTIONS[index]}) {proposals[index]}" for index in range(len(proposals))
    ]
    return "\n".join(proposal_lines)


def make_dialogue(
    game_state: pm.VoteGameState,
    context_name: Optional[str],
    examples: Sequence[tuple[str, str]],
    qualification: bool,
    show_chart: bool,
    chart_type: str,
    contexts: Mapping[str, Mapping[str, str]],
) -> List[Mapping[str, object]]:
    """Build the LiteLLM message list for a scenario."""

    if qualification:
        description = make_question_description(game_state, context=None, unit=None)
        question = qual_question(game_state, chart_type)
        dialogue = description + question
    else:
        context = contexts[context_name]["context"] if context_name else None
        unit = contexts[context_name]["unit"] if context_name else None
        description = make_question_description(game_state, context, unit)
        if show_chart:
            if chart_type == "area":
                description += mec_qual_description(game_state)
        description += COMPROMISE_Q + "\n"
        dialogue = description + make_question_list(game_state)

    messages: List[Mapping[str, object]] = [{"role": "system", "content": SYSTEM}]
    for role, content in examples:
        messages.append({"role": role, "content": content})
    messages.append(
        {
            "role": "user",
            "content": QUAL_DIRECTIONS if qualification else HIT_DIRECTIONS,
        }
    )
    messages.append({"role": "user", "content": dialogue})
    return messages


def populate_examples() -> List[tuple[str, str]]:
    """Return in-context examples for the qualification tasks."""

    examples: List[tuple[str, str]] = []
    for example in EXAMPLE_DATA:
        game_state = pm.VoteGameState.fromArray(example)
        description = make_question_description(game_state, context=None, unit=None)

        question = qual_question(game_state, "area")
        model_response = mec_qual_description(game_state)
        correct_answer = _get_option_letter(game_state, pm.run_mec(game_state))
        examples.append(("user", description + question))
        examples.append(("assistant", model_response))
        examples.append(("assistant", correct_answer))
    return examples


def _get_option_letter(game_state: pm.VoteGameState, result: str) -> str:
    """Return the option letter for the winning proposal."""

    index = game_state.getVotes().index(result)
    return OPTIONS[index]


def build_request_items(
    df_exploded: pd.DataFrame,
    *,
    examples: Sequence[tuple[str, str]],
    qualification: bool,
    show_chart: bool,
    chart_type: str,
    contexts: Mapping[str, Mapping[str, str]],
) -> List[RequestItem]:
    """Build request items for each scenario row."""

    request_items: List[RequestItem] = []
    for row_id, row in df_exploded.iterrows():
        question_index = int(row["question"]) - 1
        context_name = ORDER[question_index] if question_index < len(ORDER) else None
        if qualification:
            context_name = None
        game_state = pm.decode_gameState(row["scenario_json"])
        messages = make_dialogue(
            game_state,
            context_name,
            examples,
            qualification,
            show_chart,
            chart_type,
            contexts,
        )
        request_items.append(
            RequestItem(
                row_id=int(row_id),
                participant_id=str(row["participant_id"]),
                scenario_hash=row["scenario_hash"],
                question=int(row["question"]),
                proposals=list(game_state.getVotes()),
                messages=messages,
            )
        )
    return request_items


def limit_request_items(
    request_items: Sequence[RequestItem],
    *,
    max_prompts: Optional[int],
    qualification: bool,
    qualification_sampling: str,
) -> List[RequestItem]:
    """Apply sampling and prompt limits for request items.

    Args:
        request_items: Prepared request items for a condition.
        max_prompts: Maximum number of prompts to keep, or None for all.
        qualification: Whether this is a qualification run.
        qualification_sampling: Qualification sampling unit (`prompt` or `row`).

    Returns:
        A list of request items limited to `max_prompts` prompts.
    """

    working_items = list(request_items)
    if qualification and qualification_sampling == "row":
        by_participant: dict[str, List[RequestItem]] = {}
        participant_order: List[str] = []
        for item in working_items:
            participant = item.participant_id
            if participant not in by_participant:
                by_participant[participant] = []
                participant_order.append(participant)
            by_participant[participant].append(item)

        sampled_by_row: List[RequestItem] = []
        for participant in participant_order:
            bucket = sorted(by_participant[participant], key=lambda item: item.question)
            if not bucket:
                continue
            participant_suffix = participant.rsplit("_", maxsplit=1)[-1]
            try:
                participant_index = int(participant_suffix)
            except ValueError:
                participant_index = len(sampled_by_row)
            pick_index = participant_index % len(bucket)
            sampled_by_row.append(bucket[pick_index])
        working_items = sampled_by_row

    if max_prompts is None or max_prompts >= len(working_items):
        return working_items
    if not qualification:
        return list(working_items[:max_prompts])
    if qualification_sampling == "row":
        return list(working_items[:max_prompts])

    by_question: dict[int, List[RequestItem]] = {}
    for item in working_items:
        by_question.setdefault(int(item.question), []).append(item)

    ordered_questions = sorted(by_question)
    limited: List[RequestItem] = []
    round_index = 0
    while len(limited) < max_prompts:
        added_this_round = False
        for question in ordered_questions:
            bucket = by_question[question]
            if round_index < len(bucket):
                limited.append(bucket[round_index])
                added_this_round = True
                if len(limited) >= max_prompts:
                    break
        if not added_this_round:
            break
        round_index += 1
    return limited


def estimate_cost_for_model(
    *,
    model_id: str,
    sampled_requests: Sequence[SampledRequest],
    max_completion_tokens: Optional[int],
) -> CostEstimate:
    """Estimate token usage and cost for a model without sending requests."""

    try:
        supports_reasoning = bool(litellm.supports_reasoning(model_id))
    except Exception:
        supports_reasoning = False
    if max_completion_tokens is None and supports_reasoning:
        logging.warning(
            "Cost estimate for reasoning model %s may be inaccurate. "
            "Consider passing --max-completion-tokens.",
            model_id,
        )

    try:
        max_tokens_raw = litellm.get_max_tokens(model_id)
    except Exception as err:
        logging.warning(
            "Could not fetch max context tokens for %s; continuing with "
            "partial dry-run estimate (%s).",
            model_id,
            err,
        )
        max_tokens_raw = None

    if isinstance(max_tokens_raw, str):
        max_tokens = int(max_tokens_raw)
    else:
        max_tokens = int(max_tokens_raw) if max_tokens_raw is not None else None

    total_prompt_tokens = 0
    prompt_tokens_known = True
    total_completion_tokens: Optional[int] = 0
    total_prompt_cost = 0.0
    total_completion_cost = 0.0
    costs_known = True

    for sampled_request in sampled_requests:
        prompt_tokens: Optional[int]
        try:
            prompt_tokens = int(
                litellm.token_counter(
                    model=model_id,
                    messages=sampled_request.messages,
                )
            )
            total_prompt_tokens += prompt_tokens
        except Exception as err:
            if prompt_tokens_known:
                logging.warning(
                    "Could not compute prompt tokens for %s; token totals will "
                    "be reported as unknown (%s).",
                    model_id,
                    err,
                )
            prompt_tokens_known = False
            prompt_tokens = None

        if max_completion_tokens is not None:
            completion_tokens_for_cost = max(max_completion_tokens, 0)
            assumed_completion_tokens = completion_tokens_for_cost
        elif max_tokens is not None and prompt_tokens is not None:
            available_completion_tokens = max(max_tokens - prompt_tokens, 0)
            completion_tokens_for_cost = available_completion_tokens
            assumed_completion_tokens = available_completion_tokens
        else:
            completion_tokens_for_cost = None
            assumed_completion_tokens = None

        if total_completion_tokens is not None:
            if assumed_completion_tokens is None:
                total_completion_tokens = None
            else:
                total_completion_tokens += assumed_completion_tokens

        if (
            costs_known
            and prompt_tokens is not None
            and completion_tokens_for_cost is not None
        ):
            try:
                prompt_cost, completion_cost = litellm.cost_per_token(
                    model=model_id,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens_for_cost,
                )
                total_prompt_cost += float(prompt_cost or 0.0)
                total_completion_cost += float(completion_cost or 0.0)
            except Exception as err:
                logging.warning(
                    "Could not compute token costs for %s; costs will be reported "
                    "as unknown (%s).",
                    model_id,
                    err,
                )
                costs_known = False

    return CostEstimate(
        model_id=model_id,
        requests=len(sampled_requests),
        prompt_tokens=total_prompt_tokens if prompt_tokens_known else None,
        assumed_completion_tokens=total_completion_tokens,
        prompt_cost_usd=total_prompt_cost if costs_known else None,
        completion_cost_usd=total_completion_cost if costs_known else None,
        total_cost_usd=(
            total_prompt_cost + total_completion_cost if costs_known else None
        ),
    )


def print_cost_estimate(cost_estimate: CostEstimate, *, label: str) -> None:
    """Print the cost estimate in a readable format."""

    prompt_tokens = (
        "unknown"
        if cost_estimate.prompt_tokens is None
        else str(cost_estimate.prompt_tokens)
    )
    completion_tokens = (
        "unknown"
        if cost_estimate.assumed_completion_tokens is None
        else str(cost_estimate.assumed_completion_tokens)
    )
    prompt_cost = (
        "unknown"
        if cost_estimate.prompt_cost_usd is None
        else f"{cost_estimate.prompt_cost_usd:.6f}"
    )
    completion_cost = (
        "unknown"
        if cost_estimate.completion_cost_usd is None
        else f"{cost_estimate.completion_cost_usd:.6f}"
    )
    total_cost = (
        "unknown"
        if cost_estimate.total_cost_usd is None
        else f"{cost_estimate.total_cost_usd:.6f}"
    )
    print(
        "\n".join(
            [
                f"Model: {cost_estimate.model_id}",
                f"Condition: {label}",
                f"Requests: {cost_estimate.requests}",
                f"Prompt tokens: {prompt_tokens}",
                f"Assumed completion tokens: {completion_tokens}",
                f"Prompt cost (USD): {prompt_cost}",
                f"Completion cost (USD): {completion_cost}",
                f"Total cost (USD): {total_cost}",
                "",
            ]
        )
    )


def load_completed_keys(
    output_path: str,
    *,
    samples: int,
) -> set[tuple[str, int]]:
    """Load completed scenario keys from an existing output file."""

    if not os.path.exists(output_path):
        return set()

    try:
        existing = pd.read_csv(output_path)
    except (OSError, pd.errors.ParserError):
        return set()

    if "question" not in existing.columns:
        return set()
    if "text" not in existing.columns:
        return set()

    completed: set[tuple[str, int]] = set()
    for _, row in existing.iterrows():
        raw_participant_id = row.get("participant_id", "")
        participant_id = (
            "" if pd.isna(raw_participant_id) else str(raw_participant_id).strip()
        )
        if not participant_id:
            raw_scenario_hash = row.get("scenario_hash", "")
            scenario_hash = (
                "" if pd.isna(raw_scenario_hash) else str(raw_scenario_hash).strip()
            )
            participant_id = f"scenario_{scenario_hash}"
        try:
            question = int(row["question"])
        except (TypeError, ValueError):
            continue

        response_value = row.get("text", "")
        if not isinstance(response_value, str):
            response_value = str(response_value)
        try:
            parsed = json.loads(response_value)
        except json.JSONDecodeError:
            parsed = []
        has_enough_samples = isinstance(parsed, list) and len(parsed) >= samples
        raw_response_label = row.get("response", "")
        response_label = (
            "" if pd.isna(raw_response_label) else str(raw_response_label).strip()
        )
        has_response_label = bool(response_label)
        if has_enough_samples and has_response_label:
            completed.add((participant_id, question))

    return completed


def append_results(
    output_path: str,
    rows: Sequence[pd.DataFrame],
    *,
    write_header: bool,
) -> None:
    """Append result rows to the output CSV."""

    if not rows:
        return
    combined = pd.concat(list(rows))
    combined.to_csv(output_path, mode="a", header=write_header, index=False)


def build_aggregate_base_rows(df: pd.DataFrame) -> dict[int, dict[str, object]]:
    """Return base rows for aggregate output keyed by original row id."""

    base_rows: dict[int, dict[str, object]] = {}
    for row_id, row in df.iterrows():
        base_rows[int(row_id)] = row.to_dict()
    return base_rows


def chunk_items(
    items: Sequence[object],
    chunk_size: int,
) -> Iterable[List[object]]:
    """Yield request items in fixed-size chunks."""

    for start in range(0, len(items), chunk_size):
        yield list(items[start : start + chunk_size])


def apply_reasoning_params(
    params: dict[str, object],
    *,
    reasoning_effort: Optional[str],
    extra_params: Optional[Mapping[str, object]],
) -> None:
    """Apply reasoning parameters to a LiteLLM request dictionary."""

    if reasoning_effort is not None:
        params["reasoning_effort"] = reasoning_effort
    if extra_params:
        params.update(extra_params)


def run_batch_requests(
    sampled_requests: Sequence[SampledRequest],
    *,
    model_config: ModelConfig,
    timeout: int,
    max_workers: int,
    temperature: Optional[float],
    max_tokens: int,
) -> List[object]:
    """Submit a batch of requests to LiteLLM and return responses."""

    request_messages = [item.messages for item in sampled_requests]
    request_params: dict[str, object] = {
        "model": model_config.model_id,
        "timeout": timeout,
        "max_workers": max_workers,
        "max_tokens": int(max_tokens),
    }
    if temperature is not None:
        request_params["temperature"] = float(temperature)
    provider_model_id = model_config.model_id.split("/", maxsplit=1)[-1]
    if provider_model_id.startswith("gpt-5") and not provider_model_id.startswith(
        "gpt-5.1"
    ):
        # gpt-5.5 rejects explicit temperature values and expects the default.
        request_params.pop("temperature", None)
    if model_config.model_id.startswith("together_ai/Qwen/"):
        # Use Together's default temperature for Qwen variants.
        request_params.pop("temperature", None)
    apply_reasoning_params(
        request_params,
        reasoning_effort=model_config.reasoning_effort,
        extra_params=model_config.extra_params,
    )
    if model_config.model_id.startswith("together_ai/"):
        # Disable LiteLLM auto reasoning defaults for Together models so
        # reasoning is only sent when explicitly configured.
        request_params["enable_reasoning_defaults"] = False
        # Ensure reasoning_content can be surfaced in a text channel when
        # providers return reasoning and leave content empty.
        request_params["merge_reasoning_content_in_choices"] = True
        # Drop unsupported optional params, but keep explicit reasoning_effort
        # when the caller has whitelisted it.
        allowed_params = request_params.get("allowed_openai_params")
        allows_reasoning = (
            isinstance(allowed_params, list) and "reasoning_effort" in allowed_params
        )
        if not allows_reasoning:
            request_params.pop("reasoning_effort", None)
        request_params["drop_params"] = True

    if model_config.model_id.startswith("together_ai/Qwen/"):
        # Together Qwen is hybrid and defaults to reasoning-on. For parity with
        # no-reasoning baselines, disable reasoning unless the config explicitly
        # requests it via reasoning_effort (e.g., "low"/"medium"/"high").
        reasoning_effort = (model_config.reasoning_effort or "").strip().lower()
        request_params.pop("reasoning_effort", None)
        if reasoning_effort and reasoning_effort != "none":
            request_params["reasoning"] = {"enabled": True}
            chat_template_kwargs = request_params.get("chat_template_kwargs")
            if isinstance(chat_template_kwargs, dict):
                chat_template_kwargs.pop("enable_thinking", None)
                chat_template_kwargs.pop("thinking", None)
                if not chat_template_kwargs:
                    request_params.pop("chat_template_kwargs", None)
        else:
            request_params["reasoning"] = {"enabled": False}
            request_params["chat_template_kwargs"] = {"enable_thinking": False}

    return list(litellm.batch_completion(messages=request_messages, **request_params))


def extract_response_text(response: object) -> str:
    """Extract response text from a LiteLLM response."""

    if isinstance(response, Exception):
        return ""

    def _extract_message_content(message: object) -> str:
        """Extract text from a message object or dict."""

        if isinstance(message, dict):
            content = message.get("content")
            reasoning = message.get("reasoning_content") or message.get("reasoning")
        else:
            content = getattr(message, "content", None)
            reasoning = getattr(message, "reasoning_content", None) or getattr(
                message, "reasoning", None
            )

        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            text_parts: List[str] = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str) and text.strip():
                        text_parts.append(text)
                        continue
                    part_content = part.get("content")
                    if isinstance(part_content, str) and part_content.strip():
                        text_parts.append(part_content)
                else:
                    text = getattr(part, "text", None)
                    if isinstance(text, str) and text.strip():
                        text_parts.append(text)
                        continue
                    part_content = getattr(part, "content", None)
                    if isinstance(part_content, str) and part_content.strip():
                        text_parts.append(part_content)
            if text_parts:
                return "\n".join(text_parts)
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning
        return ""

    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    if isinstance(response, dict):
        choices = response.get("choices", [])
        if choices:
            message = choices[0].get("message")
            message_text = _extract_message_content(message)
            if message_text:
                return message_text
            if "text" in choices[0]:
                choice_text = choices[0]["text"]
                if isinstance(choice_text, str) and choice_text.strip():
                    return choice_text
                if choice_text is not None:
                    return str(choice_text)
    if hasattr(response, "choices") and response.choices:
        first_choice = response.choices[0]
        message = getattr(first_choice, "message", None)
        message_text = _extract_message_content(message)
        if message_text:
            return message_text
        if hasattr(first_choice, "text"):
            choice_text = first_choice.text
            if isinstance(choice_text, str) and choice_text.strip():
                return choice_text
            if choice_text is not None:
                return str(choice_text)
    return ""


def extract_response_error(response: object) -> Optional[str]:
    """Extract error text from a failed or partial response object."""

    if isinstance(response, Exception):
        return str(response)
    if isinstance(response, dict):
        error = response.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str):
                return message
            return str(error)
        if error is not None:
            return str(error)
    error_attr = getattr(response, "error", None)
    if error_attr is not None:
        return str(error_attr)
    return None


def find_answer_letter(text: str, proposals: Sequence[str]) -> Optional[str]:
    """Return the option letter extracted from model output."""

    option_letters = set(OPTIONS[: len(proposals)])
    normalized = text.strip().upper()
    match = re.search(r"^\(?\s*([A-Z])\s*\)?[.)]?\s*$", normalized)
    if match:
        letter = match.group(1)
        if letter in option_letters:
            return letter
    return None


def find_answer(text: str, proposals: Sequence[str]) -> Optional[str]:
    """Return the proposal label inferred from a single-letter response."""

    letter = find_answer_letter(text, proposals)
    if letter is not None:
        return proposals[OPTIONS.index(letter)]
    return None


def build_result_row(
    *,
    item: RequestItem,
    base_row: Mapping[str, object],
    responses: Sequence[str],
    full_payloads: Sequence[str],
) -> pd.DataFrame:
    """Build a single-row DataFrame with response metadata."""

    counts: Counter[str] = Counter()
    for response_text in responses:
        answer = find_answer(response_text, item.proposals)
        if answer is not None:
            counts[answer] += 1

    probabilities = [0.0 for _ in item.proposals]
    total = sum(counts.values())
    if total > 0:
        for index, proposal in enumerate(item.proposals):
            probabilities[index] = counts.get(proposal, 0) / total

    if counts:
        answer = max(counts.items(), key=lambda pair: pair[1])[0]
    else:
        answer = None

    answer_columns = [f"probability: {proposal}" for proposal in item.proposals]
    row = dict(base_row)
    row.update({column: value for column, value in zip(answer_columns, probabilities)})
    row.update(
        {
            "participant_id": item.participant_id,
            "scenario_hash": item.scenario_hash,
            "question": int(item.question),
            "response": answer,
            "text": json.dumps(list(responses)),
            "full": json.dumps(list(full_payloads)),
        }
    )
    return pd.DataFrame([row])


def model_catalog() -> dict[str, ModelConfig]:
    """Return the supported model catalog."""

    return {
        "kimi-k2.5": ModelConfig(
            name="kimi-k2.5",
            model_id="together_ai/moonshotai/Kimi-K2.5",
        ),
        "gpt-5.1": ModelConfig(
            name="gpt-5.1-2025-11-13",
            model_id="openai/gpt-5.1-2025-11-13",
            reasoning_effort="medium",
        ),
        "gpt-5.1-no-reasoning": ModelConfig(
            name="gpt-5.1-2025-11-13-no-reasoning",
            model_id="openai/gpt-5.1-2025-11-13",
            reasoning_effort="none",
        ),
        "qwen-3.5-397b-a17b-no-reasoning": ModelConfig(
            name="Qwen_Qwen3.5-397B-A17B-no-reasoning",
            model_id="together_ai/Qwen/Qwen3.5-397B-A17B",
            reasoning_effort="none",
        ),
        "qwen-3.5-397b-a17b-low-reasoning": ModelConfig(
            name="Qwen_Qwen3.5-397B-A17B-low-reasoning",
            model_id="together_ai/Qwen/Qwen3.5-397B-A17B",
            reasoning_effort="low",
        ),
        "qwen-3.5-397b-a17b-high-reasoning": ModelConfig(
            name="Qwen_Qwen3.5-397B-A17B-high-reasoning",
            model_id="together_ai/Qwen/Qwen3.5-397B-A17B",
            reasoning_effort="high",
        ),
        "gpt-5.5-no-reasoning": ModelConfig(
            name="gpt-5.5-no-reasoning",
            model_id="openai/gpt-5.5",
            reasoning_effort="none",
        ),
        "gpt-5.5-high-reasoning": ModelConfig(
            name="gpt-5.5-high-reasoning",
            model_id="openai/gpt-5.5",
            reasoning_effort="high",
        ),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        prog="new_model_experiment",
        description="Runs the intuition experiment using LiteLLM batch completion.",
    )
    parser.add_argument(
        "filename", help="The file generated by scenario_utils.py to test on."
    )
    parser.add_argument(
        "--input_directory",
        default=INPUT_DIR,
        help="Where to look for filename",
    )
    parser.add_argument(
        "--output_directory",
        default=OUTPUT_DIR,
        help="Where to output the results.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--qualification",
        default=False,
        action="store_true",
        help=(
            "Whether to treat this as a qualification task. This asks which proposal "
            "has the biggest area depending on chart-type."
        ),
    )
    group.add_argument(
        "--show-chart",
        default=False,
        action="store_true",
        help=(
            "Whether to describe the charts shown in the human version of this "
            "experiment."
        ),
    )
    parser.add_argument(
        "--chart-type",
        choices=["area"],
        default="area",
        help="The type of chart to show (or ask for if qualification), if any.",
    )
    parser.add_argument(
        "--zero-shot",
        default=False,
        action="store_true",
        help="Whether to skip in-context examples.",
    )
    parser.add_argument(
        "--model",
        choices=sorted(model_catalog().keys()) + ["all"],
        default="all",
        help="Which model to run (defaults to all supported models).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout in seconds for each request.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of samples to collect per scenario question.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=(
            "Sampling temperature used for model calls. "
            "Omit to use model/provider default."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Estimate token usage and cost without sending requests.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=512,
        help="Completion token cap for dry-run estimates.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=32,
        help="Maximum number of worker threads for LiteLLM batch completion.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of requests to submit in each batch.",
    )
    parser.add_argument(
        "--conditions",
        choices=["both", "area", "none"],
        default="both",
        help=(
            "Which non-qualification condition(s) to run. "
            "Use 'area' or 'none' for smoke tests."
        ),
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=None,
        help=(
            "Maximum number of unique prompts to run per selected condition "
            "(for smoke tests)."
        ),
    )
    parser.add_argument(
        "--qualification-sampling",
        choices=["prompt", "row"],
        default="prompt",
        help=(
            "Qualification sampling unit: 'prompt' uses all exploded question "
            "prompts, while 'row' uses one qualification prompt per participant row."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Maximum completion tokens for each model response.",
    )
    return parser.parse_args(argv)


def resolve_models(model_name: str) -> List[ModelConfig]:
    """Return the model configurations to run."""

    catalog = model_catalog()
    if model_name == "all":
        return list(catalog.values())
    return [catalog[model_name]]


def build_output_path(
    *,
    output_directory: str,
    model_name: str,
    run_name: str,
    chart_type: str,
    qualification: bool,
    show_chart: bool,
    zero_shot: bool,
    temperature: Optional[float],
    samples: int,
    qualification_sampling: str,
) -> str:
    """Return the output file path for a model run."""

    model_dir = os.path.join(output_directory, model_name)
    run_dir = os.path.join(model_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    if temperature is None:
        temperature_value = "default"
    else:
        temperature_value = (
            int(temperature) if float(temperature).is_integer() else temperature
        )
    output_file = f"temp={temperature_value}_api=chat"
    output_file += f"_qualification={qualification}_show-charts={show_chart}"
    output_file += f"_chart-type={chart_type}_samples={samples}"
    if qualification and qualification_sampling != "prompt":
        output_file += f"_qual-sampling={qualification_sampling}"
    output_file += f"_zero-shot={zero_shot}.csv"

    return os.path.join(run_dir, output_file)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run the batch experiment workflow."""

    args = parse_args(argv)
    temperature = float(args.temperature) if args.temperature is not None else None

    if os.path.isabs(args.filename) or os.path.sep in args.filename:
        filename = args.filename
    else:
        filename = os.path.join(args.input_directory, args.filename)
    base_filename = os.path.basename(filename)

    file_vars = filename_to_variables(base_filename)
    maximize = file_vars.get("maximize", ["True"])[0] == "True"

    contexts = build_contexts(maximize)

    logging.info("Reading scenarios from %s", filename)
    df = pd.read_csv(filename)

    num_scenarios = int(file_vars.get("num-scenarios", [len(ORDER)])[0])
    if num_scenarios > len(ORDER):
        raise ValueError("num-scenarios exceeds available question types")

    df_exploded = explode_df_with_fehr(df, num_scenarios)

    examples = [] if args.zero_shot else populate_examples()

    if args.samples < 1:
        raise ValueError("samples must be at least 1")
    if args.max_prompts is not None and args.max_prompts < 1:
        raise ValueError("max-prompts must be at least 1")
    if temperature == 0.0 and args.samples > 1:
        logging.info(
            "temperature=0 detected; reducing samples from %s to 1",
            args.samples,
        )
        args.samples = 1

    run_name = os.path.splitext(base_filename)[0]
    condition_args_list: List[dict[str, object]]
    if args.qualification:
        condition_args_list = [
            {"show_chart": False, "chart_type": args.chart_type},
        ]
    else:
        if args.conditions == "area":
            condition_args_list = [{"show_chart": True, "chart_type": "area"}]
        elif args.conditions == "none":
            condition_args_list = [{"show_chart": False, "chart_type": "area"}]
        else:
            condition_args_list = [
                {"show_chart": False, "chart_type": "area"},
                {"show_chart": True, "chart_type": "area"},
            ]

    for model_config in resolve_models(args.model):
        logging.info("Querying model %s", model_config.model_id)
        for condition_args in condition_args_list:
            if args.qualification:
                condition_label = f"qualification-{args.chart_type}"
            else:
                condition_label = "area" if condition_args["show_chart"] else "none"
            request_items = build_request_items(
                df_exploded,
                examples=examples,
                qualification=args.qualification,
                show_chart=bool(condition_args["show_chart"]),
                chart_type=str(condition_args["chart_type"]),
                contexts=contexts,
            )
            should_limit = args.max_prompts is not None or (
                args.qualification and args.qualification_sampling == "row"
            )
            if should_limit:
                request_items = limit_request_items(
                    request_items,
                    max_prompts=args.max_prompts,
                    qualification=args.qualification,
                    qualification_sampling=args.qualification_sampling,
                )
                if args.max_prompts is not None:
                    logging.info(
                        "Smoke limit active: %s prompt(s) for %s condition.",
                        len(request_items),
                        condition_label,
                    )
                elif args.qualification and args.qualification_sampling == "row":
                    logging.info(
                        "Qualification row sampling active: %s prompt(s).",
                        len(request_items),
                    )
            output_path = build_output_path(
                output_directory=args.output_directory,
                model_name=model_config.name,
                run_name=run_name,
                chart_type=str(condition_args["chart_type"]),
                qualification=args.qualification,
                show_chart=bool(condition_args["show_chart"]),
                zero_shot=args.zero_shot,
                temperature=temperature,
                samples=args.samples,
                qualification_sampling=args.qualification_sampling,
            )

            completed_keys = load_completed_keys(output_path, samples=args.samples)
            if completed_keys:
                logging.info(
                    "Resuming %s (%s): %s completed rows found.",
                    model_config.name,
                    condition_label,
                    len(completed_keys),
                )

            sampled_requests: List[SampledRequest] = []
            pending_indices: List[int] = []
            for index, item in enumerate(request_items):
                key = (item.participant_id, item.question)
                if key in completed_keys:
                    continue
                pending_indices.append(index)
                for sample_index in range(args.samples):
                    sampled_requests.append(
                        SampledRequest(
                            item_index=index,
                            sample_index=sample_index,
                            messages=item.messages,
                        )
                    )

            if args.dry_run:
                try:
                    estimate = estimate_cost_for_model(
                        model_id=model_config.model_id,
                        sampled_requests=sampled_requests,
                        max_completion_tokens=args.max_completion_tokens,
                    )
                except (ValueError, TypeError, KeyError) as err:
                    raise ValueError(f"Failed to estimate cost: {err}") from err
                except LITELLM_API_ERRORS as err:
                    raise RuntimeError(
                        f"LiteLLM cost estimation failed: {err}"
                    ) from err
                print_cost_estimate(estimate, label=condition_label)
                continue

            responses_by_item: dict[int, dict[str, list[str]]] = {
                index: {"responses": [], "full": []} for index in pending_indices
            }
            written_items: set[int] = set()
            wrote_header = not os.path.exists(output_path)

            aggregate_base_rows = build_aggregate_base_rows(df_exploded)

            total_requests = len(sampled_requests)
            progress = tqdm(
                total=total_requests,
                desc=f"{model_config.name} ({condition_label})",
            )
            for chunk in chunk_items(sampled_requests, args.batch_size):
                responses = run_batch_requests(
                    chunk,
                    model_config=model_config,
                    timeout=args.timeout,
                    max_workers=args.max_workers,
                    temperature=temperature,
                    max_tokens=args.max_tokens,
                )
                rows_to_write: List[pd.DataFrame] = []
                for sampled_request, response in zip(chunk, responses):
                    response_text = extract_response_text(response)
                    item_index = sampled_request.item_index
                    item = request_items[item_index]
                    if not response_text:
                        error_text = extract_response_error(response)
                        error_suffix = f" | error={error_text}" if error_text else ""
                        logging.warning(
                            "Empty response for scenario %s question %s "
                            "(sample %s)%s",
                            item.scenario_hash,
                            item.question,
                            sampled_request.sample_index,
                            error_suffix,
                        )
                    full_payload = json.dumps(
                        list(item.messages)
                        + [{"role": "assistant", "content": response_text}]
                    )
                    responses_by_item[item_index]["responses"].append(response_text)
                    responses_by_item[item_index]["full"].append(full_payload)

                    if (
                        len(responses_by_item[item_index]["responses"]) >= args.samples
                        and item_index not in written_items
                    ):
                        payloads = responses_by_item[item_index]
                        rows_to_write.append(
                            build_result_row(
                                item=item,
                                base_row=aggregate_base_rows[item.row_id],
                                responses=payloads["responses"],
                                full_payloads=payloads["full"],
                            )
                        )
                        written_items.add(item_index)
                progress.update(len(chunk))
                append_results(output_path, rows_to_write, write_header=wrote_header)
                if rows_to_write:
                    wrote_header = False
            progress.close()

            remaining_indices = [
                index for index in pending_indices if index not in written_items
            ]
            if remaining_indices:
                rows_to_write = []
                for index in remaining_indices:
                    payloads = responses_by_item[index]
                    if not payloads["responses"]:
                        continue
                    rows_to_write.append(
                        build_result_row(
                            item=request_items[index],
                            base_row=aggregate_base_rows[request_items[index].row_id],
                            responses=payloads["responses"],
                            full_payloads=payloads["full"],
                        )
                    )
                append_results(output_path, rows_to_write, write_header=wrote_header)

            logging.info("Results appended to %s", output_path)


if __name__ == "__main__":
    main()
