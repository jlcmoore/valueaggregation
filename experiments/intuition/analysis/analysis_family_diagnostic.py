"""Compare Nash with the full IE family on family-diagnostic human trials."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from common.qualification_utils import (
    load_qualification_answers,
    qualification_passed_row,
)
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import t as student_t

import value_aggregation as aggregation

ACTIONS = np.array(["one", "two", "three"], dtype=object)
MINIMUM_CONSISTENCY = 1.0 / len(ACTIONS)


@dataclass(frozen=True)
class ModelFit:
    """Store one model's likelihood, parameter count, BIC, and consistency."""

    name: str
    log_likelihood: float
    num_parameters: int
    bic: float
    consistency: float


@dataclass(frozen=True)
class HierarchicalFit:
    """Store hierarchical IE population parameters and likelihood."""

    alpha_shape: float
    beta_shape: float
    consistency: float
    log_likelihood: float


@dataclass(frozen=True)
class DiagnosticTrial:
    """Represent one trial where Nash lies outside the full IE family."""

    worker_id: str
    response: str
    nash_prediction: str
    ie_predictions: np.ndarray
    scenario_json: str


@dataclass(frozen=True)
class CrossValidationSummary:
    """Store participant-cross-fitted predictive comparisons."""

    ie_accuracy_trial_weighted: float
    nash_accuracy_trial_weighted: float
    ie_accuracy_participant_weighted: float
    nash_accuracy_participant_weighted: float
    accuracy_difference: float
    accuracy_difference_se: float
    ie_log_score: float
    nash_log_score: float
    log_score_difference: float
    log_score_difference_se: float
    log_score_t_statistic: float
    log_score_degrees_freedom: int
    log_score_p_value: float
    log_score_ci_lower: float
    log_score_ci_upper: float


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare Nash with common-alpha and hierarchical IE models on "
            "trials where Nash is outside the IE family for every alpha."
        )
    )
    parser.add_argument("--file", required=True, help="Wide MTurk/JATOS CSV export.")
    parser.add_argument(
        "--qualification-answers",
        default="mturk/qualification_answers.json",
        help="Qualification answer-key JSON.",
    )
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=18,
        help="Number of trials per participant (default: 18).",
    )
    parser.add_argument(
        "--attention-threshold",
        type=float,
        default=1.0,
        help="Minimum participant attention accuracy (default: 1.0).",
    )
    parser.add_argument(
        "--alpha-grid-size",
        type=int,
        default=201,
        help="Grid size for integrating latent alpha (default: 201).",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Participant-level cross-validation folds (default: 5).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Repeated cross-validation partitions (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=22000,
        help="Initial cross-validation random seed (default: 22000).",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path for a machine-readable result summary.",
    )
    return parser.parse_args()


def alpha_grid(grid_size: int) -> np.ndarray:
    """Return midpoint alpha values spanning zero to one.

    Parameters:
        grid_size: Number of integration points.

    Returns:
        Midpoint alpha values.
    """

    if grid_size < 2:
        raise ValueError("alpha-grid-size must be at least 2.")
    return (np.arange(grid_size) + 0.5) / grid_size


def load_filtered_participants(
    csv_path: str,
    qualification_path: str,
    num_scenarios: int,
    attention_threshold: float,
) -> pd.DataFrame:
    """Load participants using the manuscript's area and attention filters.

    Parameters:
        csv_path: Wide MTurk/JATOS CSV export.
        qualification_path: Qualification answer-key JSON.
        num_scenarios: Number of trials per participant.
        attention_threshold: Minimum participant attention accuracy.

    Returns:
        Filtered participant rows.
    """

    if not 0.0 <= attention_threshold <= 1.0:
        raise ValueError("attention-threshold must lie between zero and one.")
    participants = pd.read_csv(csv_path)
    qualification_answers = load_qualification_answers(qualification_path)
    area_columns = sorted(
        column for column in participants.columns if "q_question-stacked" in column
    )
    qualification_pass = participants.apply(
        qualification_passed_row,
        axis=1,
        columns=area_columns,
        answers=qualification_answers,
    )
    attention_columns = []
    for index in range(1, num_scenarios + 1):
        attention_columns.append(
            participants[f"Answer.q_question-{index}_attn"]
            == participants[f"Answer.q_question-{index}_attn_answer"]
        )
    attention_rate = pd.concat(attention_columns, axis=1).mean(axis=1)
    return participants[
        qualification_pass & (attention_rate >= attention_threshold)
    ].copy()


def equality_efficiency_lines(game_state) -> dict[str, tuple[float, float]]:
    """Return IE score-line intercepts and slopes for every proposal.

    This exactly matches the symmetric-group calculation in
    ``run_equality_efficiency``. Each pair defines
    ``score(alpha) = intercept + slope * alpha``.

    Parameters:
        game_state: Decoded value-aggregation game state.

    Returns:
        Proposal labels mapped to score-line intercept and slope pairs.
    """

    if not game_state.equalCredences():
        raise ValueError("Family-diagnostic analysis requires equal credences.")
    agents = game_state.getAgents()
    agent_pairs = [
        (agents[left], agents[right])
        for left in range(len(agents))
        for right in range(left + 1, len(agents))
    ]
    score_lines = {}
    for proposal, outcomes in game_state.vote_agent_outcomes.items():
        efficiency = sum(
            outcomes[agent] * game_state.getNormalizedCredence(agent)
            for agent in agents
        )
        inequality = sum(
            abs(outcomes[first] - outcomes[second]) for first, second in agent_pairs
        ) / len(agent_pairs)
        score_lines[proposal] = (efficiency, -(efficiency + inequality))
    return score_lines


def proposal_is_ie_optimal(
    score_lines: dict[str, tuple[float, float]], proposal: str
) -> bool:
    """Return whether a proposal strictly maximizes IE for some alpha.

    Parameters:
        score_lines: Proposal score-line intercepts and slopes.
        proposal: Proposal to test.

    Returns:
        True if a nonempty alpha interval in [0, 1] favors the proposal.
    """

    lower_bound = 0.0
    upper_bound = 1.0
    intercept, slope = score_lines[proposal]
    for other_proposal, (other_intercept, other_slope) in score_lines.items():
        if other_proposal == proposal:
            continue
        intercept_difference = intercept - other_intercept
        slope_difference = slope - other_slope
        if slope_difference == 0.0:
            if intercept_difference <= 0.0:
                return False
            continue
        crossing = -intercept_difference / slope_difference
        if slope_difference > 0.0:
            lower_bound = max(lower_bound, crossing)
        else:
            upper_bound = min(upper_bound, crossing)
    return lower_bound < upper_bound and upper_bound > 0.0 and lower_bound < 1.0


def grid_predictions(
    score_lines: dict[str, tuple[float, float]], grid: np.ndarray
) -> np.ndarray:
    """Return IE-optimal proposals across an alpha integration grid.

    Parameters:
        score_lines: Proposal score-line intercepts and slopes.
        grid: Alpha integration values.

    Returns:
        Proposal labels for each alpha-grid value.
    """

    proposals = np.array(list(score_lines), dtype=object)
    scores = np.vstack(
        [intercept + slope * grid for intercept, slope in score_lines.values()]
    )
    return proposals[np.argmax(scores, axis=0)]


def extract_diagnostic_trials(
    participants: pd.DataFrame,
    num_scenarios: int,
    grid: np.ndarray,
) -> list[DiagnosticTrial]:
    """Extract assigned-disagreement trials where Nash is outside IE.

    Parameters:
        participants: Filtered wide participant rows.
        num_scenarios: Number of trials per participant.
        grid: Alpha values used for hierarchical integration.

    Returns:
        Family-diagnostic trial presentations.
    """

    diagnostic_trials = []
    for _, participant in participants.iterrows():
        worker_id = str(participant["WorkerId"])
        for index in range(1, num_scenarios + 1):
            nash_prediction = str(participant[f"Input.scenario_{index}_nbs"])
            assigned_ie = str(participant[f"Input.scenario_{index}_ie"])
            if nash_prediction == assigned_ie:
                continue
            scenario_json = str(participant[f"Input.scenario_{index}_json"])
            game_state = aggregation.decode_gameState(scenario_json)
            score_lines = equality_efficiency_lines(game_state)
            if proposal_is_ie_optimal(score_lines, nash_prediction):
                continue
            diagnostic_trials.append(
                DiagnosticTrial(
                    worker_id=worker_id,
                    response=str(participant[f"Answer.q_question-{index}"]),
                    nash_prediction=nash_prediction,
                    ie_predictions=grid_predictions(score_lines, grid),
                    scenario_json=scenario_json,
                )
            )
    return diagnostic_trials


def group_trials_by_worker(
    trials: list[DiagnosticTrial],
) -> list[list[DiagnosticTrial]]:
    """Group diagnostic trials by participant.

    Parameters:
        trials: Family-diagnostic trial presentations.

    Returns:
        Participant trial lists sorted by participant identifier.
    """

    grouped: dict[str, list[DiagnosticTrial]] = {}
    for trial in trials:
        grouped.setdefault(trial.worker_id, []).append(trial)
    return [grouped[worker] for worker in sorted(grouped)]


def beta_log_weights(
    grid: np.ndarray, alpha_shape: float, beta_shape: float
) -> np.ndarray:
    """Return normalized beta-density log weights on an alpha grid.

    Parameters:
        grid: Midpoint alpha integration values.
        alpha_shape: First beta shape parameter.
        beta_shape: Second beta shape parameter.

    Returns:
        Normalized log probability masses.
    """

    log_weights = (alpha_shape - 1.0) * np.log(grid) + (beta_shape - 1.0) * np.log1p(
        -grid
    )
    return log_weights - logsumexp(log_weights)


def logistic(value: float) -> float:
    """Return a numerically stable logistic transformation.

    Parameters:
        value: Unconstrained real value.

    Returns:
        A probability strictly between zero and one.
    """

    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def consistency_from_raw(raw_value: float) -> float:
    """Map an unconstrained value to consistency in (1/3, 1).

    Parameters:
        raw_value: Unconstrained optimizer parameter.

    Returns:
        Probability assigned to a model-optimal proposal.
    """

    consistency = MINIMUM_CONSISTENCY + (1.0 - MINIMUM_CONSISTENCY) * logistic(
        raw_value
    )
    return float(np.clip(consistency, MINIMUM_CONSISTENCY + 1e-12, 1.0 - 1e-12))


def fit_hierarchical_ie(
    match_counts: np.ndarray,
    trial_counts: np.ndarray,
    grid: np.ndarray,
) -> HierarchicalFit:
    """Fit a beta population of personal alphas by marginal likelihood.

    Parameters:
        match_counts: Participant by alpha-grid IE match counts.
        trial_counts: Diagnostic-trial counts for each participant.
        grid: Midpoint alpha integration values.

    Returns:
        Fitted population alpha distribution and consistency.
    """

    def population_parameters(
        raw_parameters: np.ndarray,
    ) -> tuple[float, float, float]:
        alpha_mean = logistic(raw_parameters[0])
        concentration = math.exp(raw_parameters[1])
        return (
            alpha_mean * concentration,
            (1.0 - alpha_mean) * concentration,
            consistency_from_raw(raw_parameters[2]),
        )

    def objective(raw_parameters: np.ndarray) -> float:
        alpha_shape, beta_shape, consistency = population_parameters(raw_parameters)
        error_probability = (1.0 - consistency) / 2.0
        conditional = match_counts * math.log(consistency) + (
            trial_counts[:, None] - match_counts
        ) * math.log(error_probability)
        participant_likelihoods = logsumexp(
            conditional + beta_log_weights(grid, alpha_shape, beta_shape),
            axis=1,
        )
        return -float(participant_likelihoods.sum())

    common_index = int(np.argmax(match_counts.sum(axis=0)))
    common_mean = float(grid[common_index])
    common_rate = float(match_counts[:, common_index].sum() / trial_counts.sum())
    scaled_consistency = np.clip(
        (max(common_rate, MINIMUM_CONSISTENCY + 1e-6) - MINIMUM_CONSISTENCY)
        / (1.0 - MINIMUM_CONSISTENCY),
        1e-6,
        1.0 - 1e-6,
    )
    raw_consistency = math.log(scaled_consistency / (1.0 - scaled_consistency))
    candidate_means = [common_mean, 0.1, 0.25, 0.5, 0.75, 0.9]
    candidate_concentrations = [0.5, 2.0, 20.0, 200.0]
    starts = [
        np.array(
            [
                math.log(mean / (1.0 - mean)),
                math.log(concentration),
                raw_consistency,
            ]
        )
        for mean in candidate_means
        for concentration in candidate_concentrations
    ]
    parameter_bounds = [
        (-9.0, 9.0),
        (math.log(0.01), math.log(10000.0)),
        (-12.0, 12.0),
    ]
    fits = [
        minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=parameter_bounds,
        )
        for start in starts
    ]
    best_fit = min(fits, key=lambda fit: fit.fun)
    if not best_fit.success:
        raise RuntimeError(str(best_fit.message))
    alpha_shape, beta_shape, consistency = population_parameters(best_fit.x)
    return HierarchicalFit(
        alpha_shape=alpha_shape,
        beta_shape=beta_shape,
        consistency=consistency,
        log_likelihood=-float(best_fit.fun),
    )


def fit_consistency(matches: np.ndarray) -> tuple[float, float]:
    """Fit a common model-consistency parameter.

    Parameters:
        matches: Boolean indicators that choices match model predictions.

    Returns:
        Fitted consistency and maximized log likelihood.
    """

    total = matches.size
    successes = int(matches.sum())
    consistency = max(MINIMUM_CONSISTENCY + 1e-9, successes / total)
    error_probability = (1.0 - consistency) / 2.0
    log_likelihood = successes * math.log(consistency) + (total - successes) * math.log(
        error_probability
    )
    return consistency, log_likelihood


def participant_arrays(
    grouped_trials: list[list[DiagnosticTrial]],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Convert grouped trials into ragged response and prediction arrays.

    Parameters:
        grouped_trials: Family-diagnostic trials grouped by participant.

    Returns:
        Response, Nash-prediction, and IE-grid arrays for each participant.
    """

    responses = []
    nash_predictions = []
    ie_predictions = []
    for trials in grouped_trials:
        responses.append(np.array([trial.response for trial in trials], dtype=object))
        nash_predictions.append(
            np.array([trial.nash_prediction for trial in trials], dtype=object)
        )
        ie_predictions.append(np.vstack([trial.ie_predictions for trial in trials]))
    return responses, nash_predictions, ie_predictions


def ie_match_counts(
    responses: list[np.ndarray], ie_predictions: list[np.ndarray]
) -> np.ndarray:
    """Return each participant's IE matches over the alpha grid.

    Parameters:
        responses: Participant response arrays.
        ie_predictions: Participant trial by alpha-grid predictions.

    Returns:
        Participant by alpha-grid match counts.
    """

    return np.vstack(
        [
            (predictions == participant_responses[:, None]).sum(axis=0)
            for participant_responses, predictions in zip(responses, ie_predictions)
        ]
    )


def model_fit(
    name: str,
    log_likelihood: float,
    num_parameters: int,
    consistency: float,
    num_participants: int,
) -> ModelFit:
    """Build and return a model fit with participant-cluster BIC."""

    bic = -2.0 * log_likelihood + num_parameters * math.log(num_participants)
    return ModelFit(
        name=name,
        log_likelihood=log_likelihood,
        num_parameters=num_parameters,
        bic=bic,
        consistency=consistency,
    )


def fit_models(
    responses: list[np.ndarray],
    nash_predictions: list[np.ndarray],
    ie_predictions: list[np.ndarray],
    grid: np.ndarray,
) -> tuple[list[ModelFit], float, HierarchicalFit]:
    """Fit Nash, common-alpha IE, and hierarchical IE models.

    Parameters:
        responses: Participant response arrays.
        nash_predictions: Participant Nash-prediction arrays.
        ie_predictions: Participant IE-grid prediction arrays.
        grid: Alpha integration values.

    Returns:
        Model fits, fitted common alpha, and hierarchical IE parameters.
    """

    num_participants = len(responses)
    trial_counts = np.array([len(values) for values in responses])
    match_counts = ie_match_counts(responses, ie_predictions)
    nash_matches = np.concatenate(
        [
            predictions == participant_responses
            for participant_responses, predictions in zip(responses, nash_predictions)
        ]
    )
    nash_consistency, nash_likelihood = fit_consistency(nash_matches)
    common_counts = match_counts.sum(axis=0)
    common_index = int(np.argmax(common_counts))
    common_matches = np.concatenate(
        [
            predictions[:, common_index] == participant_responses
            for participant_responses, predictions in zip(responses, ie_predictions)
        ]
    )
    common_consistency, common_likelihood = fit_consistency(common_matches)
    hierarchical_fit = fit_hierarchical_ie(match_counts, trial_counts, grid)
    fits = [
        model_fit("Nash", nash_likelihood, 1, nash_consistency, num_participants),
        model_fit(
            "IE common alpha",
            common_likelihood,
            2,
            common_consistency,
            num_participants,
        ),
        model_fit(
            "IE hierarchical alpha",
            hierarchical_fit.log_likelihood,
            3,
            hierarchical_fit.consistency,
            num_participants,
        ),
    ]
    return fits, float(grid[common_index]), hierarchical_fit


def predictive_probabilities(
    predictions: np.ndarray,
    posterior_weights: np.ndarray,
    consistency: float,
) -> np.ndarray:
    """Return posterior predictive probabilities for all proposals."""

    error_probability = (1.0 - consistency) / 2.0
    return np.array(
        [
            error_probability
            + (consistency - error_probability)
            * posterior_weights[predictions == action].sum()
            for action in ACTIONS
        ]
    )


def cross_validate(
    responses: list[np.ndarray],
    nash_predictions: list[np.ndarray],
    ie_predictions: list[np.ndarray],
    grid: np.ndarray,
    folds: int,
    repeats: int,
    seed: int,
) -> CrossValidationSummary:
    """Cross-fit population parameters and score held-out personal choices."""

    num_participants = len(responses)
    if folds < 2 or folds > num_participants:
        raise ValueError("folds must be between 2 and the participant count.")
    if repeats < 1:
        raise ValueError("repeats must be at least 1.")
    trial_counts = np.array([len(values) for values in responses])
    match_counts = ie_match_counts(responses, ie_predictions)
    records = []
    for repeat in range(repeats):
        random_generator = np.random.default_rng(seed + repeat)
        partitions = np.array_split(
            random_generator.permutation(num_participants), folds
        )
        for test_indices in partitions:
            train_mask = np.ones(num_participants, dtype=bool)
            train_mask[test_indices] = False
            hierarchical_fit = fit_hierarchical_ie(
                match_counts[train_mask], trial_counts[train_mask], grid
            )
            train_nash_matches = np.concatenate(
                [
                    nash_predictions[index] == responses[index]
                    for index in np.flatnonzero(train_mask)
                ]
            )
            nash_consistency, _ = fit_consistency(train_nash_matches)
            prior_log_weights = beta_log_weights(
                grid,
                hierarchical_fit.alpha_shape,
                hierarchical_fit.beta_shape,
            )
            ie_error_probability = (1.0 - hierarchical_fit.consistency) / 2.0
            for participant in test_indices:
                for held_out in range(trial_counts[participant]):
                    held_out_matches = (
                        ie_predictions[participant][held_out]
                        == responses[participant][held_out]
                    )
                    training_matches = match_counts[participant] - held_out_matches
                    posterior_log_weights = (
                        prior_log_weights
                        + training_matches * math.log(hierarchical_fit.consistency)
                        + (trial_counts[participant] - 1 - training_matches)
                        * math.log(ie_error_probability)
                    )
                    posterior_weights = np.exp(
                        posterior_log_weights - logsumexp(posterior_log_weights)
                    )
                    ie_probabilities = predictive_probabilities(
                        ie_predictions[participant][held_out],
                        posterior_weights,
                        hierarchical_fit.consistency,
                    )
                    actual_response = responses[participant][held_out]
                    actual_index = int(np.flatnonzero(ACTIONS == actual_response)[0])
                    nash_prediction = nash_predictions[participant][held_out]
                    nash_probability = (
                        nash_consistency
                        if actual_response == nash_prediction
                        else (1.0 - nash_consistency) / 2.0
                    )
                    records.append(
                        {
                            "participant": participant,
                            "ie_log_score": math.log(ie_probabilities[actual_index]),
                            "nash_log_score": math.log(nash_probability),
                            "ie_correct": int(
                                ACTIONS[np.argmax(ie_probabilities)] == actual_response
                            ),
                            "nash_correct": int(nash_prediction == actual_response),
                        }
                    )
    scores = pd.DataFrame(records)
    participant_scores = scores.groupby("participant").mean(numeric_only=True)
    accuracy_differences = (
        participant_scores["ie_correct"] - participant_scores["nash_correct"]
    )
    log_score_differences = (
        participant_scores["ie_log_score"] - participant_scores["nash_log_score"]
    )
    cluster_scale = math.sqrt(num_participants)
    log_score_difference = float(log_score_differences.mean())
    log_score_difference_se = float(log_score_differences.std(ddof=1) / cluster_scale)
    degrees_freedom = num_participants - 1
    log_score_t_statistic = log_score_difference / log_score_difference_se
    log_score_p_value = float(
        2.0 * student_t.sf(abs(log_score_t_statistic), degrees_freedom)
    )
    confidence_margin = float(
        student_t.ppf(0.975, degrees_freedom) * log_score_difference_se
    )
    return CrossValidationSummary(
        ie_accuracy_trial_weighted=float(scores["ie_correct"].mean()),
        nash_accuracy_trial_weighted=float(scores["nash_correct"].mean()),
        ie_accuracy_participant_weighted=float(participant_scores["ie_correct"].mean()),
        nash_accuracy_participant_weighted=float(
            participant_scores["nash_correct"].mean()
        ),
        accuracy_difference=float(accuracy_differences.mean()),
        accuracy_difference_se=float(accuracy_differences.std(ddof=1) / cluster_scale),
        ie_log_score=float(participant_scores["ie_log_score"].mean()),
        nash_log_score=float(participant_scores["nash_log_score"].mean()),
        log_score_difference=log_score_difference,
        log_score_difference_se=log_score_difference_se,
        log_score_t_statistic=log_score_t_statistic,
        log_score_degrees_freedom=degrees_freedom,
        log_score_p_value=log_score_p_value,
        log_score_ci_lower=log_score_difference - confidence_margin,
        log_score_ci_upper=log_score_difference + confidence_margin,
    )


def result_summary(
    participants: pd.DataFrame,
    trials: list[DiagnosticTrial],
    grouped_trials: list[list[DiagnosticTrial]],
    fits: list[ModelFit],
    common_alpha: float,
    hierarchical_fit: HierarchicalFit,
    cross_validation: CrossValidationSummary,
) -> dict[str, object]:
    """Build and return a serializable analysis summary."""

    nash_choices = sum(trial.response == trial.nash_prediction for trial in trials)
    ie_family_choices = sum(
        trial.response in set(trial.ie_predictions) for trial in trials
    )
    trial_counts = [len(values) for values in grouped_trials]
    return {
        "sample": {
            "filtered_participants": len(participants),
            "eligible_participants": len(grouped_trials),
            "diagnostic_trial_presentations": len(trials),
            "unique_diagnostic_scenarios": len(
                {trial.scenario_json for trial in trials}
            ),
            "diagnostic_trials_per_participant": {
                "minimum": min(trial_counts),
                "median": float(np.median(trial_counts)),
                "maximum": max(trial_counts),
            },
        },
        "descriptive_choices": {
            "nash": nash_choices,
            "ie_family": ie_family_choices,
            "total": len(trials),
        },
        "models": [asdict(fit) for fit in fits],
        "common_alpha": common_alpha,
        "hierarchical_ie": asdict(hierarchical_fit),
        "cross_validation": asdict(cross_validation),
    }


def print_summary(summary: dict[str, object]) -> None:
    """Print a compact human-readable analysis summary."""

    sample = summary["sample"]
    choices = summary["descriptive_choices"]
    hierarchical = summary["hierarchical_ie"]
    cross_validation = summary["cross_validation"]
    print(
        "Sample: "
        f"{sample['filtered_participants']} filtered participants; "
        f"{sample['eligible_participants']} with diagnostic trials; "
        f"{sample['diagnostic_trial_presentations']} trial presentations; "
        f"{sample['unique_diagnostic_scenarios']} unique scenarios."
    )
    print(
        "Choices: "
        f"Nash={choices['nash']}/{choices['total']} "
        f"({choices['nash'] / choices['total']:.1%}); "
        f"IE family={choices['ie_family']}/{choices['total']} "
        f"({choices['ie_family'] / choices['total']:.1%})."
    )
    print("\nParticipant-marginal BIC (lower is better):")
    for fit in summary["models"]:
        print(
            f"  {fit['name']}: BIC={fit['bic']:.3f}; "
            f"logLik={fit['log_likelihood']:.3f}; "
            f"k={fit['num_parameters']}; q={fit['consistency']:.3f}"
        )
    print(f"  Common IE alpha={summary['common_alpha']:.4f}")
    print(
        "  Hierarchical IE alpha distribution: "
        f"Beta({hierarchical['alpha_shape']:.3f}, "
        f"{hierarchical['beta_shape']:.3f})"
    )
    print("\nCross-fitted held-out prediction:")
    print(
        "  Trial-weighted accuracy: "
        f"IE={cross_validation['ie_accuracy_trial_weighted']:.3f}; "
        f"Nash={cross_validation['nash_accuracy_trial_weighted']:.3f}"
    )
    print(
        "  Participant-weighted accuracy: "
        f"IE={cross_validation['ie_accuracy_participant_weighted']:.3f}; "
        f"Nash={cross_validation['nash_accuracy_participant_weighted']:.3f}; "
        f"difference={cross_validation['accuracy_difference']:.3f} "
        f"(cluster SE={cross_validation['accuracy_difference_se']:.3f})"
    )
    print(
        "  Participant-weighted log score: "
        f"IE={cross_validation['ie_log_score']:.4f}; "
        f"Nash={cross_validation['nash_log_score']:.4f}; "
        f"difference={cross_validation['log_score_difference']:.4f} "
        f"(cluster SE={cross_validation['log_score_difference_se']:.4f})"
    )
    print(
        "  Paired participant test of log-score difference: "
        f"t({cross_validation['log_score_degrees_freedom']})="
        f"{cross_validation['log_score_t_statistic']:.2f}; "
        f"p={cross_validation['log_score_p_value']:.4f}; "
        "95% CI=["
        f"{cross_validation['log_score_ci_lower']:.4f}, "
        f"{cross_validation['log_score_ci_upper']:.4f}]"
    )


def main() -> None:
    """Run the family-diagnostic model comparison."""

    args = parse_args()
    grid = alpha_grid(args.alpha_grid_size)
    participants = load_filtered_participants(
        args.file,
        args.qualification_answers,
        args.num_scenarios,
        args.attention_threshold,
    )
    trials = extract_diagnostic_trials(participants, args.num_scenarios, grid)
    grouped_trials = group_trials_by_worker(trials)
    if not grouped_trials:
        raise ValueError("No family-diagnostic trials found.")
    responses, nash_predictions, ie_predictions = participant_arrays(grouped_trials)
    fits, common_alpha, hierarchical_fit = fit_models(
        responses, nash_predictions, ie_predictions, grid
    )
    cross_validation = cross_validate(
        responses,
        nash_predictions,
        ie_predictions,
        grid,
        args.folds,
        args.repeats,
        args.seed,
    )
    summary = result_summary(
        participants,
        trials,
        grouped_trials,
        fits,
        common_alpha,
        hierarchical_fit,
        cross_validation,
    )
    print_summary(summary)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")


if __name__ == "__main__":
    main()
