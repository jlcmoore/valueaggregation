"""Analyze JATOS and GPT-5.1 results and plot inequality-sum agreement."""

import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from common.qualification_utils import (
    load_qualification_answers,
    qualification_columns_in_header,
    qualification_passed_group,
    qualification_passed_row,
)
from common.shared_analysis import (
    AGGREGATION_TO_COLOR,
    CONDITION_TO_COLOR,
    add_fehr,
    control,
    control_success,
    function_success,
    jatos_dfs,
    llm_dfs,
    not_null_ties,
    save_fig,
    test,
)

SCENARIO = (
    "maximize=True_num-agents=3_belief-steps=1_belief-range=3,3_"
    "action-steps=101_action-range=1,101_action-function-log=False_"
    "prevent-ties=True_agg-functions=['fehr','nash']_disagrees-only=False_"
    "num-scenarios=4_sample-size=34"
)
NUM_SCENARIOS = 4
INCLUDE_QUAL_FAILED = False
INCLUDE_ATTENTION_FAILED = False
QUAL_FAILED_ONLY = False
QUAL_TYPE = "any"
ATTENTION_THRESHOLD = 1.0
GPT5_MODEL = "gpt-5.1-2025-11-13"
GPT5_NO_REASONING_MODEL = "gpt-5.1-2025-11-13-no-reasoning"
KIMI_K2_5_MODEL = "kimi-k2.5"
AGREE_BOTH_COLOR = "grey"
AGREE_THEORY_ORDER = ("scenario_nbs", "scenario_mec", "scenario_fehr")
AGREE_THEORY_LABELS = {
    "scenario_nbs": "Nash Product",
    "scenario_mec": "Utilitarian Sum",
    "scenario_fehr": "Inequality Sum",
}
BRMS_CONFIG = {
    "human": {
        "disagree": "data/analysis/alpha_probabilities_disagree_human.csv",
        "agree": "data/analysis/alpha_probabilities_agree_human.csv",
    },
    "gpt5_1": {
        "disagree": "data/analysis/alpha_probabilities_disagree_llm.csv",
        "agree": "data/analysis/alpha_probabilities_agree_llm.csv",
    },
}
JATOS_RESULTS_DIR = "data/results/jatos"


def get_condition_df(dfs, condition):
    """Return the first dataframe with a matching condition name."""
    for df in dfs:
        if df.name == condition:
            return df
    return None


def build_inequality_sum_agreement(df, between=("scenario_nbs", "scenario_fehr")):
    """Compute agreement and disagreement success across alpha values."""
    values = {between[0]: [], between[1]: [], "both": []}
    lower_errors = {between[0]: [], between[1]: [], "both": []}
    upper_errors = {between[0]: [], between[1]: [], "both": []}

    x_vals = np.arange(0, 1.1, 0.1)
    base_df = df.copy()

    for alpha in x_vals:
        df_work = base_df.copy()
        if "scenario_fehr" in df_work:
            del df_work["scenario_fehr"]
        df_work = add_fehr(df_work, alpha=alpha)

        agreements = not_null_ties(control(df_work, between))
        disagreements = test(df_work, between)

        for function in between:
            datum, (low, high) = function_success(
                disagreements,
                normalize=True,
                function=function,
            )
            values[function].append(datum)
            lower_errors[function].append(low)
            upper_errors[function].append(high)

        datum, (low, high) = control_success(
            agreements,
            normalize=True,
            between=between,
        )
        values["both"].append(datum)
        lower_errors["both"].append(low)
        upper_errors["both"].append(high)

    return x_vals, values, lower_errors, upper_errors


def disagree_mask_three(df, columns):
    """Return mask for rows where all three theories are non-null and disagree."""
    first, second, third = columns
    all_present = df[first].notnull() & df[second].notnull() & df[third].notnull()
    all_equal = (df[first] == df[second]) & (df[second] == df[third])
    return all_present & (~all_equal)


def bootstrap_ci(proportions, n_boot=2000, alpha=0.05, rng=None):
    """Bootstrap confidence interval for a mean of 0/1 proportions."""
    if rng is None:
        rng = np.random.default_rng(0)
    if len(proportions) == 0:
        return 0.0, 0.0, 0.0
    samples = rng.choice(proportions, size=(n_boot, len(proportions)), replace=True)
    means = samples.mean(axis=1)
    low = np.quantile(means, alpha / 2)
    high = np.quantile(means, 1 - alpha / 2)
    return proportions.mean(), low, high


def pooled_disagreements(df, alphas, columns=AGREE_THEORY_ORDER):
    """Pool disagreement rows across alphas, keeping repeats per alpha."""
    pooled = {col: [] for col in columns}
    base_df = df.copy()
    for alpha in alphas:
        df_work = base_df.copy()
        if "scenario_fehr" in df_work:
            del df_work["scenario_fehr"]
        df_work = add_fehr(df_work, alpha=alpha)
        mask = disagree_mask_three(df_work, columns)
        disagreements = df_work[mask]
        for col in columns:
            pooled[col].append(
                (disagreements["response"] == disagreements[col]).to_numpy()
            )
    return {
        col: np.concatenate(pooled[col]) if pooled[col] else np.array([])
        for col in columns
    }


def plot_overall_theory_bars(area_df, none_df, participant_label, output_name):
    """Plot overall theory agreement for Area/None conditions as subplots."""
    if area_df is None or none_df is None:
        print(f"Skipping plot {output_name} because data is missing.")
        return
    alphas = np.arange(0, 1.1, 0.1)
    fig, axes = plt.subplots(ncols=2, sharey=True, figsize=(7, 3.5))
    rng = np.random.default_rng(0)

    for ax, condition, df in zip(axes, ["Area", "None"], [area_df, none_df]):
        pooled = pooled_disagreements(df, alphas, columns=AGREE_THEORY_ORDER)
        labels = [AGREE_THEORY_LABELS[col] for col in AGREE_THEORY_ORDER]
        colors = [AGGREGATION_TO_COLOR[col] for col in AGREE_THEORY_ORDER]

        means = []
        lows = []
        highs = []
        for col in AGREE_THEORY_ORDER:
            mean, low, high = bootstrap_ci(pooled[col], rng=rng)
            means.append(mean)
            lows.append(mean - low)
            highs.append(high - mean)

        ax.bar(labels, means, yerr=[lows, highs], color=colors, capsize=3)
        ax.set_title(condition)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("% Agreement")
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle(participant_label, fontsize=12)
    fig.tight_layout()
    save_fig(fig, name=output_name)
    plt.close(fig)


def agreement_success_series(df, between=("scenario_nbs", "scenario_fehr")):
    """Return binary agreement-success series for control trials."""
    agreements = not_null_ties(control(df, between))
    if agreements.empty:
        return np.array([])
    success = (agreements["response"] == agreements[between[0]]) & (
        agreements["response"] == agreements[between[1]]
    )
    return success.to_numpy()


def plot_agreement_by_condition(
    human_groups,
    llm_groups,
    output_name,
    brms_agree_by_label=None,
):
    """Plot agreement rates by condition for humans and LLMs."""
    fig, axes = plt.subplots(ncols=2, sharey=True, figsize=(8, 3.5))
    for ax, title, groups in zip(
        axes,
        ["Humans", "LLMs"],
        [human_groups, llm_groups],
    ):
        x = np.arange(len(groups))
        width = 0.5
        area_means = []
        area_lows = []
        area_highs = []
        none_means = []
        none_lows = []
        none_highs = []
        labels = []
        for label, area_df, none_df in groups:
            labels.append(label)
            if area_df is None or none_df is None:
                area_means.append(0)
                area_lows.append(0)
                area_highs.append(0)
                none_means.append(0)
                none_lows.append(0)
                none_highs.append(0)
                continue

            brms_data = None
            if brms_agree_by_label is not None:
                brms_data = brms_agree_by_label.get(label)

            area_stats = brms_agree_stats(brms_data, "Area")
            if area_stats is None:
                area_series = agreement_success_series(area_df)
                mean, low, high = bootstrap_ci(area_series)
            else:
                mean, low, high = area_stats
            area_means.append(mean)
            area_lows.append(mean - low)
            area_highs.append(high - mean)

            none_stats = brms_agree_stats(brms_data, "None")
            if none_stats is None:
                none_series = agreement_success_series(none_df)
                mean, low, high = bootstrap_ci(none_series)
            else:
                mean, low, high = none_stats
            none_means.append(mean)
            none_lows.append(mean - low)
            none_highs.append(high - mean)

        area_color = CONDITION_TO_COLOR.get("Area", "tab:blue")
        none_color = CONDITION_TO_COLOR.get("None", "tab:orange")
        ax.bar(
            x - width / 2,
            area_means,
            width,
            yerr=[area_lows, area_highs],
            color=area_color,
            capsize=3,
            label="Area",
        )
        ax.bar(
            x + width / 2,
            none_means,
            width,
            yerr=[none_lows, none_highs],
            color=none_color,
            capsize=3,
            label="None",
        )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("% Agreement")
        ax.legend(loc="lower right")
    fig.tight_layout()
    save_fig(fig, name=output_name)
    plt.close(fig)


def plot_inequality_sum_disagreements(
    df,
    condition,
    participant_label,
    output_name,
    brms_data=None,
):
    """Plot disagreement support across alpha values with BRMS overlay."""
    if df is None:
        print(f"Skipping plot {output_name} because data is missing.")
        return
    if df is None:
        print(f"Skipping plot {output_name} because data is missing.")
        return
    between = ("scenario_nbs", "scenario_fehr")
    names = {"scenario_nbs": "Nash Product", "scenario_fehr": "Inequality Sum"}

    x_vals, values, lower, upper = build_inequality_sum_agreement(
        df,
        between=between,
    )

    with plt.rc_context({"font.size": 10}):
        fig, ax1 = plt.subplots(figsize=(6, 3.5))
        fig.suptitle(f"{participant_label} - {condition}", fontsize=12)

        symbols = ["o", "+"]
        for function, symbol in zip(between, symbols):
            ax1.errorbar(
                x_vals,
                values[function],
                yerr=[lower[function], upper[function]],
                linestyle=":",
                marker=symbol,
                color=AGGREGATION_TO_COLOR[function],
                label=names[function],
                capsize=2,
            )

        condition_key = condition.lower()
        for function, choice in zip(between, ["Nash", "IE"]):
            series = brms_series_for_condition(brms_data, condition_key, choice)
            if not series:
                continue
            alpha, mean, lower_ci, upper_ci = zip(*series)
            color = AGGREGATION_TO_COLOR[function]
            ax1.fill_between(alpha, lower_ci, upper_ci, color=color, alpha=0.15)
            ax1.plot(alpha, mean, color=color, linewidth=1.2)

        ax1.set_xlabel("Inequality Aversion")
        ax1.set_ylabel("% Agreement")
        ax1.legend(loc="lower right")
        ax1.set_xticks(x_vals)
        ax1.set_ylim(0, 0.95)

        fig.tight_layout()
        save_fig(fig, name=output_name)
        plt.close(fig)


def find_jatos_file(condition):
    """Find the JATOS CSV file for a given condition."""
    matches = [
        name
        for name in os.listdir(JATOS_RESULTS_DIR)
        if name.endswith(".csv") and f"chart_type={condition}" in name
    ]
    if not matches:
        raise FileNotFoundError(f"No JATOS file for condition: {condition}")
    if len(matches) > 1:
        raise ValueError(f"Multiple JATOS files for condition: {condition}")
    return os.path.join(JATOS_RESULTS_DIR, matches[0])


def qualification_worker_sets(path):
    """Return qual-passed and qual-failed worker sets for a raw CSV."""
    if not path:
        return set(), set()
    answers = load_qualification_answers()
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing headers in {path}")
        qual_columns, _ = qualification_columns_in_header(reader.fieldnames, answers)
        if not qual_columns:
            raise ValueError(
                f"Missing qualification answer columns in {path}; "
                "cannot recompute qual pass/fail."
            )
        worker_pass = {}
        for row in reader:
            worker_id = str(row.get("WorkerId", "")).strip()
            if not worker_id:
                continue
            row_pass = qualification_passed_row(row, qual_columns, answers)
            if worker_id not in worker_pass:
                worker_pass[worker_id] = row_pass
            else:
                worker_pass[worker_id] = worker_pass[worker_id] and row_pass
    passed = {wid for wid, ok in worker_pass.items() if ok}
    failed = set(worker_pass) - passed
    return passed, failed


def summarize_jatos_quality(path):
    """Return qualification/attention failure counts and included total."""
    if not path:
        return {
            "failed_qual_only": 0,
            "failed_attention_only": 0,
            "failed_both": 0,
            "failed_attention": 0,
            "included": 0,
            "total": 0,
        }
    all_workers = set()
    failed_attention = set()
    qual_true = set()

    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing headers in {path}")
        qual_answers = load_qualification_answers()
        qual_columns, _ = qualification_columns_in_header(
            reader.fieldnames, qual_answers
        )
        if not qual_columns:
            raise ValueError(
                f"Missing qualification answer columns in {path}; "
                "cannot recompute qual pass/fail."
            )
        for row in reader:
            worker_id = row.get("WorkerId", "").strip()
            if not worker_id:
                continue
            all_workers.add(worker_id)
            if qualification_passed_row(row, qual_columns, qual_answers):
                qual_true.add(worker_id)

            for idx in range(1, NUM_SCENARIOS + 1):
                attn = row.get(f"Answer.q_question-{idx}_attn")
                attn_answer = row.get(f"Answer.q_question-{idx}_attn_answer")
                if attn is None or attn_answer is None:
                    continue
                if attn != attn_answer:
                    failed_attention.add(worker_id)
                    break

    failed_qual = all_workers - qual_true
    failed_both = failed_qual & failed_attention
    failed_qual_only = failed_qual - failed_attention
    failed_attention_only = failed_attention - failed_qual
    included = all_workers - failed_attention - failed_qual
    return {
        "failed_qual_only": len(failed_qual_only),
        "failed_attention_only": len(failed_attention_only),
        "failed_both": len(failed_both),
        "failed_qual": len(failed_qual),
        "failed_attention": len(failed_attention),
        "included": len(included),
        "total": len(all_workers),
    }


def load_brms_alpha_probabilities(path):
    """Load compact brms alpha probabilities into nested dicts."""
    if path is None:
        return None
    if not os.path.exists(path):
        print(f"Warning: missing BRMS file: {path}")
        return None
    data = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            condition = row["condition"]
            choice = row["choice"]
            alpha = float(row["alpha"])
            mean = float(row["mean"])
            lower = float(row["lower"])
            upper = float(row["upper"])
            data.setdefault(condition, {}).setdefault(choice, []).append(
                (alpha, mean, lower, upper)
            )
    for condition, choices in data.items():
        for choice in choices:
            choices[choice].sort(key=lambda item: item[0])
    return data


def load_brms_agree_probabilities(path):
    """Load brms agree probabilities into nested dicts."""
    if path is None:
        return None
    if not os.path.exists(path):
        print(f"Warning: missing BRMS file: {path}")
        return None
    data = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            condition = row["condition"]
            choice = row["choice"]
            mean = float(row["mean"])
            lower = float(row["lower"])
            upper = float(row["upper"])
            data.setdefault(condition, {})[choice] = (mean, lower, upper)
    return data


def split_by_qualification(df):
    """Return mutually exclusive passed/failed dataframes by worker."""
    if "WorkerId" not in df.columns:
        return df, df.iloc[0:0]
    qual_answers = load_qualification_answers()
    qual_columns, missing = qualification_columns_in_header(df.columns, qual_answers)
    if not qual_columns:
        raise ValueError(
            "Missing qualification answer columns; cannot recompute pass/fail."
        )
    if missing:
        print(
            "Warning: missing qualification columns in dataframe: " + ", ".join(missing)
        )
    qual_passed = set()
    qual_failed = set()
    for worker_id, group in df.groupby("WorkerId"):
        if qualification_passed_group(group, qual_columns, qual_answers):
            qual_passed.add(worker_id)
        else:
            qual_failed.add(worker_id)
    passed_df = df[df["WorkerId"].isin(qual_passed)]
    failed_df = df[df["WorkerId"].isin(qual_failed)]
    return passed_df, failed_df


def report_qualification_split(label, passed_df, failed_df):
    """Report sizes and overlaps for qualification splits."""
    passed_workers = set(passed_df["WorkerId"].unique())
    failed_workers = set(failed_df["WorkerId"].unique())
    overlap = passed_workers & failed_workers
    print(
        f"{label}: qual passed workers={len(passed_workers)}, "
        f"qual failed workers={len(failed_workers)}, "
        f"overlap={len(overlap)}"
    )


def report_qualification_values(label, df):
    """Report qualification answer coverage for debugging."""
    qual_answers = load_qualification_answers()
    qual_columns, missing = qualification_columns_in_header(df.columns, qual_answers)
    print(
        f"{label}: qual columns present={len(qual_columns)}, " f"missing={len(missing)}"
    )


def brms_series_for_condition(brms_data, condition, choice):
    """Return alpha/mean/lower/upper series for a condition and choice."""
    if brms_data is None:
        return None
    return brms_data.get(condition, {}).get(choice)


def brms_agree_stats(brms_data, condition, choice="Nash/IE"):
    """Return mean/lower/upper for a condition choice from agree data."""
    if brms_data is None:
        return None
    return brms_data.get(condition, {}).get(choice)


def compute_choice_distribution(df, mode):
    """Return choice proportions for disagreement or agreement rows."""
    if df is None or df.empty:
        return None
    if mode == "disagree":
        subset = df[df["scenario_nbs"] != df["scenario_fehr"]]
        if subset.empty:
            return None
        response = subset["response"]
        nash_action = subset["scenario_nbs"]
        ie_action = subset["scenario_fehr"]
        labels = np.where(
            response == nash_action,
            "Nash",
            np.where(response == ie_action, "IE", "Decoy"),
        )
        counts = {
            "Nash": np.sum(labels == "Nash"),
            "IE": np.sum(labels == "IE"),
            "Decoy": np.sum(labels == "Decoy"),
        }
    elif mode == "agree":
        subset = df[df["scenario_nbs"] == df["scenario_fehr"]]
        if subset.empty:
            return None
        response = subset["response"]
        target_action = subset["scenario_nbs"]
        labels = np.where(response == target_action, "Nash/IE", "Decoy")
        counts = {
            "Nash/IE": np.sum(labels == "Nash/IE"),
            "Decoy": np.sum(labels == "Decoy"),
        }
    else:
        raise ValueError(f"Unknown mode: {mode}")
    total = sum(counts.values())
    if total == 0:
        return None
    return {key: value / total for key, value in counts.items()}


def js_divergence(dist_a, dist_b, keys, epsilon=1e-12):
    """Compute Jensen-Shannon divergence between two distributions."""
    vec_a = np.array([dist_a.get(key, 0.0) for key in keys], dtype=float)
    vec_b = np.array([dist_b.get(key, 0.0) for key in keys], dtype=float)
    vec_a = np.clip(vec_a, epsilon, 1.0)
    vec_b = np.clip(vec_b, epsilon, 1.0)
    vec_a = vec_a / vec_a.sum()
    vec_b = vec_b / vec_b.sum()
    mean_vec = 0.5 * (vec_a + vec_b)

    def kl_divergence(vec_p, vec_q):
        return float(np.sum(vec_p * np.log2(vec_p / vec_q)))

    return 0.5 * kl_divergence(vec_a, mean_vec) + 0.5 * kl_divergence(vec_b, mean_vec)


def bootstrap_jsd_ci(df_a, df_b, mode, keys, n_boot=1000, alpha=0.05, rng=None):
    """Bootstrap JSD confidence interval between two groups."""
    if rng is None:
        rng = np.random.default_rng(0)
    if df_a.empty or df_b.empty:
        return None

    def subset_df(df, mode_value):
        if mode_value == "disagree":
            return df[df["scenario_nbs"] != df["scenario_fehr"]]
        if mode_value == "agree":
            return df[df["scenario_nbs"] == df["scenario_fehr"]]
        raise ValueError(f"Unknown mode: {mode_value}")

    sub_a = subset_df(df_a, mode)
    sub_b = subset_df(df_b, mode)
    if sub_a.empty or sub_b.empty:
        return None

    idx_a = np.arange(len(sub_a))
    idx_b = np.arange(len(sub_b))
    stats = []
    for _ in range(n_boot):
        sample_a = sub_a.iloc[rng.choice(idx_a, size=len(idx_a), replace=True)]
        sample_b = sub_b.iloc[rng.choice(idx_b, size=len(idx_b), replace=True)]
        dist_a = compute_choice_distribution(sample_a, mode)
        dist_b = compute_choice_distribution(sample_b, mode)
        if dist_a is None or dist_b is None:
            continue
        stats.append(js_divergence(dist_a, dist_b, keys))

    if not stats:
        return None
    stats = np.array(stats)
    mean = float(np.mean(stats))
    low = float(np.quantile(stats, alpha / 2))
    high = float(np.quantile(stats, 1 - alpha / 2))
    low = max(0.0, low)
    high = min(1.0, high)
    return mean, low, high


def plot_js_divergence_by_condition(
    base_label,
    base_area,
    base_none,
    comparisons,
    output_name,
):
    """Plot JSD vs baseline for Area/None conditions (disagree + agree)."""
    conditions = ["Area", "None"]
    base_dists = {
        "disagree": {
            "Area": compute_choice_distribution(base_area, "disagree"),
            "None": compute_choice_distribution(base_none, "disagree"),
        },
        "agree": {
            "Area": compute_choice_distribution(base_area, "agree"),
            "None": compute_choice_distribution(base_none, "agree"),
        },
    }
    fig, axes = plt.subplots(ncols=2, sharey=True, figsize=(10, 3.5))
    x_vals = np.arange(len(conditions))
    width = 0.22
    offsets = np.linspace(-width, width, num=len(comparisons))

    rng = np.random.default_rng(0)
    for ax, mode, title, keys in zip(
        axes,
        ["disagree", "agree"],
        ["Disagree", "Agree"],
        [["Nash", "IE", "Decoy"], ["Nash/IE", "Decoy"]],
    ):
        max_upper = 0.0
        for offset, (label, area_df, none_df, color) in zip(offsets, comparisons):
            dist_area = compute_choice_distribution(area_df, mode)
            dist_none = compute_choice_distribution(none_df, mode)
            jsd_area = np.nan
            jsd_none = np.nan
            area_err = (np.nan, np.nan)
            none_err = (np.nan, np.nan)
            if base_dists[mode]["Area"] and dist_area:
                jsd_area = js_divergence(base_dists[mode]["Area"], dist_area, keys)
                ci = bootstrap_jsd_ci(base_area, area_df, mode, keys, rng=rng)
                if ci is not None:
                    _, low, high = ci
                    lower_bound = min(low, jsd_area)
                    upper_bound = max(high, jsd_area)
                    area_err = (
                        max(0.0, jsd_area - lower_bound),
                        max(0.0, upper_bound - jsd_area),
                    )
            if base_dists[mode]["None"] and dist_none:
                jsd_none = js_divergence(base_dists[mode]["None"], dist_none, keys)
                ci = bootstrap_jsd_ci(base_none, none_df, mode, keys, rng=rng)
                if ci is not None:
                    _, low, high = ci
                    lower_bound = min(low, jsd_none)
                    upper_bound = max(high, jsd_none)
                    none_err = (
                        max(0.0, jsd_none - lower_bound),
                        max(0.0, upper_bound - jsd_none),
                    )
            err = np.array([[area_err[0], none_err[0]], [area_err[1], none_err[1]]])
            for value, upper in zip([jsd_area, jsd_none], [area_err[1], none_err[1]]):
                if not np.isnan(value) and not np.isnan(upper):
                    max_upper = max(max_upper, value + upper)
            ax.bar(
                x_vals + offset,
                [jsd_area, jsd_none],
                width,
                color=color,
                label=label,
                yerr=err,
                capsize=3,
            )
        ax.set_xticks(x_vals)
        ax.set_xticklabels(conditions)
        ax.set_title(title)
        if max_upper > 0:
            ax.set_ylim(0, max_upper * 1.1)
        else:
            ax.set_ylim(bottom=0)

    axes[0].set_ylabel("Jensen-Shannon Divergence")
    axes[0].legend(loc="upper left")
    fig.suptitle(f"Choice JSD vs {base_label}")
    fig.tight_layout()
    save_fig(fig, name=output_name)
    plt.close(fig)


def main():
    """Generate inequality-sum agreement plots for human and GPT-5.1 data."""
    brms_human_disagree = load_brms_alpha_probabilities(
        BRMS_CONFIG["human"]["disagree"]
    )
    brms_gpt5_disagree = load_brms_alpha_probabilities(
        BRMS_CONFIG["gpt5_1"]["disagree"]
    )
    brms_human_agree = load_brms_agree_probabilities(BRMS_CONFIG["human"]["agree"])
    brms_gpt5_agree = load_brms_agree_probabilities(BRMS_CONFIG["gpt5_1"]["agree"])
    jatos_area_dfs = jatos_dfs(
        SCENARIO,
        num_scenarios=NUM_SCENARIOS,
        desired_condition="chart_type=area",
        delete_user_fail_check=not INCLUDE_ATTENTION_FAILED,
        require_qualification=not INCLUDE_QUAL_FAILED,
        aggregate_responses=False,
        plot=False,
        verbose=False,
    )
    jatos_none_dfs = jatos_dfs(
        SCENARIO,
        num_scenarios=NUM_SCENARIOS,
        desired_condition="chart_type=none",
        delete_user_fail_check=not INCLUDE_ATTENTION_FAILED,
        require_qualification=not INCLUDE_QUAL_FAILED,
        aggregate_responses=False,
        plot=False,
        verbose=False,
    )
    jatos_area_all_dfs = jatos_dfs(
        SCENARIO,
        num_scenarios=NUM_SCENARIOS,
        desired_condition="chart_type=area",
        delete_user_fail_check=True,
        require_qualification=False,
        aggregate_responses=False,
        plot=False,
        verbose=False,
    )
    jatos_none_all_dfs = jatos_dfs(
        SCENARIO,
        num_scenarios=NUM_SCENARIOS,
        desired_condition="chart_type=none",
        delete_user_fail_check=True,
        require_qualification=False,
        aggregate_responses=False,
        plot=False,
        verbose=False,
    )
    gpt5_area_dfs = llm_dfs(
        SCENARIO,
        desired_condition="show-charts=True_chart-type=area",
        desired_model=GPT5_MODEL,
        qualification=False,
        aggregate_responses=False,
    )
    gpt5_none_dfs = llm_dfs(
        SCENARIO,
        desired_condition="show-charts=False_chart-type=area",
        desired_model=GPT5_MODEL,
        qualification=False,
        aggregate_responses=False,
    )
    gpt5_nr_area_dfs = llm_dfs(
        SCENARIO,
        desired_condition="show-charts=True_chart-type=area",
        desired_model=GPT5_NO_REASONING_MODEL,
        qualification=False,
        aggregate_responses=False,
    )
    gpt5_nr_none_dfs = llm_dfs(
        SCENARIO,
        desired_condition="show-charts=False_chart-type=area",
        desired_model=GPT5_NO_REASONING_MODEL,
        qualification=False,
        aggregate_responses=False,
    )
    kimi_area_dfs = llm_dfs(
        SCENARIO,
        desired_condition="show-charts=True_chart-type=area",
        desired_model=KIMI_K2_5_MODEL,
        qualification=False,
        aggregate_responses=False,
    )
    kimi_none_dfs = llm_dfs(
        SCENARIO,
        desired_condition="show-charts=False_chart-type=area",
        desired_model=KIMI_K2_5_MODEL,
        qualification=False,
        aggregate_responses=False,
    )
    jatos_area = get_condition_df(jatos_area_dfs, "Area")
    jatos_none = get_condition_df(jatos_none_dfs, "None")
    jatos_area_all = get_condition_df(jatos_area_all_dfs, "Area")
    jatos_none_all = get_condition_df(jatos_none_all_dfs, "None")
    area_path = find_jatos_file("area")
    none_path = find_jatos_file("none")
    area_pass_workers, area_fail_workers = qualification_worker_sets(area_path)
    none_pass_workers, none_fail_workers = qualification_worker_sets(none_path)
    if jatos_area_all is not None:
        area_worker_ids = jatos_area_all["WorkerId"].astype(str).str.strip()
        jatos_area = jatos_area_all[area_worker_ids.isin(area_pass_workers)]
        jatos_area_qual_failed = jatos_area_all[area_worker_ids.isin(area_fail_workers)]
        report_qualification_split("Area", jatos_area, jatos_area_qual_failed)
    else:
        jatos_area = None
        jatos_area_qual_failed = None

    if jatos_none_all is not None:
        none_worker_ids = jatos_none_all["WorkerId"].astype(str).str.strip()
        jatos_none = jatos_none_all[none_worker_ids.isin(none_pass_workers)]
        jatos_none_qual_failed = jatos_none_all[none_worker_ids.isin(none_fail_workers)]
        report_qualification_split("None", jatos_none, jatos_none_qual_failed)
    else:
        jatos_none = None
        jatos_none_qual_failed = None
    gpt5_area = get_condition_df(gpt5_area_dfs, "Area")
    gpt5_none = get_condition_df(gpt5_none_dfs, "None")
    gpt5_nr_area = get_condition_df(gpt5_nr_area_dfs, "Area")
    gpt5_nr_none = get_condition_df(gpt5_nr_none_dfs, "None")
    kimi_area = get_condition_df(kimi_area_dfs, "Area")
    kimi_none = get_condition_df(kimi_none_dfs, "None")
    plot_inequality_sum_disagreements(
        jatos_area,
        condition="Area",
        participant_label="Human",
        output_name="inequality_sum_human_area_disagree",
        brms_data=brms_human_disagree,
    )
    plot_inequality_sum_disagreements(
        jatos_none,
        condition="None",
        participant_label="Human",
        output_name="inequality_sum_human_none_disagree",
        brms_data=brms_human_disagree,
    )
    plot_inequality_sum_disagreements(
        gpt5_area,
        condition="Area",
        participant_label="GPT-5.1",
        output_name="inequality_sum_gpt5_1_area_disagree",
        brms_data=brms_gpt5_disagree,
    )
    plot_inequality_sum_disagreements(
        gpt5_none,
        condition="None",
        participant_label="GPT-5.1",
        output_name="inequality_sum_gpt5_1_none_disagree",
        brms_data=brms_gpt5_disagree,
    )
    plot_inequality_sum_disagreements(
        gpt5_nr_area,
        condition="Area",
        participant_label="GPT-5.1 (No Reasoning)",
        output_name="inequality_sum_gpt5_1_no_reasoning_area_disagree",
        brms_data=None,
    )
    plot_inequality_sum_disagreements(
        gpt5_nr_none,
        condition="None",
        participant_label="GPT-5.1 (No Reasoning)",
        output_name="inequality_sum_gpt5_1_no_reasoning_none_disagree",
        brms_data=None,
    )
    plot_inequality_sum_disagreements(
        kimi_area,
        condition="Area",
        participant_label="Kimi K2.5",
        output_name="inequality_sum_kimi_k2_5_area_disagree",
        brms_data=None,
    )
    plot_inequality_sum_disagreements(
        kimi_none,
        condition="None",
        participant_label="Kimi K2.5",
        output_name="inequality_sum_kimi_k2_5_none_disagree",
        brms_data=None,
    )

    plot_overall_theory_bars(
        jatos_area,
        jatos_none,
        participant_label="Human",
        output_name="overall_theory_bars_human",
    )
    plot_overall_theory_bars(
        gpt5_area,
        gpt5_none,
        participant_label="GPT-5.1",
        output_name="overall_theory_bars_gpt5_1",
    )
    plot_overall_theory_bars(
        gpt5_nr_area,
        gpt5_nr_none,
        participant_label="GPT-5.1 (No Reasoning)",
        output_name="overall_theory_bars_gpt5_1_no_reasoning",
    )
    plot_overall_theory_bars(
        kimi_area,
        kimi_none,
        participant_label="Kimi K2.5",
        output_name="overall_theory_bars_kimi_k2_5",
    )

    plot_agreement_by_condition(
        [("Human", jatos_area, jatos_none)],
        [
            ("GPT-5.1", gpt5_area, gpt5_none),
            ("GPT-5.1 (No Reasoning)", gpt5_nr_area, gpt5_nr_none),
            ("Kimi K2.5", kimi_area, kimi_none),
        ],
        output_name="agreement_by_condition_human_llm",
        brms_agree_by_label={
            "Human": brms_human_agree,
            "GPT-5.1": brms_gpt5_agree,
            "GPT-5.1 (No Reasoning)": None,
            "Kimi K2.5": None,
        },
    )

    plot_js_divergence_by_condition(
        base_label="Human (Qual Passed)",
        base_area=jatos_area,
        base_none=jatos_none,
        comparisons=[
            (
                "Human (Qual Failed)",
                jatos_area_qual_failed,
                jatos_none_qual_failed,
                "tab:gray",
            ),
            ("GPT-5.1", gpt5_area, gpt5_none, "tab:blue"),
            ("GPT-5.1 (No Reasoning)", gpt5_nr_area, gpt5_nr_none, "tab:orange"),
        ],
        output_name="jsd_vs_human_qual_passed",
    )

    area_stats = summarize_jatos_quality(area_path)
    none_stats = summarize_jatos_quality(none_path)
    print(
        "Area: "
        f"{area_stats['failed_qual_only']} failed qual only, "
        f"{area_stats['failed_attention_only']} failed attention only, "
        f"{area_stats['failed_both']} failed both, "
        f"{area_stats['failed_attention']} failed attention total, "
        f"{area_stats['included']} included "
        f"(total {area_stats['total']})"
    )
    print(
        "None: "
        f"{none_stats['failed_qual_only']} failed qual only, "
        f"{none_stats['failed_attention_only']} failed attention only, "
        f"{none_stats['failed_both']} failed both, "
        f"{none_stats['failed_attention']} failed attention total, "
        f"{none_stats['included']} included "
        f"(total {none_stats['total']})"
    )
    total_included = area_stats["included"] + none_stats["included"]
    print(f"Total included: {total_included}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze JATOS and LLM results")
    parser.add_argument(
        "--scenario",
        type=str,
        default=SCENARIO,
        help="The scenario filename base to process",
    )
    parser.add_argument(
        "--num-scenarios",
        type=int,
        default=NUM_SCENARIOS,
        help="Number of scenarios in the data",
    )

    parser.add_argument(
        "--include-qual-failed",
        action="store_true",
        help="Include participants who failed the qualification check",
    )
    parser.add_argument(
        "--include-attention-failed",
        action="store_true",
        help="Include participants who failed attention checks",
    )
    parser.add_argument(
        "--attention-threshold",
        type=float,
        default=1.0,
        help="Minimum proportion of attention checks a worker must pass (default: 1.0)",
    )
    parser.add_argument(
        "--qual-failed-only",
        action="store_true",
        help="ONLY include participants who FAILED the qualification check",
    )
    parser.add_argument(
        "--qual-type",
        type=str,
        choices=["any", "area", "volume"],
        default="any",
        help="Specify which qualification test to filter by (default: any). Useful if participants took both.",
    )

    args = parser.parse_args()

    SCENARIO = args.scenario
    NUM_SCENARIOS = args.num_scenarios
    INCLUDE_QUAL_FAILED = args.include_qual_failed
    INCLUDE_ATTENTION_FAILED = args.include_attention_failed
    QUAL_FAILED_ONLY = args.qual_failed_only
    QUAL_TYPE = args.qual_type
    ATTENTION_THRESHOLD = args.attention_threshold

    main()
