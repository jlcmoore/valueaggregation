"""Within-participant analysis and classification for intuition experiments."""

import argparse
from math import comb

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis.analysis import get_condition_df
from common.shared_analysis import jatos_dfs
from scipy.stats import binomtest, wilcoxon


def binomial_tail_probability(num_trials, success_probability, threshold):
    """Return P(X >= threshold) for X ~ Binomial(num_trials, success_probability)."""
    if threshold <= 0:
        return 1.0
    if threshold > num_trials:
        return 0.0
    return sum(
        comb(num_trials, i)
        * (success_probability**i)
        * ((1 - success_probability) ** (num_trials - i))
        for i in range(threshold, num_trials + 1)
    )


def determine_classification_threshold(num_trials, success_probability, alpha):
    """Return the smallest threshold k with P(X >= k) < alpha and its tail probability."""
    for threshold in range(0, num_trials + 1):
        tail_probability = binomial_tail_probability(
            num_trials, success_probability, threshold
        )
        if tail_probability < alpha:
            return threshold, tail_probability
    return num_trials + 1, 0.0


def _sanitize_label_for_filename(label):
    """Return a filesystem-safe suffix derived from an analysis label."""
    result = "".join(
        char if char.isalnum() or char in ("-", "_") else "_" for char in str(label)
    )
    result = result.strip("_")
    return result


def _analyze_subset(
    df_passed,
    classification_threshold,
    auto_classification_threshold,
    false_positive_alpha,
    label_suffix="",
    save_mean_agreement_plots=True,
):
    """Run agree/disagree within-participant analysis on a pre-filtered subset.

    Parameters:
        df_passed: Long-form response dataframe after qualification/attention filters.
        classification_threshold: Fixed threshold k for disagreement classification.
        auto_classification_threshold: Whether to derive disagreement k from p_guess.
        false_positive_alpha: Alpha used for threshold derivation.
        label_suffix: Optional suffix added to printed headers and output files.
        save_mean_agreement_plots: Whether to save mean-agreement scatter/hist plots.

    Returns:
        A dictionary with disagree-trial classification counts and proportions.
        Returns None when there are no agree/disagree trials in the subset.
    """
    safe_suffix = _sanitize_label_for_filename(label_suffix)
    file_suffix = f"_{safe_suffix}" if safe_suffix else ""
    printed_label = f" [{label_suffix}]" if label_suffix else ""

    df_working = df_passed.copy()
    if "scenario_ie" not in df_working.columns:
        df_working["scenario_ie"] = df_working["scenario_mec"]
    else:
        df_working["scenario_ie"] = df_working["scenario_ie"].where(
            df_working["scenario_ie"].notna(),
            df_working["scenario_mec"],
        )

    df_working["trial_agreement"] = (
        df_working["scenario_nbs"] == df_working["scenario_ie"]
    )
    df_disagree = df_working[~df_working["trial_agreement"]].copy()
    df_agree = df_working[df_working["trial_agreement"]].copy()
    if df_disagree.empty:
        print(f"No disagreement trials found in the filtered dataset{printed_label}.")
        return None
    if df_agree.empty:
        print(f"No agreement trials found in the filtered dataset{printed_label}.")
        return None

    # Disagree trials: classify Nash / IE / Random.
    df_disagree["agree_ie"] = df_disagree["response"] == df_disagree["scenario_ie"]
    df_disagree["agree_nash"] = df_disagree["response"] == df_disagree["scenario_nbs"]
    df_disagree["agree_random"] = ~(df_disagree["agree_ie"] | df_disagree["agree_nash"])

    disagree_trials_per_worker = df_disagree.groupby("WorkerId").size()
    num_disagree_trials = int(round(disagree_trials_per_worker.median()))
    agree_trials_per_worker = df_agree.groupby("WorkerId").size()
    num_agree_trials = int(round(agree_trials_per_worker.median()))

    random_rate = df_disagree["agree_random"].mean()
    p_guess = (1 - random_rate) / 2
    derived_threshold, derived_fpr = determine_classification_threshold(
        num_disagree_trials,
        p_guess,
        false_positive_alpha,
    )
    effective_threshold = classification_threshold
    classification_fpr = derived_fpr
    if auto_classification_threshold:
        effective_threshold = derived_threshold

    worker_stats = df_disagree.groupby("WorkerId")[
        ["agree_ie", "agree_nash", "agree_random"]
    ].mean()
    print(f"Plotting data for {len(worker_stats)} participants{printed_label}.\n")
    print(f"--- Disagree Trials: Guessing Baseline{printed_label} ---")
    print(
        f"Disagree/Agree trials per participant (median): "
        f"{num_disagree_trials}/{num_agree_trials}"
    )
    if disagree_trials_per_worker.nunique() > 1:
        print(
            "Warning: participants have varying disagreement trial counts. "
            "Classifications use a fixed count threshold from the median trial count."
        )
    print(f"Random rate R          : {random_rate:.1%}")
    print(f"p_guess = (1 - R) / 2  : {p_guess:.3f}")
    print(
        f"Derived k (P(X>=k)<{false_positive_alpha:.2f}) for n={num_disagree_trials}: "
        f"{derived_threshold} (FPR={derived_fpr:.3f})"
    )
    if auto_classification_threshold:
        print(f"Using derived threshold k={effective_threshold}")
    else:
        manual_fpr = binomial_tail_probability(
            num_disagree_trials, p_guess, effective_threshold
        )
        classification_fpr = manual_fpr
        print(
            f"Using user threshold k={effective_threshold} "
            f"(implied FPR={manual_fpr:.3f})"
        )
    print("-------------------------\n")

    worker_counts = (
        df_disagree.groupby("WorkerId")[["agree_nash", "agree_ie", "agree_random"]]
        .sum()
        .astype(int)
    )

    def classify_participant(row):
        if row["agree_nash"] >= effective_threshold:
            return "Nash"
        if row["agree_ie"] >= effective_threshold:
            return "IE"
        if row["agree_random"] >= effective_threshold:
            return "Random"
        return "Indeterminate"

    worker_counts["classification"] = worker_counts.apply(classify_participant, axis=1)
    worker_stats["classification"] = worker_counts["classification"]
    counts = worker_stats["classification"].value_counts()
    props = worker_stats["classification"].value_counts(normalize=True)
    disagree_categories = ["Nash", "IE", "Random", "Indeterminate"]

    print(
        f"--- Disagree Classifications (>= {effective_threshold}/"
        f"{num_disagree_trials} choices){printed_label} ---"
    )
    for category in disagree_categories:
        count = counts.get(category, 0)
        prop = props.get(category, 0.0)
        print(f"{category:13s}: {count:2d} ({prop:.1%})")
    print("--------------------------------------------------\n")

    # Agree trials: principled-vs-decoy classification.
    df_agree["choose_shared"] = df_agree["response"] == df_agree["scenario_nbs"]
    df_agree["choose_decoy"] = ~df_agree["choose_shared"]
    agree_threshold, agree_fpr = determine_classification_threshold(
        num_agree_trials,
        1 / 3,
        false_positive_alpha,
    )
    worker_agree = df_agree.groupby("WorkerId")[
        ["choose_shared", "choose_decoy"]
    ].mean()

    def classify_agree(row):
        shared_count = round(row["choose_shared"] * num_agree_trials)
        decoy_count = round(row["choose_decoy"] * num_agree_trials)
        if shared_count >= agree_threshold:
            return "Principled"
        if decoy_count >= agree_threshold:
            return "Decoy"
        return "Indeterminate"

    worker_agree["classification"] = worker_agree.apply(classify_agree, axis=1)
    agree_counts = worker_agree["classification"].value_counts()
    agree_props = worker_agree["classification"].value_counts(normalize=True)

    print(f"--- Agree Trials: Principled Classification{printed_label} ---")
    print(
        f"Baseline p_guess (shared option by chance): 1/3; "
        f"k={agree_threshold}/{num_agree_trials} (FPR={agree_fpr:.3f})"
    )
    if agree_trials_per_worker.nunique() > 1:
        print(
            "Warning: participants have varying agreement trial counts. "
            "Classifications use median count rounding."
        )
    for category in ["Principled", "Decoy", "Indeterminate"]:
        count = agree_counts.get(category, 0)
        prop = agree_props.get(category, 0.0)
        print(f"{category:13s}: {count:2d} ({prop:.1%})")
    print("------------------------------------------------\n")

    shared_successes = int(df_agree["choose_shared"].sum())
    total_agree_trials = int(len(df_agree))
    agree_binom = binomtest(
        k=shared_successes,
        n=total_agree_trials,
        p=1 / 3,
        alternative="greater",
    )
    agree_ci = agree_binom.proportion_ci(confidence_level=0.95)
    print(f"--- Agree Trials: One-Sided Binomial Test{printed_label} ---")
    print(
        f"H3: P(Nash/IE) > 1/3 | shared choices = {shared_successes}/"
        f"{total_agree_trials} ({shared_successes / total_agree_trials:.1%})"
    )
    print(f"Exact binomial p-value (greater, p0=1/3): {agree_binom.pvalue:.6f}")
    print(f"95% exact CI for P(Nash/IE): " f"[{agree_ci.low:.3f}, {agree_ci.high:.3f}]")
    print("-------------------------------------------------\n")

    worker_counts["d_i"] = worker_counts["agree_nash"] - worker_counts["agree_ie"]
    non_zero_differences = worker_counts.loc[worker_counts["d_i"] != 0, "d_i"]
    difference_sign_counts = {
        "nash_lean": int((worker_counts["d_i"] > 0).sum()),
        "tie": int((worker_counts["d_i"] == 0).sum()),
        "ie_lean": int((worker_counts["d_i"] < 0).sum()),
    }
    indeterminate_rows = worker_counts[
        worker_counts["classification"] == "Indeterminate"
    ]
    indeterminate_sign_counts = {
        "nash_lean": int((indeterminate_rows["d_i"] > 0).sum()),
        "tie": int((indeterminate_rows["d_i"] == 0).sum()),
        "ie_lean": int((indeterminate_rows["d_i"] < 0).sum()),
    }
    difference_by_class = {}
    for category in disagree_categories:
        category_counts = (
            worker_counts.loc[worker_counts["classification"] == category, "d_i"]
            .value_counts()
            .sort_index()
        )
        difference_by_class[category] = {
            int(difference_value): int(count)
            for difference_value, count in category_counts.items()
        }
    wilcoxon_pvalue = None
    wilcoxon_statistic = None
    print(f"--- Within-Participant Wilcoxon Signed-Rank Test{printed_label} ---")
    print("Difference definition: d_i = N_i - I_i")
    print(
        f"Participants (all / nonzero d_i): "
        f"{len(worker_counts)} / {len(non_zero_differences)}"
    )
    if len(non_zero_differences) == 0:
        print("Wilcoxon test not run: all participants have d_i = 0.")
    else:
        wilcoxon_result = wilcoxon(
            non_zero_differences,
            alternative="greater",
            zero_method="wilcox",
        )
        print(
            f"One-sided Wilcoxon p-value (median d_i > 0): "
            f"{wilcoxon_result.pvalue:.6f}"
        )
        print(f"Wilcoxon statistic W: {wilcoxon_result.statistic:.3f}")
        wilcoxon_pvalue = float(wilcoxon_result.pvalue)
        wilcoxon_statistic = float(wilcoxon_result.statistic)
    print(
        "Participant difference signs (N > IE / N = IE / N < IE): "
        f"{difference_sign_counts['nash_lean']} / "
        f"{difference_sign_counts['tie']} / "
        f"{difference_sign_counts['ie_lean']}"
    )
    if len(indeterminate_rows) > 0:
        print(
            "Indeterminate only (N > IE / N = IE / N < IE): "
            f"{indeterminate_sign_counts['nash_lean']} / "
            f"{indeterminate_sign_counts['tie']} / "
            f"{indeterminate_sign_counts['ie_lean']}"
        )
    print("-----------------------------------------------------\n")

    if save_mean_agreement_plots:
        # Plot 1: Bar chart + jittered participant values.
        plt.figure(figsize=(6, 5))
        means = [worker_stats["agree_ie"].mean(), worker_stats["agree_nash"].mean()]
        sems = [worker_stats["agree_ie"].sem(), worker_stats["agree_nash"].sem()]
        x_pos = [0, 1]
        plt.bar(
            x_pos,
            means,
            yerr=sems,
            color=["#4C72B0", "#DD8452"],
            alpha=0.6,
            capsize=5,
            edgecolor="black",
        )
        np.random.seed(42)
        jitter_ie = np.random.normal(x_pos[0], 0.08, size=len(worker_stats))
        jitter_nash = np.random.normal(x_pos[1], 0.08, size=len(worker_stats))
        plt.scatter(
            jitter_ie,
            worker_stats["agree_ie"],
            color="#4C72B0",
            alpha=0.7,
            edgecolor="white",
            zorder=3,
        )
        plt.scatter(
            jitter_nash,
            worker_stats["agree_nash"],
            color="#DD8452",
            alpha=0.7,
            edgecolor="white",
            zorder=3,
        )
        for index in range(len(worker_stats)):
            plt.plot(
                [jitter_ie[index], jitter_nash[index]],
                [
                    worker_stats["agree_ie"].iloc[index],
                    worker_stats["agree_nash"].iloc[index],
                ],
                color="gray",
                alpha=0.15,
                zorder=2,
            )

        plt.xticks(x_pos, ["IE", "Nash"])
        plt.ylabel("Mean Agreement per Participant")
        plt.title("Average Agreement with Individual Means")
        plt.ylim(-0.05, 1.05)
        plt.tight_layout()
        plot_one = f"figures/new_data_bar_scatter{file_suffix}.pdf"
        plt.savefig(plot_one)
        plt.close()
        print(f"Saved plot to {plot_one}")

        # Plot 2: Overlaid histograms of participant means.
        plt.figure(figsize=(6, 5))
        bins = np.arange(
            -0.5 / num_disagree_trials,
            (num_disagree_trials + 0.5) / num_disagree_trials,
            1 / num_disagree_trials,
        )
        plt.hist(
            worker_stats["agree_ie"],
            bins=bins,
            alpha=0.5,
            label="IE",
            color="#4C72B0",
            histtype="stepfilled",
            density=True,
            edgecolor="black",
        )
        plt.hist(
            worker_stats["agree_nash"],
            bins=bins,
            alpha=0.5,
            label="Nash",
            color="#DD8452",
            histtype="stepfilled",
            density=True,
            edgecolor="black",
        )
        plt.xlabel("Mean Agreement per Participant")
        plt.ylabel("Density")
        plt.title("Overlaid Distributions of Participant Means")
        plt.xlim(-0.05, 1.05)
        plt.legend()
        plt.tight_layout()
        plot_two = f"figures/new_data_overlaid_dist{file_suffix}.pdf"
        plt.savefig(plot_two)
        plt.close()
        print(f"Saved plot to {plot_two}")

    categories = disagree_categories
    n_total = len(worker_stats)
    if save_mean_agreement_plots:
        # Plot 3: Classification bars.
        plt.figure(figsize=(7, 5))
        if n_total > 0:
            plot_props = [props.get(category, 0.0) for category in categories]
            sems = [
                np.sqrt(p_value * (1 - p_value) / n_total) for p_value in plot_props
            ]
            x_pos = np.arange(len(categories))
            colors = ["#DD8452", "#4C72B0", "#55A868", "lightgray"]
            bars = plt.bar(
                x_pos,
                plot_props,
                yerr=sems,
                color=colors,
                alpha=0.8,
                capsize=5,
                edgecolor="black",
            )
            for index, bar in enumerate(bars):
                count = counts.get(categories[index], 0)
                prop = plot_props[index]
                y_value = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    y_value + sems[index] + 0.02,
                    f"{count}\n({prop:.1%})",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            plt.xticks(x_pos, categories)
            plt.ylabel("Proportion of Participants")
            max_y = (
                max([p_value + sem for p_value, sem in zip(plot_props, sems)])
                if sems
                else 0
            )
            plt.ylim(0, max(1.0, max_y + 0.15))
            plt.tight_layout()
            plot_three = f"figures/new_data_classification_bar{file_suffix}.pdf"
            plt.savefig(plot_three)
            plt.close()
            print(f"Saved plot to {plot_three}")

    disagree_prereg_binom = binomtest(
        k=int(counts.get("Nash", 0)),
        n=n_total,
        p=classification_fpr,
        alternative="greater",
    )
    agree_prereg_binom = binomtest(
        k=int(agree_counts.get("Principled", 0)),
        n=len(worker_agree),
        p=agree_fpr,
        alternative="greater",
    )

    return {
        "n_total": n_total,
        "worker_counts": worker_counts,
        "num_disagree_trials": num_disagree_trials,
        "num_agree_trials": num_agree_trials,
        "effective_threshold": effective_threshold,
        "random_rate": float(random_rate),
        "p_guess": float(p_guess),
        "disagree_threshold": int(effective_threshold),
        "disagree_fpr": float(classification_fpr),
        "derived_disagree_threshold": int(derived_threshold),
        "derived_disagree_fpr": float(derived_fpr),
        "counts": {category: int(counts.get(category, 0)) for category in categories},
        "props": {category: float(props.get(category, 0.0)) for category in categories},
        "agree_n_total": len(worker_agree),
        "agree_threshold": int(agree_threshold),
        "agree_fpr": float(agree_fpr),
        "agree_counts": {
            category: int(agree_counts.get(category, 0))
            for category in ["Principled", "Decoy", "Indeterminate"]
        },
        "agree_props": {
            category: float(agree_props.get(category, 0.0))
            for category in ["Principled", "Decoy", "Indeterminate"]
        },
        "shared_successes": int(shared_successes),
        "total_agree_trials": int(total_agree_trials),
        "agree_binom_pvalue": float(agree_binom.pvalue),
        "agree_ci_low": float(agree_ci.low),
        "agree_ci_high": float(agree_ci.high),
        "wilcoxon_pvalue": wilcoxon_pvalue,
        "wilcoxon_statistic": wilcoxon_statistic,
        "non_zero_differences": int(len(non_zero_differences)),
        "difference_sign_counts": difference_sign_counts,
        "indeterminate_sign_counts": indeterminate_sign_counts,
        "difference_by_class": difference_by_class,
        "disagree_prereg_pvalue": float(disagree_prereg_binom.pvalue),
        "agree_prereg_pvalue": float(agree_prereg_binom.pvalue),
    }


def _plot_alpha_classification_proportions(alpha_summaries):
    """Plot count-based participant classification proportions across alpha bins."""
    if not alpha_summaries:
        return

    categories = ["Nash", "IE", "Random", "Indeterminate"]
    colors = {
        "Nash": "#DD8452",
        "IE": "#4C72B0",
        "Random": "#55A868",
        "Indeterminate": "gray",
    }

    sorted_summaries = sorted(alpha_summaries, key=lambda row: row["alpha"])
    x_values = [row["alpha"] for row in sorted_summaries]

    plt.figure(figsize=(8, 5))
    for category in categories:
        y_values = [row["props"].get(category, 0.0) for row in sorted_summaries]
        lower_errors = []
        upper_errors = []
        for row in sorted_summaries:
            n_total = row["n_total"]
            category_count = row["counts"].get(category, 0)
            if n_total <= 0:
                lower_errors.append(0.0)
                upper_errors.append(0.0)
                continue
            ci = binomtest(k=category_count, n=n_total).proportion_ci(
                confidence_level=0.95
            )
            proportion = category_count / n_total
            lower_errors.append(max(0.0, proportion - ci.low))
            upper_errors.append(max(0.0, ci.high - proportion))
        plt.errorbar(
            x_values,
            y_values,
            yerr=[lower_errors, upper_errors],
            marker="o",
            linewidth=2,
            color=colors[category],
            label=category,
            capsize=3,
        )

    plt.xticks(x_values, [f"{alpha:.2f}" for alpha in x_values])
    plt.xlabel("Alpha")
    plt.ylabel("Proportion of Participants")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plot_path = "figures/new_data_classification_by_alpha.pdf"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to {plot_path}")


def _plot_alpha_relaxed_classification_proportions(
    alpha_summaries,
    plot_path="figures/new_data_classification_by_alpha_relaxed.pdf",
):
    """Plot all-participant maximum-response classifications across alpha bins.

    Each participant is assigned to the response type selected most often over
    their disagreement trials. Exact ties use the fixed category order below,
    matching pandas ``idxmax`` behavior.

    Parameters:
        alpha_summaries: Per-alpha outputs from ``_analyze_subset``.
        plot_path: Destination for the generated PDF.
    """
    if not alpha_summaries:
        return

    categories = ["Nash", "IE", "Random"]
    labels = {"Nash": "Nash", "IE": "IE", "Random": "Decoy"}
    colors = {
        "Nash": "#DD8452",
        "IE": "#4C72B0",
        "Random": "#55A868",
    }
    sorted_summaries = sorted(alpha_summaries, key=lambda row: row["alpha"])
    x_values = [row["alpha"] for row in sorted_summaries]

    plt.figure(figsize=(8, 5))
    for category in categories:
        proportions = []
        lower_errors = []
        upper_errors = []
        for row in sorted_summaries:
            worker_counts = row["worker_counts"]
            response_counts = worker_counts[
                ["agree_nash", "agree_ie", "agree_random"]
            ].rename(
                columns={
                    "agree_nash": "Nash",
                    "agree_ie": "IE",
                    "agree_random": "Random",
                }
            )
            classifications = response_counts[categories].idxmax(axis=1)
            count = int((classifications == category).sum())
            total = len(classifications)
            proportion = count / total
            interval = binomtest(k=count, n=total).proportion_ci(confidence_level=0.95)
            proportions.append(proportion)
            lower_errors.append(proportion - interval.low)
            upper_errors.append(interval.high - proportion)
        plt.errorbar(
            x_values,
            proportions,
            yerr=[lower_errors, upper_errors],
            marker="o",
            linewidth=2,
            color=colors[category],
            label=labels[category],
            capsize=3,
        )

    plt.xticks(x_values, [f"{alpha:.2f}" for alpha in x_values])
    plt.xlabel("Alpha")
    plt.ylabel("Proportion of Participants")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to {plot_path}")


def _plot_alpha_wilcoxon_breakdown(
    alpha_summaries,
    plot_path="figures/new_data_wilcoxon_by_alpha.pdf",
):
    """Plot participant-level d_i = Nash - IE distributions across alpha bins."""
    if not alpha_summaries:
        return

    sorted_summaries = sorted(alpha_summaries, key=lambda row: row["alpha"])
    max_num_trials = max(summary["num_disagree_trials"] for summary in sorted_summaries)
    difference_values = list(range(-max_num_trials, max_num_trials + 1))
    categories = ["IE", "Indeterminate", "Random", "Nash"]
    colors = {
        "IE": "#4C72B0",
        "Indeterminate": "#B0B0B0",
        "Random": "#55A868",
        "Nash": "#DD8452",
    }
    labels = {
        "IE": "IE",
        "Indeterminate": "Indeterminate",
        "Random": "Decoy",
        "Nash": "Nash",
    }

    num_panels = len(sorted_summaries)
    figure, axes = plt.subplots(
        1,
        num_panels,
        figsize=(max(10, 3.0 * num_panels), 3.8),
        sharey=True,
    )
    if num_panels == 1:
        axes = [axes]

    for axis, summary in zip(axes, sorted_summaries):
        n_total = summary["n_total"]
        bottoms = np.zeros(len(difference_values))
        for category in categories:
            category_distribution = summary["difference_by_class"].get(category, {})
            heights = np.array(
                [
                    category_distribution.get(difference_value, 0) / n_total
                    for difference_value in difference_values
                ]
            )
            axis.bar(
                difference_values,
                heights,
                bottom=bottoms,
                width=0.82,
                color=colors[category],
                edgecolor="black",
                linewidth=0.3,
                label=category,
            )
            bottoms += heights

        sign_counts = summary["difference_sign_counts"]
        wilcoxon_text = (
            f"{summary['wilcoxon_pvalue']:.2g}"
            if summary["wilcoxon_pvalue"] is not None
            else "NA"
        )
        binomial_text = (
            f"{summary['disagree_prereg_pvalue']:.2g}"
            if summary["disagree_prereg_pvalue"] is not None
            else "NA"
        )
        axis.axvline(0, color="black", linewidth=1, alpha=0.8)
        axis.set_title(f"alpha={summary['alpha']:.2f}", fontsize=9)
        axis.text(
            0.03,
            0.96,
            f"Wilcoxon p={wilcoxon_text}\nBinom p={binomial_text}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
        )
        axis.text(
            0.97,
            0.96,
            (
                f"N>IE: {sign_counts['nash_lean']}\n"
                f"N=IE: {sign_counts['tie']}\n"
                f"N<IE: {sign_counts['ie_lean']}"
            ),
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5},
        )
        axis.set_xticks(
            sorted(
                {
                    -summary["num_disagree_trials"],
                    -5,
                    0,
                    5,
                    summary["num_disagree_trials"],
                }
            )
        )
        axis.set_xlabel("$d_i = N_i - IE_i$")
        axis.set_xlim(-max_num_trials - 0.8, max_num_trials + 0.8)

    axes[0].set_ylabel("Proportion of Participants")
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[category], ec="black", lw=0.3)
        for category in categories
    ]
    figure.legend(
        legend_handles,
        [labels[category] for category in categories],
        loc="upper center",
        ncol=len(categories),
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(plot_path, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved plot to {plot_path}")


def _plot_alpha_agree_classification_proportions(alpha_summaries):
    """Plot agree-trial classifications across alpha with exact binomial error bars."""
    if not alpha_summaries:
        return

    categories = ["Principled", "Decoy", "Indeterminate"]
    labels = {
        "Principled": "Principled",
        "Decoy": "Random/Decoy",
        "Indeterminate": "Indeterminate",
    }
    colors = {
        "Principled": "#4C72B0",
        "Decoy": "#55A868",
        "Indeterminate": "gray",
    }

    sorted_summaries = sorted(alpha_summaries, key=lambda row: row["alpha"])
    x_values = [row["alpha"] for row in sorted_summaries]

    plt.figure(figsize=(8, 5))
    for category in categories:
        y_values = [row["agree_props"].get(category, 0.0) for row in sorted_summaries]
        lower_errors = []
        upper_errors = []
        for row in sorted_summaries:
            n_total = row["agree_n_total"]
            category_count = row["agree_counts"].get(category, 0)
            if n_total <= 0:
                lower_errors.append(0.0)
                upper_errors.append(0.0)
                continue
            ci = binomtest(k=category_count, n=n_total).proportion_ci(
                confidence_level=0.95
            )
            proportion = category_count / n_total
            lower_errors.append(max(0.0, proportion - ci.low))
            upper_errors.append(max(0.0, ci.high - proportion))
        plt.errorbar(
            x_values,
            y_values,
            yerr=[lower_errors, upper_errors],
            marker="o",
            linewidth=2,
            color=colors[category],
            label=labels[category],
            capsize=3,
        )

    plt.xticks(x_values, [f"{alpha:.2f}" for alpha in x_values])
    plt.xlabel("Alpha")
    plt.ylabel("Proportion of Participants")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plot_path = "figures/new_data_agree_classification_by_alpha.pdf"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved plot to {plot_path}")


def _print_attention_threshold_summary(attention_threshold, alpha_summaries):
    """Print a compact by-alpha robustness summary for one attention threshold.

    Parameters:
        attention_threshold: Minimum worker-level attention pass rate used.
        alpha_summaries: Summary dictionaries returned by ``_analyze_subset``.

    Returns:
        None.
    """

    total_workers = sum(summary["n_total"] for summary in alpha_summaries)
    print(
        f"\n=== Attention-threshold summary: {attention_threshold:.0%} "
        f"(total workers={total_workers}) ==="
    )
    print(
        "\t".join(
            [
                "alpha",
                "N",
                "Nash",
                "IE",
                "Principled",
                "R",
                "k_dis",
                "FPR_dis",
                "k_agr",
                "FPR_agr",
                "wilcoxon_p",
                "agree_binom_p",
                "prereg_dis_p",
                "prereg_agr_p",
            ]
        )
    )
    for summary in alpha_summaries:
        print(
            "\t".join(
                [
                    f"{summary['alpha']:.2f}",
                    str(summary["n_total"]),
                    f"{summary['props'].get('Nash', 0.0):.4f}",
                    f"{summary['props'].get('IE', 0.0):.4f}",
                    f"{summary['agree_props'].get('Principled', 0.0):.4f}",
                    f"{summary['random_rate']:.4f}",
                    f"{summary['disagree_threshold']}/{summary['num_disagree_trials']}",
                    f"{summary['disagree_fpr']:.4f}",
                    f"{summary['agree_threshold']}/{summary['num_agree_trials']}",
                    f"{summary['agree_fpr']:.4f}",
                    (
                        f"{summary['wilcoxon_pvalue']:.6g}"
                        if summary["wilcoxon_pvalue"] is not None
                        else "NA"
                    ),
                    f"{summary['agree_binom_pvalue']:.6g}",
                    f"{summary['disagree_prereg_pvalue']:.6g}",
                    f"{summary['agree_prereg_pvalue']:.6g}",
                ]
            )
        )


def analyze_new_data(
    scenario,
    num_scenarios,
    include_qual_failed,
    include_attention_failed,
    qual_failed_only=False,
    qual_type="any",
    attention_threshold=1.0,
    classification_threshold=6,
    file_path=None,
    auto_classification_threshold=False,
    false_positive_alpha=0.05,
    plot_alpha_summary_figures=True,
):
    """Analyze within-participant agreement and run summary inference.

    Parameters:
        scenario: Scenario filename base used by JATOS loading helpers.
        num_scenarios: Number of scenarios per participant.
        include_qual_failed: Whether to include qualification-failed participants.
        include_attention_failed: Whether to include attention-failed participants.
        qual_failed_only: Whether to include only qualification-failed participants.
        qual_type: Qualification subset to evaluate ("any", "area", or "volume").
        attention_threshold: Minimum attention pass rate per participant.
        classification_threshold: Fixed threshold k for participant classification.
        file_path: Optional direct path to MTurk/JATOS-style CSV.
        auto_classification_threshold: Whether to derive k from p_guess and alpha.
        false_positive_alpha: Alpha used when deriving k from p_guess.
        plot_alpha_summary_figures: Whether to save alpha-summary figures.

    Returns:
        Summary data for the analyzed subset(s). Prints summary statistics and
        optionally writes plots to disk.
    """
    import os

    from common.shared_analysis import (
        JATOS_RESULTS_DIR,
        MTURK_RESULTS_DIR,
        jatos_dfs,
        mturk_dfs,
    )

    # Try JATOS dir first unless we have an explicit input file.
    df_passed = None
    if not file_path:
        dfs = jatos_dfs(
            scenario,
            num_scenarios=num_scenarios,
            desired_condition="chart_type=area",
            delete_user_fail_check=not include_attention_failed,
            require_qualification=not include_qual_failed,
            aggregate_responses=False,
            plot=False,
            verbose=False,
        )
        df_passed = get_condition_df(dfs, "Area")

    if file_path:
        df = pd.read_csv(file_path)
        # Custom loading logic
        from common.qualification_utils import (
            load_qualification_answers,
            qualification_columns_in_header,
            qualification_passed_row,
        )
        from common.shared_analysis import mturk_explode_df

        # Apply qualification filter if requested
        if "qualificationPassed" in df.columns:
            qual = df["qualificationPassed"].astype(str).str.lower()

            if qual_type != "any":
                from common.qualification_utils import load_qualification_answers

                answers = load_qualification_answers()

                if qual_type == "area":
                    qual_cols = [c for c in df.columns if "q_question-stacked" in c]
                else:
                    qual_cols = [c for c in df.columns if "q_question-3D" in c]

                def check_manual_qual(row):
                    for c in qual_cols:
                        k = c.replace("Answer.", "")
                        expected = answers.get(f"Answer.{k}")
                        if expected is None:
                            continue
                        if str(row.get(c, "")).strip() != str(expected).strip():
                            return False
                    return True

                manual_pass_series = df.apply(check_manual_qual, axis=1)
                if qual_failed_only:
                    df = df[~manual_pass_series]
                elif not include_qual_failed:
                    df = df[manual_pass_series]
            else:
                if qual_failed_only:
                    df = df[~qual.isin(["true", "1", "yes"])]
                elif not include_qual_failed:
                    df = df[qual.isin(["true", "1", "yes"])]

        df_exploded = mturk_explode_df(df, num_scenarios)
        df_exploded["passed_attention"] = (
            df_exploded["attention-response"] == df_exploded["attention-answer"]
        )

        if not include_attention_failed:
            # Calculate proportion of passed attention checks per worker
            worker_attention_rates = df_exploded.groupby("WorkerId")[
                "passed_attention"
            ].mean()
            failed_workers = worker_attention_rates[
                worker_attention_rates < attention_threshold
            ].index.unique()
            df_passed = df_exploded[
                ~df_exploded["WorkerId"].isin(failed_workers)
            ].copy()
        else:
            df_passed = df_exploded.copy()

    elif df_passed is None or df_passed.empty:
        # Check MTURK dir where the new data currently lives
        # I need to find the specific file because mturk_dfs expects a run_dir
        # No, I'll just manually load it and use mturk_explode_df
        from common.shared_analysis import mturk_explode_df

        found = False
        # Look for files with morally_best or best compromise
        for f in os.listdir(MTURK_RESULTS_DIR):
            if "morally_best-area" in f or "best compromise-area" in f:
                df = pd.read_csv(os.path.join(MTURK_RESULTS_DIR, f))

                # Apply qualification filter if requested
                if "qualificationPassed" in df.columns:
                    qual = df["qualificationPassed"].astype(str).str.lower()

                    if qual_type != "any":
                        from common.qualification_utils import (
                            load_qualification_answers,
                        )

                        answers = load_qualification_answers()

                        if qual_type == "area":
                            qual_cols = [
                                c for c in df.columns if "q_question-stacked" in c
                            ]
                        else:
                            qual_cols = [c for c in df.columns if "q_question-3D" in c]

                        def check_manual_qual(row):
                            for c in qual_cols:
                                k = c.replace("Answer.", "")
                                expected = answers.get(f"Answer.{k}")
                                if expected is None:
                                    continue
                                if str(row.get(c, "")).strip() != str(expected).strip():
                                    return False
                            return True

                        manual_pass_series = df.apply(check_manual_qual, axis=1)
                        if qual_failed_only:
                            df = df[~manual_pass_series]
                        elif not include_qual_failed:
                            df = df[manual_pass_series]
                    else:
                        if qual_failed_only:
                            df = df[~qual.isin(["true", "1", "yes"])]
                        elif not include_qual_failed:
                            df = df[qual.isin(["true", "1", "yes"])]

                df_exploded = mturk_explode_df(df, num_scenarios)
                df_exploded["passed_attention"] = (
                    df_exploded["attention-response"] == df_exploded["attention-answer"]
                )

                if not include_attention_failed:
                    # Calculate proportion of passed attention checks per worker
                    worker_attention_rates = df_exploded.groupby("WorkerId")[
                        "passed_attention"
                    ].mean()
                    failed_workers = worker_attention_rates[
                        worker_attention_rates < attention_threshold
                    ].index.unique()
                    df_passed = df_exploded[
                        ~df_exploded["WorkerId"].isin(failed_workers)
                    ].copy()
                else:
                    df_passed = df_exploded.copy()
                found = True
                break

        if not found or df_passed is None or df_passed.empty:
            print("No valid data found for the given parameters.")
            return

    if "scenario_ie" not in df_passed.columns:
        df_passed["scenario_ie"] = df_passed["scenario_mec"]
    else:
        df_passed["scenario_ie"] = df_passed["scenario_ie"].where(
            df_passed["scenario_ie"].notna(),
            df_passed["scenario_mec"],
        )

    alpha_col = "scenario_alpha_bin"
    has_alpha_bins = (
        alpha_col in df_passed.columns and df_passed[alpha_col].notna().any()
    )
    if has_alpha_bins:
        alpha_values = sorted(
            {
                str(value).strip()
                for value in df_passed[alpha_col].dropna().unique()
                if str(value).strip() != ""
            },
            key=lambda x: float(x.replace("_", ".")),
        )
        print(
            "Detected alpha-specific rows. Running separate within-participant "
            f"analyses for alpha bins: {', '.join(alpha_values)}"
        )
        alpha_summaries = []
        for alpha_value in alpha_values:
            subset = df_passed[
                df_passed[alpha_col].astype(str).str.strip() == alpha_value
            ].copy()
            subset_summary = _analyze_subset(
                df_passed=subset,
                classification_threshold=classification_threshold,
                auto_classification_threshold=auto_classification_threshold,
                false_positive_alpha=false_positive_alpha,
                label_suffix=f"alpha_{alpha_value}",
                save_mean_agreement_plots=False,
            )
            if subset_summary is not None:
                alpha_summaries.append(
                    {
                        "alpha": float(alpha_value.replace("_", ".")),
                        **subset_summary,
                    }
                )
        if plot_alpha_summary_figures:
            _plot_alpha_classification_proportions(alpha_summaries)
            _plot_alpha_relaxed_classification_proportions(alpha_summaries)
            _plot_alpha_wilcoxon_breakdown(alpha_summaries)
            _plot_alpha_agree_classification_proportions(alpha_summaries)
        return alpha_summaries

    return _analyze_subset(
        df_passed=df_passed,
        classification_threshold=classification_threshold,
        auto_classification_threshold=auto_classification_threshold,
        false_positive_alpha=false_positive_alpha,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze within-participant data")
    parser.add_argument(
        "--scenario",
        type=str,
        help=(
            "Scenario filename base used by JATOS loading helpers. Optional when "
            "--file is provided."
        ),
    )
    parser.add_argument(
        "--num-scenarios",
        type=int,
        required=True,
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
        help=(
            "Minimum proportion of attention checks a worker must pass "
            "(default: 1.0). If set to e.g. 0.75, workers who pass 75%% of "
            "checks will be included."
        ),
    )
    parser.add_argument(
        "--attention-thresholds",
        type=float,
        nargs="+",
        help=(
            "Run and print compact by-alpha summaries for one or more attention "
            "thresholds. When provided, the script skips alpha-summary plot "
            "writing so multiple thresholds can be compared in one invocation."
        ),
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
        help="Specify which qualification test to filter by (default: any).",
    )
    parser.add_argument(
        "--classification-threshold",
        type=int,
        default=6,
        help="Number of choices needed to classify as a type (default: 6)",
    )
    parser.add_argument(
        "--auto-classification-threshold",
        action="store_true",
        help="Set k from p_guess so P(X>=k|guesser) < false-positive-alpha.",
    )
    parser.add_argument(
        "--false-positive-alpha",
        type=float,
        default=0.05,
        help="Alpha for deriving k from the guessing baseline (default: 0.05).",
    )
    parser.add_argument("--file", type=str, help="Path to the results CSV file")

    args = parser.parse_args()
    if not args.file and not args.scenario:
        parser.error("--scenario is required unless --file is provided.")
    if args.attention_thresholds:
        for threshold in args.attention_thresholds:
            alpha_summaries = analyze_new_data(
                args.scenario,
                args.num_scenarios,
                args.include_qual_failed,
                args.include_attention_failed,
                args.qual_failed_only,
                args.qual_type,
                threshold,
                args.classification_threshold,
                args.file,
                args.auto_classification_threshold,
                args.false_positive_alpha,
                plot_alpha_summary_figures=False,
            )
            if alpha_summaries:
                _print_attention_threshold_summary(threshold, alpha_summaries)
    else:
        analyze_new_data(
            args.scenario,
            args.num_scenarios,
            args.include_qual_failed,
            args.include_attention_failed,
            args.qual_failed_only,
            args.qual_type,
            args.attention_threshold,
            args.classification_threshold,
            args.file,
            args.auto_classification_threshold,
            args.false_positive_alpha,
        )
