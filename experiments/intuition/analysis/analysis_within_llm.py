"""Trial-level analysis for LLM intuition experiment outputs.

This script is separate from the human within-participant analysis. For LLM runs,
it treats unique prompt keys as the unit of analysis and reports trial-level
proportions and binomial tests.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

DISAGREE_CATEGORIES = ["Nash", "IE", "Random"]
DISAGREE_COLORS = {"Nash": "#DD8452", "IE": "#4C72B0", "Random": "#55A868"}
DISAGREE_LABELS = {"Nash": "Nash", "IE": "IE", "Random": "Decoy"}


def _sanitize_label_for_filename(label):
    """Return a filesystem-safe suffix derived from a label."""
    result = "".join(
        char if char.isalnum() or char in ("-", "_") else "_" for char in str(label)
    )
    return result.strip("_")


def _parse_alpha_bin(value):
    """Parse an alpha-bin value that may use underscores instead of decimal points."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text.replace("_", "."))
    except ValueError:
        return None


def _validate_and_prepare(df):
    """Validate required columns and apply fallback columns when possible."""
    required = {"scenario_hash", "question", "response", "scenario_nbs"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
            + ". Provide LLM long-format output."
        )

    prepared = df.copy()
    if "scenario_ie" not in prepared.columns:
        if "scenario_mec" not in prepared.columns:
            raise ValueError("Missing both scenario_ie and scenario_mec columns.")
        prepared["scenario_ie"] = prepared["scenario_mec"]
    else:
        prepared["scenario_ie"] = prepared["scenario_ie"].where(
            prepared["scenario_ie"].notna(),
            prepared.get("scenario_mec"),
        )

    if "scenario_alpha_bin" not in prepared.columns:
        prepared["scenario_alpha_bin"] = np.nan
    return prepared


def _dedupe_by_prompt(df):
    """Deduplicate rows by prompt key and report conflicting responses."""
    key_cols = ["scenario_hash", "question"]
    response_nunique = df.groupby(key_cols, dropna=False)["response"].nunique(
        dropna=False
    )
    conflicting_prompts = int((response_nunique > 1).sum())
    deduped = df.drop_duplicates(subset=key_cols, keep="last").copy()
    return deduped, conflicting_prompts


def _summarize_disagree(disagree_df):
    """Compute disagree-trial category counts, proportions, and exact CIs."""
    total = len(disagree_df)
    counts = {
        "Nash": int((disagree_df["response"] == disagree_df["scenario_nbs"]).sum()),
        "IE": int((disagree_df["response"] == disagree_df["scenario_ie"]).sum()),
    }
    counts["Random"] = total - counts["Nash"] - counts["IE"]

    props = {}
    ci_bounds = {}
    for category in DISAGREE_CATEGORIES:
        if total == 0:
            props[category] = 0.0
            ci_bounds[category] = (0.0, 0.0)
            continue
        count = counts[category]
        proportion = count / total
        ci = binomtest(k=count, n=total).proportion_ci(confidence_level=0.95)
        props[category] = proportion
        ci_bounds[category] = (float(ci.low), float(ci.high))

    return {"n": total, "counts": counts, "props": props, "ci": ci_bounds}


def _print_disagree_stats(summary):
    """Print disagree-trial summary and binomial test statistics."""
    total = summary["n"]
    print(f"Disagree trials (denominator): {total}")
    for category in DISAGREE_CATEGORIES:
        count = summary["counts"][category]
        prop = summary["props"][category]
        low, high = summary["ci"][category]
        print(f"{category:6s}: {count:4d} ({prop:.1%}) 95% CI [{low:.3f}, {high:.3f}]")

    nash_count = summary["counts"]["Nash"]
    ie_count = summary["counts"]["IE"]
    paired_total = nash_count + ie_count
    if paired_total == 0:
        print("Nash vs IE binomial test not run: no Nash/IE selections.")
        return

    result = binomtest(k=nash_count, n=paired_total, p=0.5, alternative="greater")
    ci = result.proportion_ci(confidence_level=0.95)
    print(
        "Nash > IE among Nash-or-IE choices: "
        f"{nash_count}/{paired_total} ({nash_count / paired_total:.1%}), "
        f"p={result.pvalue:.6f}, 95% CI [{ci.low:.3f}, {ci.high:.3f}]"
    )


def _print_agree_stats(agree_df):
    """Print agree-trial summary and one-sided binomial test against chance."""
    total = len(agree_df)
    if total == 0:
        print("Agree trials: none")
        return

    shared = int((agree_df["response"] == agree_df["scenario_nbs"]).sum())
    test = binomtest(k=shared, n=total, p=1 / 3, alternative="greater")
    ci = test.proportion_ci(confidence_level=0.95)
    print(
        "Agree trials (shared choice > 1/3): "
        f"{shared}/{total} ({shared / total:.1%}), "
        f"p={test.pvalue:.6f}, 95% CI [{ci.low:.3f}, {ci.high:.3f}]"
    )


def _print_stats_by_alpha(df):
    """Print disagree and agree test summaries for each alpha bin."""
    working = df.copy()
    working["alpha_numeric"] = working["scenario_alpha_bin"].apply(_parse_alpha_bin)
    working = working[working["alpha_numeric"].notna()]
    if working.empty:
        print("No alpha bins found; skipped per-alpha test output.")
        return

    alphas = sorted(working["alpha_numeric"].unique())
    print("\nPer-alpha test output:")
    for alpha in alphas:
        alpha_subset = working[working["alpha_numeric"] == alpha]
        disagree = alpha_subset[
            alpha_subset["scenario_nbs"] != alpha_subset["scenario_ie"]
        ]
        agree = alpha_subset[
            alpha_subset["scenario_nbs"] == alpha_subset["scenario_ie"]
        ]
        print(f"\n--- Alpha {alpha:.2f} ---")
        summary = _summarize_disagree(disagree)
        _print_disagree_stats(summary)
        _print_agree_stats(agree)


def _plot_disagree_overall(summary, output_path):
    """Plot overall disagree-trial category proportions with exact-binomial CIs."""
    x_vals = np.arange(len(DISAGREE_CATEGORIES))
    props = [summary["props"][category] for category in DISAGREE_CATEGORIES]
    lows = [summary["ci"][category][0] for category in DISAGREE_CATEGORIES]
    highs = [summary["ci"][category][1] for category in DISAGREE_CATEGORIES]
    lower_errors = [max(0.0, p - lo) for p, lo in zip(props, lows)]
    upper_errors = [max(0.0, hi - p) for p, hi in zip(props, highs)]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(
        x_vals,
        props,
        yerr=[lower_errors, upper_errors],
        color=[DISAGREE_COLORS[c] for c in DISAGREE_CATEGORIES],
        alpha=0.85,
        capsize=4,
        edgecolor="black",
    )
    for index, bar_patch in enumerate(bars):
        category = DISAGREE_CATEGORIES[index]
        count = summary["counts"][category]
        prop = props[index]
        plt.text(
            bar_patch.get_x() + bar_patch.get_width() / 2.0,
            prop + upper_errors[index] + 0.02,
            f"{count}\n({prop:.1%})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xticks(x_vals, [DISAGREE_LABELS[category] for category in DISAGREE_CATEGORIES])
    plt.ylim(0.0, 1.0)
    plt.ylabel("Proportion of Trials")
    plt.title("LLM Disagree-Trial Choice Proportions")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to {output_path}")


def _plot_disagree_by_alpha(df, output_path):
    """Plot disagree-trial category proportions across alpha bins."""
    working = df.copy()
    working["alpha_numeric"] = working["scenario_alpha_bin"].apply(_parse_alpha_bin)
    working = working[working["alpha_numeric"].notna()]
    if working.empty:
        print("No alpha bins found; skipped alpha plot.")
        return

    working["is_disagree"] = working["scenario_nbs"] != working["scenario_ie"]
    disagree = working[working["is_disagree"]].copy()
    if disagree.empty:
        print("No disagree trials with alpha bins; skipped alpha plot.")
        return

    disagree["choice_category"] = "Random"
    disagree.loc[
        disagree["response"] == disagree["scenario_nbs"], "choice_category"
    ] = "Nash"
    disagree.loc[disagree["response"] == disagree["scenario_ie"], "choice_category"] = (
        "IE"
    )

    alphas = sorted(disagree["alpha_numeric"].unique())
    plt.figure(figsize=(8, 5))
    for category in DISAGREE_CATEGORIES:
        y_vals = []
        lower_errors = []
        upper_errors = []
        for alpha in alphas:
            subset = disagree[disagree["alpha_numeric"] == alpha]
            total = len(subset)
            count = int((subset["choice_category"] == category).sum())
            if total == 0:
                y_vals.append(0.0)
                lower_errors.append(0.0)
                upper_errors.append(0.0)
                continue
            proportion = count / total
            ci = binomtest(k=count, n=total).proportion_ci(confidence_level=0.95)
            y_vals.append(proportion)
            lower_errors.append(max(0.0, proportion - ci.low))
            upper_errors.append(max(0.0, ci.high - proportion))

        plt.errorbar(
            alphas,
            y_vals,
            yerr=[lower_errors, upper_errors],
            marker="o",
            linewidth=2,
            capsize=3,
            color=DISAGREE_COLORS[category],
            label=DISAGREE_LABELS[category],
        )

    plt.xticks(alphas, [f"{alpha:.2f}" for alpha in alphas])
    plt.ylim(0.0, 1.0)
    plt.xlabel("Alpha")
    plt.ylabel("Proportion of Disagree Trials")
    plt.title("LLM Disagree-Trial Proportions by Alpha")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to {output_path}")


def analyze_llm_trials(file_path, output_prefix, dedupe_prompts):
    """Run trial-level LLM analysis and save summary plots."""
    raw_df = pd.read_csv(file_path)
    prepared = _validate_and_prepare(raw_df)

    print(f"Loaded rows: {len(prepared)}")
    analysis_df = prepared
    if dedupe_prompts:
        analysis_df, conflicting_prompts = _dedupe_by_prompt(prepared)
        print(f"Unique prompts (scenario_hash, question): {len(analysis_df)}")
        print(
            "Prompts with conflicting responses before dedupe: "
            f"{conflicting_prompts}"
        )
    else:
        print("Dedupe disabled: using all rows as trials.")

    disagree = analysis_df[analysis_df["scenario_nbs"] != analysis_df["scenario_ie"]]
    agree = analysis_df[analysis_df["scenario_nbs"] == analysis_df["scenario_ie"]]

    summary = _summarize_disagree(disagree)
    _print_disagree_stats(summary)
    _print_agree_stats(agree)
    _print_stats_by_alpha(analysis_df)

    safe_prefix = _sanitize_label_for_filename(output_prefix)
    overall_plot = f"figures/{safe_prefix}_disagree_proportions.pdf"
    alpha_plot = f"figures/{safe_prefix}_disagree_by_alpha.pdf"
    _plot_disagree_overall(summary, overall_plot)
    _plot_disagree_by_alpha(analysis_df, alpha_plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Trial-level LLM analysis "
            "(separate from human within-participant analysis)."
        )
    )
    parser.add_argument("--file", required=True, help="Path to an LLM output CSV.")
    parser.add_argument(
        "--output-prefix",
        default="llm_trial",
        help="Prefix for output figure filenames (written to figures/).",
    )
    parser.add_argument(
        "--no-dedupe-prompts",
        action="store_true",
        help="Use all rows as trials (default dedupes by scenario_hash/question).",
    )
    cli_args = parser.parse_args()
    analyze_llm_trials(
        file_path=cli_args.file,
        output_prefix=cli_args.output_prefix,
        dedupe_prompts=not cli_args.no_dedupe_prompts,
    )
