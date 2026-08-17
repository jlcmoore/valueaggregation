import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common.qualification_utils import load_qualification_answers
from common.shared_analysis import (
    JATOS_RESULTS_DIR,
    MTURK_RESULTS_DIR,
    mturk_explode_df,
)


def find_data_file():
    # Search for the JATOS results file in both directories
    for d in [MTURK_RESULTS_DIR, JATOS_RESULTS_DIR]:
        if not os.path.exists(d):
            continue
        for f in os.listdir(d):
            if (
                "jatos_results_data" in f
                and f.endswith(".csv")
                and ("morally_best-area" in f or "best compromise-area" in f)
            ):
                return os.path.join(d, f)
    return None


def check_manual_qual(row, qual_type, answers, df_columns):
    cols = [
        c for c in df_columns if ("stacked" in c if qual_type == "area" else "3D" in c)
    ]
    for c in cols:
        k = c.replace("Answer.", "")
        expected = answers.get(f"Answer.{k}")
        if expected is None:
            continue
        if str(row.get(c, "")).strip() != str(expected).strip():
            return False
    return True


def analyze_comparisons(num_scenarios, classification_threshold, file_path=None):
    if not file_path:
        file_path = find_data_file()
    if not file_path:
        print("Could not find the dataset.")
        return

    print(f"Loading data from {file_path}")
    df_raw = pd.read_csv(file_path)

    # 1. Evaluate Qualifications per Worker
    answers = load_qualification_answers()
    worker_quals = df_raw[["WorkerId"]].copy().drop_duplicates()
    worker_quals["area_pass"] = df_raw.apply(
        lambda r: check_manual_qual(r, "area", answers, df_raw.columns), axis=1
    ).values
    worker_quals["vol_pass"] = df_raw.apply(
        lambda r: check_manual_qual(r, "volume", answers, df_raw.columns), axis=1
    ).values

    # 2. Explode Data and Evaluate Attention & Responses
    df_exp = mturk_explode_df(df_raw, num_scenarios)
    df_exp["passed_attention"] = (
        df_exp["attention-response"] == df_exp["attention-answer"]
    )
    df_exp["agree_util"] = df_exp["response"] == df_exp["scenario_mec"]
    df_exp["agree_nash"] = df_exp["response"] == df_exp["scenario_nbs"]
    df_exp["agree_neither"] = ~(df_exp["agree_util"] | df_exp["agree_nash"])

    # Overall trial-level percentages across ALL trials before subsetting
    total_trials = len(df_exp)
    overall_nash_pct = df_exp["agree_nash"].sum() / total_trials
    overall_util_pct = df_exp["agree_util"].sum() / total_trials
    overall_neither_pct = df_exp["agree_neither"].sum() / total_trials

    print(f"\n{'='*50}")
    print(
        f"OVERALL RAW TRIAL RATES (All {total_trials} trials from all {df_raw['WorkerId'].nunique()} workers)"
    )
    print(f"{'='*50}")
    print(f"Nash  : {overall_nash_pct:.1%}")
    print(f"Util  : {overall_util_pct:.1%}")
    print(f"Decoy : {overall_neither_pct:.1%}\n")

    # Aggregate per worker
    worker_stats = (
        df_exp.groupby("WorkerId")
        .agg(
            attention_rate=("passed_attention", "mean"),
            agree_util=("agree_util", "mean"),
            agree_nash=("agree_nash", "mean"),
            agree_neither=("agree_neither", "mean"),
        )
        .reset_index()
    )

    # Classify
    def classify(row):
        # Multiply by num_scenarios and round to avoid floating point issues
        nash_count = round(row["agree_nash"] * num_scenarios)
        util_count = round(row["agree_util"] * num_scenarios)
        neither_count = round(row["agree_neither"] * num_scenarios)

        if nash_count >= classification_threshold:
            return "Nash"
        elif util_count >= classification_threshold:
            return "Util"
        elif neither_count >= classification_threshold:
            return "Neither"
        else:
            return "Indeterminate"

    worker_stats["classification"] = worker_stats.apply(classify, axis=1)

    # Merge quals
    worker_stats = worker_stats.merge(worker_quals, on="WorkerId")

    # Define the subsets
    subset_defs = {
        "Passed Both\n(Strict Qual)": lambda df: df[(df.area_pass) & (df.vol_pass)],
        "Passed Area\nOnly": lambda df: df[df.area_pass],
        "Passed Volume\nOnly": lambda df: df[df.vol_pass],
        "Passed\nEither": lambda df: df[(df.area_pass) | (df.vol_pass)],
        "Failed\nBoth": lambda df: df[~(df.area_pass) & ~(df.vol_pass)],
    }

    for threshold in [1.0, 0.75]:
        print(
            f"\n{'='*50}\nANALYZING WITH ATTENTION THRESHOLD: {threshold:.0%}\n{'='*50}"
        )
        df_t = worker_stats[worker_stats.attention_rate >= threshold]

        subset_data = {}
        for name, filter_fn in subset_defs.items():
            sub_df = filter_fn(df_t)
            subset_data[name] = sub_df

            n_ppt = len(sub_df)
            if n_ppt > 0:
                # We need the trial-level rates! In df_t we only have worker_stats (means).
                # The mean of the means is mathematically identical to the overall trial rate
                # because every participant has exactly 9 trials (no missing data per participant here).
                overall_nash = sub_df["agree_nash"].mean()
                overall_util = sub_df["agree_util"].mean()
                overall_neither = sub_df["agree_neither"].mean()

                print(
                    f"{name.replace(chr(10), ' '):25s}: {n_ppt:2d} participants | Trial Base Rates -> Nash: {overall_nash:.1%}, Util: {overall_util:.1%}, Decoy: {overall_neither:.1%}"
                )
            else:
                print(f"{name.replace(chr(10), ' '):25s}:  0 participants")

        subset_names = list(subset_defs.keys())

        # ---------------------------------------------------------------------
        # PLOT 1: BAR + SCATTER
        # ---------------------------------------------------------------------
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        x = np.arange(len(subset_names))
        width = 0.35

        util_means = [
            subset_data[name]["agree_util"].mean() if len(subset_data[name]) > 0 else 0
            for name in subset_names
        ]
        util_sems = [
            subset_data[name]["agree_util"].sem() if len(subset_data[name]) > 0 else 0
            for name in subset_names
        ]
        nash_means = [
            subset_data[name]["agree_nash"].mean() if len(subset_data[name]) > 0 else 0
            for name in subset_names
        ]
        nash_sems = [
            subset_data[name]["agree_nash"].sem() if len(subset_data[name]) > 0 else 0
            for name in subset_names
        ]

        ax1.bar(
            x - width / 2,
            util_means,
            width,
            yerr=util_sems,
            label="Util (MEC)",
            color="#4C72B0",
            alpha=0.6,
            capsize=5,
            edgecolor="black",
        )
        ax1.bar(
            x + width / 2,
            nash_means,
            width,
            yerr=nash_sems,
            label="Nash (NBS)",
            color="#DD8452",
            alpha=0.6,
            capsize=5,
            edgecolor="black",
        )

        for i, name in enumerate(subset_names):
            df_sub = subset_data[name]
            if len(df_sub) == 0:
                continue

            # Jitter
            np.random.seed(42 + i)
            j_util = np.random.normal(x[i] - width / 2, 0.05, size=len(df_sub))
            j_nash = np.random.normal(x[i] + width / 2, 0.05, size=len(df_sub))

            ax1.scatter(
                j_util,
                df_sub["agree_util"],
                color="#4C72B0",
                alpha=0.7,
                edgecolor="white",
                zorder=3,
                s=20,
            )
            ax1.scatter(
                j_nash,
                df_sub["agree_nash"],
                color="#DD8452",
                alpha=0.7,
                edgecolor="white",
                zorder=3,
                s=20,
            )

            for j in range(len(df_sub)):
                ax1.plot(
                    [j_util[j], j_nash[j]],
                    [df_sub["agree_util"].iloc[j], df_sub["agree_nash"].iloc[j]],
                    color="gray",
                    alpha=0.15,
                    zorder=2,
                )

        ax1.set_xticks(x)
        ax1.set_xticklabels(subset_names)
        ax1.set_ylabel("Mean Agreement per Participant")
        ax1.set_title(
            f"Mean Agreement by Qualification Subset (Attn >= {threshold:.0%})"
        )
        ax1.set_ylim(-0.05, 1.05)
        ax1.legend()
        fig1.tight_layout()
        fname1 = f"figures/compare_attn_{int(threshold*100)}_bar_scatter.pdf"
        fig1.savefig(fname1)
        plt.close(fig1)
        print(f"Saved {fname1}")

        # ---------------------------------------------------------------------
        # PLOT 2: DISTRIBUTIONS (SUBPLOTS)
        # ---------------------------------------------------------------------
        fig2, axes2 = plt.subplots(1, 5, figsize=(16, 4), sharey=True, sharex=True)
        bins = np.arange(-0.5 / 9, 10.5 / 9, 1 / 9)

        for i, name in enumerate(subset_names):
            ax = axes2[i]
            df_sub = subset_data[name]
            if len(df_sub) > 0:
                ax.hist(
                    df_sub["agree_util"],
                    bins=bins,
                    alpha=0.5,
                    label="Util",
                    color="#4C72B0",
                    histtype="stepfilled",
                    density=True,
                    edgecolor="black",
                )
                ax.hist(
                    df_sub["agree_nash"],
                    bins=bins,
                    alpha=0.5,
                    label="Nash",
                    color="#DD8452",
                    histtype="stepfilled",
                    density=True,
                    edgecolor="black",
                )
            ax.set_title(f"{name}\n(n={len(df_sub)})")
            ax.set_xlabel("Mean Agreement")
            if i == 0:
                ax.set_ylabel("Density")

        axes2[-1].legend()
        fig2.suptitle(
            f"Distributions of Participant Means by Qualification Subset (Attn >= {threshold:.0%})",
            y=1.05,
        )
        fig2.tight_layout()
        fname2 = f"figures/compare_attn_{int(threshold*100)}_distros.pdf"
        fig2.savefig(fname2)
        plt.close(fig2)
        print(f"Saved {fname2}")

        # ---------------------------------------------------------------------
        # PLOT 3: CATEGORICAL CLASSIFICATION (GROUPED BARS)
        # ---------------------------------------------------------------------
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        w = 0.2
        categories = ["Nash", "Util", "Neither", "Indeterminate"]
        colors = ["#DD8452", "#4C72B0", "#55A868", "lightgray"]

        for cat_idx, cat in enumerate(categories):
            props = []
            sems = []
            for name in subset_names:
                df_sub = subset_data[name]
                n_total = len(df_sub)
                if n_total == 0:
                    props.append(0)
                    sems.append(0)
                else:
                    p = (df_sub["classification"] == cat).mean()
                    props.append(p)
                    sems.append(np.sqrt(p * (1 - p) / n_total))

            offset = (cat_idx - 1.5) * w
            bars = ax3.bar(
                x + offset,
                props,
                w,
                yerr=sems,
                label=cat,
                color=colors[cat_idx],
                alpha=0.8,
                capsize=4,
                edgecolor="black",
            )

            # Add text labels (count and percentage)
            for j, bar in enumerate(bars):
                df_sub = subset_data[subset_names[j]]
                count = (
                    (df_sub["classification"] == cat).sum() if len(df_sub) > 0 else 0
                )
                prop = props[j]
                if prop > 0:
                    yval = bar.get_height()
                    ax3.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        yval + sems[j] + 0.02,
                        f"{count}\n({prop:.0%})",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        ax3.set_xticks(x)
        ax3.set_xticklabels(subset_names)
        ax3.set_ylabel("Proportion of Participants")
        ax3.set_title(
            f"Participant Classifications (>= {classification_threshold}/{num_scenarios} choices) by Qual Subset (Attn >= {threshold:.0%})"
        )
        ax3.set_ylim(0, 1.1)
        ax3.legend()
        fig3.tight_layout()
        fname3 = f"figures/compare_attn_{int(threshold*100)}_cutoff_{classification_threshold}_classifications.pdf"
        fig3.savefig(fname3)
        plt.close(fig3)
        print(f"Saved {fname3}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, required=False)
    parser.add_argument("--num-scenarios", type=int, default=9)
    parser.add_argument(
        "--classification-threshold",
        type=int,
        default=6,
        help="Number of choices needed to classify as a type (default: 6)",
    )
    parser.add_argument("--file", type=str, help="Path to the results CSV file")
    args = parser.parse_args()
    analyze_comparisons(args.num_scenarios, args.classification_threshold, args.file)
