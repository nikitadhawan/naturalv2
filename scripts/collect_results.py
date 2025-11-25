import argparse
import ast
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def _extract_data_count(value):
    if isinstance(value, str):
        return ast.literal_eval(value)["data_count"]
    return None


def collect_results_from_directories(results_dir, experiment_name):
    all_results = []

    for subdir in os.listdir(results_dir):
        subdir_path = os.path.join(results_dir, subdir)
        if os.path.isdir(subdir_path) and subdir.endswith(experiment_name):
            apo_csv_path = os.path.join(subdir_path, "apo_results.csv")
            if os.path.exists(apo_csv_path):
                df = pd.read_csv(apo_csv_path, index_col=0)
                subset_cols = [
                    c for c in ["treatment", "outcome", "estimator"] if c in df.columns
                ]
                if len(subset_cols) > 0:
                    df = df.drop_duplicates(subset=subset_cols)
                df["source"] = subdir
                all_results.append(df)

    return all_results


def process_and_combine_results(all_results, experiment_name):
    final_df = pd.concat(all_results, ignore_index=True)

    # Extract nct_id from source column
    final_df["nct_id"] = final_df["source"].apply(
        lambda x: x.split(experiment_name)[0] if isinstance(x, str) else None
    )
    # Extract data_count from dict-like columns
    for col_name, out_col in [
        ("treatment_outcome_filter", "treatment_outcome_filter_data_count"),
        ("knowns", "knowns_data_count"),
        ("imputations", "imputations_data_count"),
    ]:
        if col_name in final_df.columns:
            final_df[out_col] = final_df[col_name].apply(_extract_data_count)

    # Ensure required columns exist, then select and order them
    required_columns = [
        "estimator",
        "outcome",
        "treatment",
        "pred_response",
        "CI_lower",
        "CI_upper",
        "true_response",
        "abs_error",
        "true_dispersion_type",
        "true_dispersion",
        "true_cohort_size",
        "treatment_outcome_filter_data_count",
        "knowns_data_count",
        "imputations_data_count",
        "inclusion_prob",
        "conditional_extraction",
        "source_name",
        "initial_curated",
        "conditions",
        "filter_by_date",
        "source",
        "nct_id",
    ]

    final_df = final_df[required_columns]
    return final_df[final_df["abs_error"].notna()]


def create_forest_plots(final_df, data_path, output_path, experiment_name, estimator):  # noqa: PLR0915
    # Filter for specified estimator rows
    estimator_df = final_df[final_df["estimator"] == estimator].copy()
    estimator_df = estimator_df.sort_values("true_response", ascending=True)

    # Group by nct_id and outcome
    grouped = estimator_df.groupby(["nct_id", "outcome"])
    num_plots = 0
    with PdfPages(output_path) as pdf:
        for (nct_id, outcome), group_df in grouped:
            # Sort by true_response
            group_df = group_df.sort_values("true_response", ascending=True)  # noqa: PLW2901

            if len(group_df) == 0:
                print(f"No treatment responses for {nct_id}: {outcome}.")
                continue

            num_plots += 1
            fig, ax = plt.subplots(figsize=(10, max(6, len(group_df) * 0.4)))

            # Create y-axis positions (one for each treatment, starting at 1)
            y_positions = range(1, len(group_df) + 1)

            # Plot confidence intervals from CI_lower to CI_upper
            for i, (_idx, row) in enumerate(group_df.iterrows()):
                y_pos = i + 1  # Start at position 1
                pred_response = row["pred_response"]
                ci_lower = row["CI_lower"]
                ci_upper = row["CI_upper"]
                lower_err = pred_response - ci_lower
                upper_err = ci_upper - pred_response
                ax.errorbar(
                    pred_response,
                    y_pos,
                    xerr=[[lower_err], [upper_err]],
                    color="purple",
                    linewidth=2,
                    alpha=0.6,
                    capsize=5,
                    capthick=2,
                    fmt="none",
                    zorder=2,
                )

                ax.scatter(
                    pred_response,
                    y_pos,
                    color="purple",
                    marker="o",
                    s=80,
                    zorder=3,
                    label="NATURAL-IPW Response" if i == 0 else "",
                )
                ax.scatter(
                    row["true_response"],
                    y_pos,
                    color="green",
                    marker="s",
                    s=80,
                    zorder=3,
                    label="Clinical Trial Response" if i == 0 else "",
                )

                # Add imputations_data_count below purple scatter point
                imp_count = row.get("imputations_data_count")
                if pd.notna(imp_count):
                    ax.text(
                        pred_response,
                        y_pos - 0.08,
                        f"{int(imp_count)}",
                        ha="center",
                        va="top",
                        fontsize=12,
                        zorder=4,
                    )

                # Add true_cohort_size above green scatter point
                cohort_size = row.get("true_cohort_size")
                if pd.notna(cohort_size):
                    ax.text(
                        row["true_response"],
                        y_pos + 0.17,
                        f"{int(cohort_size)}",
                        ha="center",
                        va="top",
                        fontsize=12,
                        zorder=4,
                    )

                # Check if true_dispersion is a confidence interval tuple
                if pd.notna(row.get("true_dispersion")):
                    try:
                        dispersion = ast.literal_eval(str(row["true_dispersion"]))
                        if (
                            isinstance(dispersion, (tuple, list))
                            and len(dispersion) == 2
                        ):
                            lower = float(dispersion[0]) / 100
                            upper = float(dispersion[1]) / 100
                            true_response = row["true_response"]
                            lower_err_disp = true_response - lower
                            upper_err_disp = upper - true_response
                            ax.errorbar(
                                true_response,
                                y_pos,
                                xerr=[[lower_err_disp], [upper_err_disp]],
                                color="green",
                                linewidth=2,
                                alpha=0.6,
                                capsize=5,
                                capthick=2,
                                fmt="none",
                                linestyle="--",
                                zorder=2,
                            )
                    except (ValueError, SyntaxError):
                        print(
                            f"Error in plotting dispersion {dispersion} for {nct_id}: {outcome}"
                        )
                        pass

            # Use treatments as y-axis labels
            y_labels = group_df["treatment"].tolist()

            ax.set_yticks(y_positions)
            ax.set_yticklabels(y_labels)
            ax.tick_params(labelsize=12)
            ax.set_xlabel("Treatment Response")
            ax.set_xlim(-0.01, 1)
            ax.set_ylim(-0.2, len(group_df) + 0.5)

            # Split title into 2 lines if it's too long
            condition = ast.literal_eval(str(group_df.iloc[0]["conditions"]))[0]
            title_text = f"{nct_id} ({condition}): {outcome}"
            if len(title_text) > 50:
                # Split at the middle space
                mid = len(title_text) // 2
                # Find the nearest space
                for i in range(mid, len(title_text)):
                    if title_text[i] == " ":
                        title_text = f"{title_text[:i]}\n{title_text[i + 1 :]}"
                        break
            ax.set_title(title_text, fontsize=12, fontweight="bold")
            # ax.axvline(x=0, color='gray', linestyle='--', linewidth=3)
            ax.legend()
            ax.grid(axis="x", alpha=0.3)

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight", dpi=300)
            plt.close(fig)

    print(f"\nForest plots saved to {output_path} ({num_plots} plots)")


def main():
    """Main function to collect and process results."""
    parser = argparse.ArgumentParser(
        description="Collect and process APO results from multiple experiment directories"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/mfs1/u/nikita/naturalv2",
        help="Path to data directory for loading experiments",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="scratch",
        help="Output directory path for saving results and plots",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="_gpt_gemini",
        help="Experiment name pattern to match subdirectories",
    )
    parser.add_argument(
        "--estimator",
        type=str,
        default="NaturalIPW",
        help="Estimator name to for which to get results",
    )

    args = parser.parse_args()

    results_dir = os.path.join(args.data_path, "results")
    os.makedirs(args.output_path, exist_ok=True)
    exp_name_clean = args.experiment_name.lstrip("_")
    csv_path = os.path.join(args.output_path, f"{exp_name_clean}_results.csv")
    plot_path = os.path.join(args.output_path, f"{exp_name_clean}_forest_plots.pdf")

    # Collect results from directories
    all_results = collect_results_from_directories(results_dir, args.experiment_name)
    final_df = process_and_combine_results(all_results, args.experiment_name)

    print("\n=== Combined Results ===")
    print(f"Total valid results: {len(final_df)}")
    final_df.to_csv(csv_path, index=False)
    print(f"Combined results saved to {csv_path}")
    create_forest_plots(
        final_df, args.data_path, plot_path, args.experiment_name, args.estimator
    )


if __name__ == "__main__":
    main()
