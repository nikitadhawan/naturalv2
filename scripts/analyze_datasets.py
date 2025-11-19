import argparse
import datetime
import os

import matplotlib.pyplot as plt

from naturalv2.clinical_trial import ClinicalTrial
from naturalv2.experiment import Experiment
from naturalv2.study import StudyDataset, get_study_filepaths


# Usage: python -m scripts.analyze_datasets --data_path /mfs1/u/nikita/naturalv2 --output_dir scratch --study hemic_and_lymphatic_diseases


def plot_effects(data_sizes, avg_effect_sizes, is_filtered, save_path, use_apo=False):
    plt.figure(figsize=(10, 6))

    # Only plot filtered data (ignore no_data_filter trials)
    filtered_sizes = [ds for ds, filt in zip(data_sizes, is_filtered) if filt]
    filtered_effects = [es for es, filt in zip(avg_effect_sizes, is_filtered) if filt]

    if filtered_sizes:
        plt.scatter(
            filtered_effects,
            filtered_sizes,
            marker="s",
            color="green",
        )

    if use_apo:
        plt.xlabel("Average Potential Outcomes")
        plt.title("Average potential outcomes vs Reddit data size")
    else:
        plt.xlabel("Absolute Effect Size")
        plt.title("Absolute average effect sizes vs Reddit data size")
    plt.ylabel("Data Size")
    plt.yscale("log")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)


def plot_dates(
    data_sizes, utc_dates, date_labels, is_filtered, save_path, include_unfiltered=False
):
    plt.figure(figsize=(10, 6))

    # Separate filtered and unfiltered data
    filtered_dates = [dt for dt, filt in zip(utc_dates, is_filtered) if filt]
    filtered_sizes = [ds for ds, filt in zip(data_sizes, is_filtered) if filt]
    unfiltered_dates = [dt for dt, filt in zip(utc_dates, is_filtered) if not filt]
    unfiltered_sizes = [ds for ds, filt in zip(data_sizes, is_filtered) if not filt]

    if filtered_dates:
        if include_unfiltered and unfiltered_dates:
            plt.scatter(
                filtered_dates,
                filtered_sizes,
                label="Date filtered",
                color="green",
                marker="o",
            )
        else:
            plt.scatter(filtered_dates, filtered_sizes, color="green", marker="o")
    if unfiltered_dates and include_unfiltered:
        plt.scatter(
            unfiltered_dates,
            unfiltered_sizes,
            label="No date filter",
            color="orange",
            marker="o",
        )

    plt.xlabel("Trial End Date")
    plt.ylabel("Data Size")
    plt.yscale("log")
    plt.title("Trial end dates vs Reddit data sizes")
    if include_unfiltered and unfiltered_dates:
        plt.legend()
    plt.grid(True)

    num_xticks = 15
    if len(utc_dates) <= num_xticks:
        xtick_indices = list(range(len(utc_dates)))
    else:
        xtick_indices = [
            round(i * (len(utc_dates) - 1) / (num_xticks - 1))
            for i in range(num_xticks - 1)
        ]
        if (len(utc_dates) - 1) not in xtick_indices:
            xtick_indices.append(len(utc_dates) - 1)

    xtick_positions = [utc_dates[i] for i in xtick_indices]
    xtick_labels = [date_labels[i].strftime("%Y-%m-%d") for i in xtick_indices]
    plt.xticks(xtick_positions, xtick_labels, rotation=45, ha="right")

    thresholds = [0, 10, 100, 1000, 10000, 50000]
    total_filtered = len(filtered_sizes)
    total_unfiltered = len(unfiltered_sizes)
    filtered_counts = [len([ds for ds in filtered_sizes if ds > t]) for t in thresholds]
    unfiltered_counts = [
        len([ds for ds in unfiltered_sizes if ds > t]) for t in thresholds
    ]
    max_data_size = max(data_sizes) if data_sizes else 1

    for t, filt_count, unfilt_count in zip(
        thresholds, filtered_counts, unfiltered_counts
    ):
        if t == 0:
            # Display total counts
            y_pos = max_data_size * 2
            plt.text(xtick_positions[0], y_pos, "Total:", color="k")
            if include_unfiltered:
                plt.text(
                    xtick_positions[0], y_pos / 1.5, f"{total_filtered},", color="green"
                )
                plt.text(
                    xtick_positions[0],
                    y_pos / 1.5,
                    f"    {total_unfiltered}",
                    color="orange",
                )
            else:
                plt.text(
                    xtick_positions[0], y_pos / 1.5, f"{total_filtered}", color="green"
                )
        else:
            # Display threshold counts on one line below threshold
            plt.text(xtick_positions[0], t, f">{t}:", color="k")
            if include_unfiltered:
                plt.text(xtick_positions[0], t / 1.8, f"{filt_count},", color="green")
                plt.text(
                    xtick_positions[0], t / 1.8, f"    {unfilt_count}", color="orange"
                )
            else:
                plt.text(xtick_positions[0], t / 1.8, f"{filt_count}", color="green")
    plt.tight_layout()
    plt.savefig(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--study", type=str)
    parser.add_argument("--apo", action="store_true", default=False)
    parser.add_argument(
        "--include_unfiltered",
        action="store_true",
        default=False,
        help="Include 'no date filter' data points in plots",
    )
    args = parser.parse_args()

    experiment_dir = os.path.join(args.data_path, "experiments")
    study_dataset_path = get_study_filepaths(args.data_path, args.study, apo=args.apo)[
        "study_dataset"
    ]
    study_dataset = StudyDataset.from_yaml(study_dataset_path)

    utc_dates, date_labels = [], []
    avg_effect_sizes, data_sizes = [], []
    ncts = []
    is_filtered = []

    for nctid, data_size in study_dataset.data_sizes["reddit"].items():
        # Check if this is filtered data (no suffix) or unfiltered (has _no_date_filter)
        if nctid.endswith("_no_date_filter"):
            nct_id = nctid.rsplit("_no_date_filter", 1)[0]
            filtered = False
        else:
            nct_id = nctid
            filtered = True
        exp_file = os.path.join(experiment_dir, f"{nct_id}.yaml")
        try:
            exp = Experiment.from_yaml(exp_file)
            trial = ClinicalTrial.from_json_file(exp.trial_path)
            exp._avg_potential_outcomes = []
            exp._set_outcome_treatment_effects(trial)
        except:
            print(f"No data curated for {nct_id}.")
            exp = Experiment(args.data_path, nct_id, status="completed")
        try:
            date_obj = datetime.datetime.strptime(exp.date, "%Y-%m-%d")
        except ValueError:
            date_obj = datetime.datetime.strptime(exp.date, "%Y-%m")
        date = int(date_obj.replace(tzinfo=datetime.timezone.utc).timestamp())
        if "test" not in exp.trial_path:
            if args.apo:
                # Use average potential outcomes
                values = exp.avg_potential_outcomes
                avg_effect_sizes.append(sum(values) / len(values))
            else:
                # Use effect sizes
                effect_sizes = [abs(effect) for effect in exp.effect_sizes]
                avg_effect_sizes.append(sum(effect_sizes) / len(effect_sizes))
        utc_dates.append(date)
        date_labels.append(date_obj)
        data_sizes.append(data_size)
        ncts.append(nct_id)
        is_filtered.append(filtered)

    sorted_lists = sorted(
        zip(utc_dates, date_labels, avg_effect_sizes, data_sizes, ncts, is_filtered),
        key=lambda x: x[0],
    )
    utc_dates, date_labels, avg_effect_sizes, data_sizes, ncts, is_filtered = map(
        list, zip(*sorted_lists)
    )

    apo_suffix = "_apo" if args.apo else ""
    save_path = os.path.join(
        args.output_dir, f"{args.study}{apo_suffix}_reddit_effect_sizes.png"
    )
    plot_effects(data_sizes, avg_effect_sizes, is_filtered, save_path, use_apo=args.apo)
    save_path = os.path.join(
        args.output_dir, f"{args.study}{apo_suffix}_reddit_dates.png"
    )
    plot_dates(
        data_sizes,
        utc_dates,
        date_labels,
        is_filtered,
        save_path,
        include_unfiltered=args.include_unfiltered,
    )
