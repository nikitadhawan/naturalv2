import argparse
import datetime
import os

import matplotlib.pyplot as plt

from naturalv2.evals.experiment import Experiment
from naturalv2.study import StudyDataset, get_study_filepaths


# Usage: python -m scripts.analyze_datasets --data_path /mfs1/u/nikita/naturalv2 --output_dir scratch --study hemic_and_lymphatic_diseases


def plot_effects(data_sizes, avg_effect_sizes, save_path):
    plt.figure(figsize=(10, 6))
    plt.scatter(
        avg_effect_sizes,
        data_sizes,
        label="Avg Abs Effect Size",
        marker="s",
        color="green",
    )
    plt.xlabel("Absolute Effect Size")
    plt.ylabel("Data Size")
    plt.yscale("log")
    plt.title("Absolute average effect sizes vs Reddit data size")
    # plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)


def plot_dates(data_sizes, utc_dates, date_labels, save_path):
    plt.figure(figsize=(10, 6))
    plt.scatter(utc_dates, data_sizes)
    plt.xlabel("Trial End Date")
    plt.ylabel("Data Size")
    plt.yscale("log")
    plt.title("Trial end dates vs Reddit data sizes")
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
    total_trials = len(data_sizes)
    fractions = [
        len([ds for ds in data_sizes if ds > t]) / total_trials for t in thresholds
    ]
    for t, frac in zip(thresholds, fractions):
        if t == 0:
            plt.text(xtick_positions[0], 1.5, f"Total: {total_trials}", color="k")
        else:
            plt.text(xtick_positions[0], t, f">{t}: {frac:.2%}", color="k")
    plt.tight_layout()
    plt.savefig(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--study", type=str)
    args = parser.parse_args()

    experiment_dir = os.path.join(args.data_path, "experiments")
    study_dataset_path = get_study_filepaths(args.data_path, args.study)[
        "study_dataset"
    ]
    study_dataset = StudyDataset.from_yaml(study_dataset_path)

    utc_dates, date_labels = [], []
    avg_effect_sizes, data_sizes = [], []
    ncts = []

    for nctid, data_size in study_dataset.data_sizes.items():
        nct_id = nctid.split("_", 1)[1] if "_" in nctid else nctid
        exp_file = os.path.join(experiment_dir, f"{nct_id}.yaml")
        try:
            exp = Experiment.from_yaml(exp_file)
        except:
            print(f"No data curated for {nct_id}.")
            exp = Experiment(args.data_path, nct_id, status="completed")
        try:
            date_obj = datetime.datetime.strptime(exp.date, "%Y-%m-%d")
        except ValueError:
            date_obj = datetime.datetime.strptime(exp.date, "%Y-%m")
        date = int(date_obj.replace(tzinfo=datetime.timezone.utc).timestamp())
        if "test" not in exp.trial_path:
            effect_sizes = [abs(effect) for effect in exp.effect_sizes]
            avg_effect_sizes.append(sum(effect_sizes) / len(effect_sizes))
        utc_dates.append(date)
        date_labels.append(date_obj)
        data_sizes.append(data_size)
        ncts.append(nct_id)

    sorted_lists = sorted(
        zip(utc_dates, date_labels, avg_effect_sizes, data_sizes, ncts),
        key=lambda x: x[0],
    )
    utc_dates, date_labels, avg_effect_sizes, data_sizes, ncts = map(
        list, zip(*sorted_lists)
    )

    save_path = os.path.join(args.output_dir, f"{args.study}_reddit_effect_sizes.png")
    plot_effects(data_sizes, avg_effect_sizes, save_path)
    save_path = os.path.join(args.output_dir, f"{args.study}_reddit_dates.png")
    plot_dates(data_sizes, utc_dates, date_labels, save_path)
