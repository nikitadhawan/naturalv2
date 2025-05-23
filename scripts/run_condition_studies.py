import argparse
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


CONDITION_LISTS = [
    ["Animal Diseases"],
    ["Cardiovascular Diseases"],
    ["Congenital, Hereditary, and Neonatal Diseases and Abnormalities"],
    ["Digestive System Diseases"],
    ["Disorders of Environmental Origin"],
    ["Endocrine System Diseases"],
    ["Eye Diseases"],
    ["Hemic and Lymphatic Diseases"],
    ["Immune System Diseases"],
    ["Infections"],
    ["Musculoskeletal Diseases"],
    ["Neoplasms"],
    ["Nervous System Diseases"],
    ["Nutritional and Metabolic Diseases"],
    ["Occupational Diseases"],
    ["Otorhinolaryngologic Diseases"],
    ["Pathological Conditions, Signs and Symptoms"],
    ["Respiratory Tract Diseases"],
    ["Skin and Connective Tissue Diseases"],
    ["Stomatognathic Diseases"],
    ["Urogenital Diseases"],
    ["Wounds and Injuries"],
]


# Function to run the study script and extract the results
def run_study(conditions: list[str], args: argparse.Namespace) -> dict[str, Any]:
    # Convert the list to a string for the command
    conditions_str = str(conditions)

    # Run the command using the provided path
    cmd = f"python {args.script_path} conditions={conditions_str} save_path={args.output_dir}"
    logger.info(f"Running: {cmd}")
    process = subprocess.Popen(
        cmd,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,  # line buffered
    )

    output_lines = []
    # Read and display output in real-time
    for line in iter(process.stdout.readline, ""):
        print(line, end="")  # display in real-time
        output_lines.append(line)  # store for later processing
        sys.stdout.flush()  # ensure it flushes to the console

    process.stdout.close()
    return_code = process.wait()

    if return_code != 0:
        logger.error(f"Error running command: {cmd}")
        return {
            "conditions": conditions,
            "train_trials": None,
            "train_labels": None,
            "val_trials": None,
            "val_labels": None,
            "test_trials": None,
            "test_labels": None,
        }

    # Join all lines to process with regex
    output = "".join(output_lines)

    # Parse the output to extract the numbers
    train_trials = re.search(r"Train: (\d+) trials", output)
    train_labels = re.search(r"Train: \d+ trials, (\d+) labels", output)
    val_trials = re.search(r"Val: (\d+) trials", output)
    val_labels = re.search(r"Val: \d+ trials, (\d+) labels", output)
    test_trials = re.search(r"Test: (\d+) trials", output)
    test_labels = re.search(r"Test: \d+ trials, (\d+) labels", output)

    # Extract the numbers or set to None if not found
    train_trials = int(train_trials.group(1)) if train_trials else None
    train_labels = int(train_labels.group(1)) if train_labels else None
    val_trials = int(val_trials.group(1)) if val_trials else None
    val_labels = int(val_labels.group(1)) if val_labels else None
    test_trials = int(test_trials.group(1)) if test_trials else None
    test_labels = int(test_labels.group(1)) if test_labels else None

    return {
        "conditions": conditions,
        "train_trials": train_trials,
        "train_labels": train_labels,
        "val_trials": val_trials,
        "val_labels": val_labels,
        "test_trials": test_trials,
        "test_labels": test_labels,
    }


def count_unique_ncts(studies_dir: str) -> int:
    unique_ncts = set()

    for yaml_file in Path(studies_dir).glob("**/*.yaml"):
        with open(yaml_file, "r") as f:
            study_data = yaml.safe_load(f)

            # Extract NCT IDs from train, val, and test trials
            for trial_list in [
                study_data["train_trials"],
                study_data["val_trials"],
                study_data["test_trials"],
            ]:
                for trial in trial_list:
                    # Each trial is a dict with one key (the NCT ID)
                    unique_ncts.update(trial.keys())

    return len(unique_ncts)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run create_study with different conditions and record stats"
    )
    parser.add_argument(
        "--script_path",
        type=str,
        default="create_study.py",
        help="Absolute or relative path to the create_study.py",
    )
    parser.add_argument(
        "--output_dir", type=str, default=".", help="Directory to save the output files"
    )
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Run all the studies and collect results
    results = []
    for conditions in CONDITION_LISTS:
        results.append(run_study(conditions, args))

    # Create a DataFrame for easier manipulation
    df = pd.DataFrame(results)

    # Save to CSV
    csv_path = os.path.join(args.output_dir, "create_study_results.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")
    logger.info(
        f"{count_unique_ncts(os.path.join(args.output_dir, 'studies'))} unique trials covered."
    )
