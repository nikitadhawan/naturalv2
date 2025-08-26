import argparse

from naturalv2.clinical_trial import download_clinical_trials


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="nct_reports")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    download_clinical_trials(args.data_path, args.test)
