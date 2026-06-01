import ast
import json
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


# Add scripts directory to path for imports
scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

# Import functions from collect_results.py
from collect_results import (
    collect_results_from_directories,
    process_and_combine_results,
)

from naturalv2.experiment import Experiment


def create_forest_plot_for_nct_id(final_df, nct_id, estimator="NaturalIPW"):
    """Create forest plots for a given nct_id and return list of figures."""
    # Filter for specified estimator rows
    estimator_df = final_df[final_df["estimator"] == estimator].copy()
    estimator_df = estimator_df.sort_values("true_response", ascending=True)

    # Filter for the given nct_id
    nct_df = estimator_df[estimator_df["nct_id"] == nct_id].copy()

    if len(nct_df) == 0:
        return []

    # Group by outcome
    grouped = nct_df.groupby("outcome")
    figures = []

    for outcome, group_df in grouped:
        # Sort by true_response
        group_df = group_df.sort_values("true_response", ascending=True)

        if len(group_df) == 0:
            continue

        fig, ax = plt.subplots(figsize=(10, max(6, len(group_df) * 0.4)))

        # Create y-axis positions (one for each treatment, starting at 1)
        y_positions = range(1, len(group_df) + 1)

        # Plot confidence intervals from CI_lower to CI_upper
        for i, (idx, row) in enumerate(group_df.iterrows()):
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
                    if isinstance(dispersion, (tuple, list)) and len(dispersion) == 2:
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
                    pass

        # Use treatments as y-axis labels
        y_labels = group_df["treatment"].tolist()

        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels, fontsize=12)
        ax.tick_params(labelsize=12)
        ax.set_xlabel("Treatment Response", fontsize=12)
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
        ax.legend(fontsize=12)
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        figures.append(fig)

    return figures


@st.cache_data
def load_results(data_path, experiment_name):
    """Load and process results with caching."""
    results_dir = os.path.join(data_path, "results")
    all_results = collect_results_from_directories(results_dir, experiment_name)
    final_df = process_and_combine_results(all_results, experiment_name)
    return final_df


@st.cache_data
def load_experiment(data_path, nct_id):
    """Load experiment from YAML file using Experiment class with caching."""
    yaml_path = os.path.join(data_path, "experiments", f"{nct_id}.yaml")
    if os.path.exists(yaml_path):
        try:
            return Experiment.from_yaml(yaml_path)
        except Exception as e:
            st.error(f"Error loading experiment: {str(e)}")
            return None
    return None


def get_nested_attr(obj, attr_path):
    """Get nested attribute using dot notation (e.g., 'treatment_common_names')."""
    attrs = attr_path.split(".")
    current = obj
    for attr in attrs:
        if hasattr(current, attr):
            current = getattr(current, attr)
        else:
            return None
    return current


def get_available_fields(experiment):
    """Get list of available fields/attributes from Experiment object."""
    # Get all properties and public attributes
    fields = []
    for attr_name in dir(experiment):
        if not attr_name.startswith("_"):
            try:
                attr = getattr(experiment, attr_name)
                # Skip methods
                if not callable(attr):
                    fields.append(attr_name)
            except:
                pass
    return sorted(fields)


def extract_data_type_from_filename(csv_file):
    """Extract data type from CSV filename."""
    data_types = sorted(
        ["ty_filter", "knowns", "imputations", "inclusion", "ty_given_x", "sample_ty"],
        key=len,
        reverse=True,
    )
    name_without_ext = csv_file[:-4] if csv_file.endswith(".csv") else csv_file

    for data_type in data_types:
        pattern = f"_{data_type}_"
        if pattern in name_without_ext:
            return data_type
    return None


def sort_files_by_data_type(files_list):
    """Sort files by data type in specified order: ty_filter, knowns, imputations, sample_ty, inclusion, ty_given_x."""
    data_type_order = [
        "ty_filter",
        "knowns",
        "imputations",
        "sample_ty",
        "inclusion",
        "ty_given_x",
    ]

    def get_sort_key(item):
        model_data_type, csv_file = item
        data_type = extract_data_type_from_filename(csv_file)
        if data_type in data_type_order:
            return data_type_order.index(data_type)
        return len(data_type_order)  # Put unknown types at the end

    return sorted(files_list, key=get_sort_key)


def get_default_columns(data_type, all_columns, covariate_names=None):
    """Get default columns to display based on data type.

    Args:
        data_type: The data type extracted from filename (e.g., 'ty_filter', 'knowns', etc.)
        all_columns: List of all available columns in the dataframe
        covariate_names: List of covariate names from the experiment (optional)

    Returns:
        List of default column names to display
    """
    defaults = []

    if data_type == "ty_filter":
        defaults = [
            "report",
            "treatments_mentioned",
            "treatment_taken_filter",
            "outcome_category_filter",
        ]
    elif data_type == "knowns":
        defaults = ["report", "meets_inclusion_criteria"] + covariate_names
    elif data_type == "imputations":
        imputed_names = [cov + "_imputed" for cov in covariate_names]
        defaults = ["report"] + imputed_names
    elif data_type == "sample_ty":
        discretized_names = [cov + "_discretized" for cov in covariate_names]
        defaults = ["report", "treatment_taken", "outcome_category"] + discretized_names
    elif data_type == "inclusion":
        discretized_names = [cov + "_discretized" for cov in covariate_names]
        sampled = [col for col in all_columns if col.endswith("_sampled")]
        defaults = ["report", "inclusion_probs"] + sampled + discretized_names
    elif data_type == "ty_given_x":
        discretized_names = [cov + "_discretized" for cov in covariate_names]
        sampled = [col for col in all_columns if col.endswith("_sampled")]
        defaults = ["report", "ty_given_x_probs"] + sampled + discretized_names

    return defaults


def get_notes_file_path(data_path):
    """Get the path to the notes JSON file."""
    return os.path.join(data_path, "results", "notes.json")


def load_notes(data_path):
    """Load notes from JSON file. Returns a dictionary mapping nct_id to notes."""
    notes_file = get_notes_file_path(data_path)
    if os.path.exists(notes_file):
        try:
            with open(notes_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            st.warning(f"Error loading notes file: {str(e)}")
            return {}
    return {}


def save_notes(data_path, notes_dict):
    """Save notes dictionary to JSON file."""
    notes_file = get_notes_file_path(data_path)
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(notes_file), exist_ok=True)
        with open(notes_file, "w") as f:
            json.dump(notes_dict, f, indent=2)
        return True
    except IOError as e:
        st.error(f"Error saving notes file: {str(e)}")
        return False


def main():
    st.set_page_config(layout="wide")
    # Add custom CSS to make buttons and columns narrower with tighter spacing
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        /* Make buttons narrower in columns */
        div[data-testid="column"] button[kind="secondary"] {
            width: auto;
            min-width: 120px;
            max-width: 200px;
        }
        /* Reduce spacing between columns */
        div[data-testid="column"] {
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("NATURALv2 Dashboard")

    # Configuration
    data_path = st.sidebar.text_input("Data Path", value="/mfs1/u/nikita/naturalv2")
    experiment_name = st.sidebar.text_input("Experiment Name", value="_gpt_gemini")
    estimator = st.sidebar.text_input("Estimator", value="NaturalIPW")

    # Load results
    try:
        final_df = load_results(data_path, experiment_name)
        st.sidebar.success(f"Loaded {len(final_df)} valid results")

        # Group nct_ids by condition
        nct_condition_map = {}
        for nct_id in final_df["nct_id"].dropna().unique():
            nct_df = final_df[final_df["nct_id"] == nct_id]
            condition = ast.literal_eval(str(nct_df.iloc[0]["conditions"]))[0]
            if condition not in nct_condition_map:
                nct_condition_map[condition] = []
            nct_condition_map[condition].append(nct_id)

        # Sort conditions and nct_ids within each condition
        for condition in nct_condition_map:
            nct_condition_map[condition] = sorted(nct_condition_map[condition])

        st.subheader("Available Results")
        # Create three columns - content in center, left and right blank
        avail_col1, avail_col2, avail_col3 = st.columns([1, 2, 1])

        with avail_col1:
            pass  # Leave blank

        with avail_col2:
            total_nct_ids = sum(len(ids) for ids in nct_condition_map.values())
            st.write(f"Found {total_nct_ids} unique NCT IDs:")

            # Initialize session state for selected nct_id
            if "selected_nct_id" not in st.session_state:
                st.session_state.selected_nct_id = None

            # Display buttons grouped by condition
            for condition, nct_ids in sorted(nct_condition_map.items()):
                st.write(f"**{condition}** ({len(nct_ids)} trials):")
                # Create columns for button layout (3 columns with bigger buttons)
                cols = st.columns(3)
                for idx, nid in enumerate(nct_ids):
                    col_idx = idx % 3
                    with cols[col_idx]:
                        if st.button(nid, key=f"btn_{nid}", width="stretch"):
                            st.session_state.selected_nct_id = nid
                st.write("")  # Add spacing between condition groups

        with avail_col3:
            pass  # Leave blank

        # Get selected nct_id from session state
        nct_id = st.session_state.selected_nct_id

        if nct_id:
            # Load notes at the start (reload if data_path changed)
            notes_path_key = f"notes_path_{data_path}"
            if (
                "notes_dict" not in st.session_state
                or st.session_state.get(notes_path_key) != data_path
            ):
                st.session_state.notes_dict = load_notes(data_path)
                st.session_state[notes_path_key] = data_path

            # Create columns for side-by-side layout with narrower proportions
            col1, col2 = st.columns([1, 1], gap="medium")

            # Left column: Experiment
            with col1:
                st.subheader(f"{nct_id} Experiment")
                experiment = load_experiment(data_path, nct_id)
                # Store experiment in session state for use in Data section
                if experiment:
                    st.session_state[f"experiment_{nct_id}"] = experiment

                if experiment:
                    # Initialize session state for selected field
                    if f"selected_field_{nct_id}" not in st.session_state:
                        st.session_state[f"selected_field_{nct_id}"] = None

                    # Get available fields
                    available_fields = get_available_fields(experiment)

                    # Display available fields as buttons for quick selection
                    if available_fields:
                        st.write("**Select field:**")
                        field_cols = st.columns(
                            min(len(available_fields), 5), gap="small"
                        )
                        for idx, field in enumerate(available_fields):
                            col_idx = idx % min(len(available_fields), 5)
                            with field_cols[col_idx]:
                                if st.button(
                                    field,
                                    key=f"field_btn_{nct_id}_{field}",
                                    width="content",
                                ):
                                    st.session_state[f"selected_field_{nct_id}"] = field

                    # Display selected field information
                    selected_field = st.session_state.get(f"selected_field_{nct_id}")
                    if selected_field:
                        field_value = get_nested_attr(experiment, selected_field)
                        if field_value is not None:
                            st.write(f"**{selected_field}:**")
                            if isinstance(field_value, (dict, list)):
                                st.json(field_value)
                            else:
                                st.write(field_value)
                        else:
                            st.error(
                                f"Field '{selected_field}' not found or not accessible."
                            )
                else:
                    st.warning(f"Experiment YAML file not found for {nct_id}")

                # Notes section
                st.subheader(f"{nct_id} Notes")
                notes_key = f"notes_{nct_id}"
                notes_saved_key = f"notes_saved_{nct_id}"

                # Get current saved notes from dict (in case they were updated elsewhere)
                current_saved_notes = st.session_state.notes_dict.get(nct_id, "")

                # Initialize notes in session state if not present, or sync if saved notes changed
                if notes_key not in st.session_state:
                    st.session_state[notes_key] = current_saved_notes
                    st.session_state[notes_saved_key] = True
                elif st.session_state[
                    notes_key
                ] != current_saved_notes and st.session_state.get(
                    notes_saved_key, False
                ):
                    # If saved notes changed externally, update session state
                    st.session_state[notes_key] = current_saved_notes

                # Text area for notes
                notes_text = st.text_area(
                    "Add or edit notes for this NCT ID:",
                    value=st.session_state[notes_key],
                    height=200,
                    key=f"notes_textarea_{nct_id}",
                    help="Notes are automatically saved when you change the text.",
                )

                # Update session state and save when notes change
                if notes_text != st.session_state[notes_key]:
                    st.session_state[notes_key] = notes_text
                    st.session_state.notes_dict[nct_id] = notes_text
                    if save_notes(data_path, st.session_state.notes_dict):
                        st.session_state[notes_saved_key] = True
                        st.success("✓ Notes saved!")
                    else:
                        st.session_state[notes_saved_key] = False
                        st.error("Failed to save notes.")

                # Display save status indicator
                if st.session_state.get(notes_saved_key, False) and notes_text:
                    st.caption("✓ Notes saved")

            # Right column: Treatment Responses
            with col2:
                # Create and display forest plots
                figures = create_forest_plot_for_nct_id(final_df, nct_id, estimator)

                if len(figures) == 0:
                    st.warning(f"No results found for NCT ID: {nct_id}")
                else:
                    st.subheader(f"{nct_id} Treatment Responses")
                    for fig in figures:
                        st.pyplot(fig)
                        plt.close(fig)

                    # Display apo_results and ate_results buttons in Treatment Responses section
                    results_dir = os.path.join(data_path, "results")
                    nct_results_dir = os.path.join(
                        results_dir, f"{nct_id}{experiment_name}"
                    )

                    if os.path.exists(nct_results_dir):
                        # Initialize session state for selected results CSV
                        if f"selected_results_csv_{nct_id}" not in st.session_state:
                            st.session_state[f"selected_results_csv_{nct_id}"] = None

                        # Check for apo_results.csv and ate_results.csv
                        results_csvs = []
                        if os.path.exists(
                            os.path.join(nct_results_dir, "apo_results.csv")
                        ):
                            results_csvs.append("apo_results.csv")
                        if os.path.exists(
                            os.path.join(nct_results_dir, "ate_results.csv")
                        ):
                            results_csvs.append("ate_results.csv")

                        if results_csvs:
                            # Display buttons for results CSVs
                            results_cols = st.columns(
                                min(len(results_csvs), 4), gap="small"
                            )
                            for idx, csv_file in enumerate(results_csvs):
                                col_idx = idx % min(len(results_csvs), 4)
                                with results_cols[col_idx]:
                                    if st.button(
                                        csv_file,
                                        key=f"results_csv_btn_{nct_id}_{csv_file}",
                                        width="content",
                                    ):
                                        st.session_state[
                                            f"selected_results_csv_{nct_id}"
                                        ] = csv_file

                            # Display selected results CSV
                            selected_results_csv = st.session_state.get(
                                f"selected_results_csv_{nct_id}"
                            )
                            if selected_results_csv:
                                csv_path = os.path.join(
                                    nct_results_dir, selected_results_csv
                                )
                                try:
                                    df = pd.read_csv(csv_path)
                                    st.write(f"**{selected_results_csv}**:")
                                    st.dataframe(df, width="stretch")
                                    st.write(
                                        f"Total rows: {len(df)}, Columns: {len(df.columns)}"
                                    )
                                except Exception as e:
                                    st.error(f"Error loading CSV file: {str(e)}")

            # Display available CSV files (excluding apo_results and ate_results)
            st.subheader(f"{nct_id} Data")
            # Create two columns - content in first, second blank
            data_col1, data_col2 = st.columns([1, 1])

            with data_col1:
                results_dir = os.path.join(data_path, "results")
                nct_results_dir = os.path.join(
                    results_dir, f"{nct_id}{experiment_name}"
                )

                if os.path.exists(nct_results_dir):
                    # Get all CSV files except apo_results.csv and ate_results.csv
                    all_csv_files = [
                        f for f in os.listdir(nct_results_dir) if f.endswith(".csv")
                    ]
                    csv_files = [
                        f
                        for f in all_csv_files
                        if f not in ["apo_results.csv", "ate_results.csv"]
                    ]

                    if csv_files:
                        # Initialize session state for selected CSV
                        if f"selected_csv_{nct_id}" not in st.session_state:
                            st.session_state[f"selected_csv_{nct_id}"] = None

                        # Parse and group CSV files by outcome
                        # Sort data_types by length (longest first) to match multi-word types like "ty_given_x" first
                        data_types = sorted(
                            [
                                "ty_filter",
                                "knowns",
                                "imputations",
                                "inclusion",
                                "ty_given_x",
                                "sample_ty",
                            ],
                            key=len,
                            reverse=True,
                        )
                        outcome_groups = {}
                        other_files = []

                        for csv_file in csv_files:
                            # Remove .csv extension
                            name_without_ext = csv_file[:-4]

                            # Try to find a data_type in the filename
                            found_data_type = None
                            data_type_start = -1
                            data_type_end = -1

                            for data_type in data_types:
                                # Search for the pattern: _{data_type}_
                                pattern = f"_{data_type}_"
                                idx = name_without_ext.find(pattern)
                                if idx != -1:
                                    found_data_type = data_type
                                    data_type_start = (
                                        idx + 1
                                    )  # Start of data_type (after first underscore)
                                    data_type_end = idx + len(
                                        pattern
                                    )  # End of pattern (after second underscore)
                                    break

                            if found_data_type and data_type_start != -1:
                                # Extract outcome (everything after _{data_type}_)
                                outcome = name_without_ext[data_type_end:]
                                # Extract model_name and data_type (everything up to and including the data_type)
                                # This gives us {model_name}_{data_type}
                                model_data_type = name_without_ext[
                                    : data_type_end - 1
                                ]  # -1 to remove trailing underscore

                                if outcome not in outcome_groups:
                                    outcome_groups[outcome] = []
                                outcome_groups[outcome].append(
                                    (model_data_type, csv_file)
                                )
                            else:
                                # File doesn't match the pattern, keep it separate
                                other_files.append(csv_file)

                        # Display files grouped by outcome
                        for outcome in sorted(outcome_groups.keys()):
                            st.write(f"**{outcome}**")
                            files_for_outcome = sort_files_by_data_type(
                                outcome_groups[outcome]
                            )
                            csv_cols = st.columns(
                                min(len(files_for_outcome), 6), gap="small"
                            )
                            for idx, (model_data_type, csv_file) in enumerate(
                                files_for_outcome
                            ):
                                col_idx = idx % min(len(files_for_outcome), 6)
                                with csv_cols[col_idx]:
                                    if st.button(
                                        model_data_type,
                                        key=f"csv_btn_{nct_id}_{csv_file}",
                                        width="content",
                                    ):
                                        st.session_state[f"selected_csv_{nct_id}"] = (
                                            csv_file
                                        )
                                        # Load dataframe immediately when button is clicked
                                        csv_path = os.path.join(
                                            nct_results_dir, csv_file
                                        )
                                        try:
                                            df = pd.read_csv(csv_path)
                                            st.session_state[
                                                f"df_{nct_id}_{csv_file}"
                                            ] = df
                                            # Initialize default columns
                                            selected_cols_key = (
                                                f"selected_cols_{nct_id}_{csv_file}"
                                            )
                                            if (
                                                selected_cols_key
                                                not in st.session_state
                                            ):
                                                data_type = (
                                                    extract_data_type_from_filename(
                                                        csv_file
                                                    )
                                                )
                                                experiment = st.session_state.get(
                                                    f"experiment_{nct_id}"
                                                )
                                                covariate_names = (
                                                    experiment.covariate_names
                                                    if experiment
                                                    else None
                                                )
                                                default_cols = get_default_columns(
                                                    data_type,
                                                    list(df.columns),
                                                    covariate_names,
                                                )
                                                st.session_state[selected_cols_key] = (
                                                    default_cols
                                                    if default_cols
                                                    else list(df.columns)
                                                )
                                        except Exception as e:
                                            st.error(
                                                f"Error loading CSV file: {str(e)}"
                                            )
                            st.write("")  # Add spacing between outcome groups

                        # Display other files that don't match the pattern
                        if other_files:
                            st.write("**Other files**")
                            other_cols = st.columns(
                                min(len(other_files), 6), gap="small"
                            )
                            for idx, csv_file in enumerate(sorted(other_files)):
                                col_idx = idx % min(len(other_files), 6)
                                with other_cols[col_idx]:
                                    if st.button(
                                        csv_file,
                                        key=f"csv_btn_{nct_id}_{csv_file}",
                                        width="content",
                                    ):
                                        st.session_state[f"selected_csv_{nct_id}"] = (
                                            csv_file
                                        )
                                        # Load dataframe immediately when button is clicked
                                        csv_path = os.path.join(
                                            nct_results_dir, csv_file
                                        )
                                        try:
                                            df = pd.read_csv(csv_path)
                                            st.session_state[
                                                f"df_{nct_id}_{csv_file}"
                                            ] = df
                                            # Initialize default columns
                                            selected_cols_key = (
                                                f"selected_cols_{nct_id}_{csv_file}"
                                            )
                                            if (
                                                selected_cols_key
                                                not in st.session_state
                                            ):
                                                data_type = (
                                                    extract_data_type_from_filename(
                                                        csv_file
                                                    )
                                                )
                                                experiment = st.session_state.get(
                                                    f"experiment_{nct_id}"
                                                )
                                                covariate_names = (
                                                    experiment.covariate_names
                                                    if experiment
                                                    else None
                                                )
                                                default_cols = get_default_columns(
                                                    data_type,
                                                    list(df.columns),
                                                    covariate_names,
                                                )
                                                st.session_state[selected_cols_key] = (
                                                    default_cols
                                                    if default_cols
                                                    else list(df.columns)
                                                )
                                        except Exception as e:
                                            st.error(
                                                f"Error loading CSV file: {str(e)}"
                                            )
                    else:
                        st.write("No other CSV files found in this directory.")
                else:
                    st.write(f"Results directory not found: {nct_results_dir}")

            with data_col2:
                # Column selection for the selected CSV
                selected_csv = st.session_state.get(f"selected_csv_{nct_id}")
                if selected_csv:
                    df_key = f"df_{nct_id}_{selected_csv}"
                    if df_key in st.session_state:
                        df = st.session_state[df_key]
                        st.write("**Select Columns:**")
                        selected_cols_key = f"selected_cols_{nct_id}_{selected_csv}"
                        # Get current selection or initialize with defaults
                        current_selection = st.session_state.get(selected_cols_key)
                        if current_selection is None:
                            # Extract data type and get default columns
                            data_type = extract_data_type_from_filename(selected_csv)
                            # Get covariate_names from experiment if available
                            experiment = st.session_state.get(f"experiment_{nct_id}")
                            covariate_names = (
                                experiment.covariate_names if experiment else None
                            )
                            current_selection = get_default_columns(
                                data_type, list(df.columns), covariate_names
                            )
                            if not current_selection:
                                current_selection = list(df.columns)
                            st.session_state[selected_cols_key] = current_selection

                        # Ensure selected columns are valid
                        valid_selection = [
                            col for col in current_selection if col in df.columns
                        ]
                        if not valid_selection:
                            valid_selection = list(df.columns)

                        # Use checkboxes instead of multiselect for consistent styling
                        st.write("Choose columns to display:")
                        selected_cols = []
                        # Create 4 columns for checkboxes
                        checkbox_cols = st.columns(4)
                        for idx, col in enumerate(df.columns):
                            col_idx = idx % 4
                            is_selected = col in valid_selection
                            with checkbox_cols[col_idx]:
                                checkbox_value = st.checkbox(
                                    col,
                                    value=is_selected,
                                    key=f"col_check_{nct_id}_{selected_csv}_{col}",
                                )
                                if checkbox_value:
                                    selected_cols.append(col)

                        # Update session state with selected columns
                        if selected_cols:
                            st.session_state[selected_cols_key] = selected_cols
                        else:
                            st.info("Select at least one column to display.")
                else:
                    st.write("Select a CSV file to view column options.")

            # Display dataframe at full width below the columns
            selected_csv = st.session_state.get(f"selected_csv_{nct_id}")
            if selected_csv:
                results_dir = os.path.join(data_path, "results")
                nct_results_dir = os.path.join(
                    results_dir, f"{nct_id}{experiment_name}"
                )

                if os.path.exists(nct_results_dir):
                    csv_path = os.path.join(nct_results_dir, selected_csv)
                    try:
                        # Load or get dataframe from session state
                        df_key = f"df_{nct_id}_{selected_csv}"
                        if df_key not in st.session_state:
                            df = pd.read_csv(csv_path)
                            st.session_state[df_key] = df

                            # Initialize default columns if not set
                            selected_cols_key = f"selected_cols_{nct_id}_{selected_csv}"
                            if selected_cols_key not in st.session_state:
                                data_type = extract_data_type_from_filename(
                                    selected_csv
                                )
                                experiment = st.session_state.get(
                                    f"experiment_{nct_id}"
                                )
                                covariate_names = (
                                    experiment.covariate_names if experiment else None
                                )
                                default_cols = get_default_columns(
                                    data_type, list(df.columns), covariate_names
                                )
                                st.session_state[selected_cols_key] = (
                                    default_cols if default_cols else list(df.columns)
                                )
                        else:
                            df = st.session_state[df_key]

                        # Get selected columns and filter dataframe
                        selected_cols_key = f"selected_cols_{nct_id}_{selected_csv}"
                        selected_cols = st.session_state.get(
                            selected_cols_key, list(df.columns)
                        )
                        display_df = df[selected_cols] if selected_cols else df

                        st.write(f"**{selected_csv}**:")
                        # Display random sample
                        sample_size = min(20, len(display_df))
                        sample_df = (
                            display_df.sample(n=sample_size, random_state=42)
                            if len(display_df) > sample_size
                            else display_df
                        )
                        st.dataframe(sample_df, width="stretch")
                        st.write(
                            f"Total rows: {len(df)}, Columns: {len(display_df.columns)}"
                        )
                    except Exception as e:
                        st.error(f"Error loading CSV file: {str(e)}")

    except Exception as e:
        st.error(f"Error loading results: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
