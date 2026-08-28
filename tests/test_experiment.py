import pytest

from naturalv2.experiment import Experiment
from tests.factories import (
    build_experiment,
    make_active_trial,
    make_arm,
    make_completed_trial,
    make_outcome_measure,
)


# -- Arm-type filtering (completed trials) -----------------------------------


def test_placebo_and_sham_arms_excluded(tmp_path):
    arms = [
        make_arm("Drug A", "EXPERIMENTAL"),
        make_arm("Placebo", "PLACEBO_COMPARATOR"),
        make_arm("Sham", "SHAM_COMPARATOR"),
        make_arm("Usual Care", "NO_INTERVENTION"),
        make_arm("Misc", "OTHER"),
    ]
    outcomes = [
        make_outcome_measure(
            "Number of Participants with Response",
            "COUNT_OF_PARTICIPANTS",
            "Participants",
            [(a["label"], 10, 50) for a in arms],
        )
    ]
    exp = build_experiment(tmp_path, make_completed_trial("NCT001", arms, outcomes))
    assert exp.treatment_names == ["Drug A"]


def test_combination_arm_with_placebo_in_name_is_kept(tmp_path):
    arms = [
        make_arm("Drug A", "EXPERIMENTAL"),
        make_arm("Drug A + Placebo B", "EXPERIMENTAL"),
        make_arm("Placebo", "PLACEBO_COMPARATOR"),
    ]
    outcomes = [
        make_outcome_measure(
            "Number of Participants with Response",
            "COUNT_OF_PARTICIPANTS",
            "Participants",
            [(a["label"], 10, 50) for a in arms],
        )
    ]
    exp = build_experiment(tmp_path, make_completed_trial("NCT002", arms, outcomes))
    assert exp.treatment_names == ["Drug A", "Drug A + Placebo B"]


def test_arm_with_missing_type_excluded(tmp_path):
    arms = [make_arm("Drug A", "EXPERIMENTAL"), {"label": "Unknown Arm", "type": None}]
    outcomes = [
        make_outcome_measure(
            "Number of Participants with Response",
            "COUNT_OF_PARTICIPANTS",
            "Participants",
            [("Drug A", 10, 50)],
        )
    ]
    exp = build_experiment(tmp_path, make_completed_trial("NCT003", arms, outcomes))
    assert exp.treatment_names == ["Drug A"]


def test_active_trial_arm_filtering(tmp_path):
    arms = [
        make_arm("Drug A", "EXPERIMENTAL"),
        make_arm("Placebo", "PLACEBO_COMPARATOR"),
    ]
    trial = make_active_trial("NCT004", arms, ["Number of Participants with Response"])
    exp = build_experiment(tmp_path, trial, status="active")
    assert exp.treatment_names == ["Drug A"]
    assert exp.avg_potential_outcomes == []  # no ground truth for active trials


# -- Outcome type / discretization -------------------------------------------


def test_count_outcome_is_binary_and_normalized_by_denom(tmp_path):
    arms = [make_arm("Drug A", "EXPERIMENTAL")]
    outcomes = [
        make_outcome_measure(
            "Number of Participants with Response",
            "COUNT_OF_PARTICIPANTS",
            "Participants",
            [("Drug A", 10, 50)],
        )
    ]
    exp = build_experiment(tmp_path, make_completed_trial("NCT005", arms, outcomes))
    assert exp.is_binary_outcome(exp.outcome_names[0])
    assert exp.avg_potential_outcomes == [0.2]


def test_percent_unit_normalizes_by_100_regardless_of_param_type(tmp_path):
    arms = [make_arm("Drug A", "EXPERIMENTAL")]
    outcomes = [
        make_outcome_measure(
            "Mean Percent Change", "MEAN", "percentage", [("Drug A", 40, 50)]
        )
    ]
    exp = build_experiment(
        tmp_path,
        make_completed_trial("NCT006", arms, outcomes),
        require_binary_endpoint=False,
    )
    assert exp.avg_potential_outcomes == [0.4]


def test_continuous_outcome_needs_require_binary_endpoint_false(tmp_path):
    arms = [make_arm("Drug A", "EXPERIMENTAL")]
    outcomes = [
        make_outcome_measure(
            "Mean Change in Pain Score", "MEAN", "points", [("Drug A", 3.5, 50)]
        )
    ]
    trial = make_completed_trial("NCT007", arms, outcomes)

    # Default require_binary_endpoint=True drops it: title has no binary phrasing.
    exp_default = build_experiment(tmp_path, trial)
    assert exp_default.outcome_names == []

    exp = build_experiment(
        tmp_path,
        make_completed_trial("NCT008", arms, outcomes),
        require_binary_endpoint=False,
    )
    assert exp.outcome_names == ["Mean Change in Pain Score"]
    assert not exp.is_binary_outcome(exp.outcome_names[0])
    assert exp.options[exp.outcome_names[0]] == []
    assert exp.avg_potential_outcomes == [3.5]


def test_active_trial_outcome_falls_back_to_title_heuristic(tmp_path):
    arms = [make_arm("Drug A", "EXPERIMENTAL")]
    trial = make_active_trial(
        "NCT009", arms, ["Number of Participants with Adverse Events"]
    )
    exp = build_experiment(tmp_path, trial, status="active")
    assert exp.is_binary_outcome(exp.outcome_names[0])


# -- APO / ATE ground-truth wiring --------------------------------------------


def test_apo_and_ate_ground_truth_for_two_arms(tmp_path):
    arms = [make_arm("Drug A", "EXPERIMENTAL"), make_arm("Drug B", "ACTIVE_COMPARATOR")]
    outcomes = [
        make_outcome_measure(
            "Number of Participants with Response",
            "COUNT_OF_PARTICIPANTS",
            "Participants",
            [("Drug A", 10, 50), ("Drug B", 20, 50)],
        )
    ]
    exp = build_experiment(tmp_path, make_completed_trial("NCT010", arms, outcomes))
    outcome = exp.outcome_names[0]

    assert [outcome, "Drug A"] in exp.apo_outcome_treatment
    assert [outcome, "Drug B"] in exp.apo_outcome_treatment
    assert [outcome, ["Drug A", "Drug B"]] in exp.outcome_treatment

    idx = exp.outcome_treatment.index([outcome, ["Drug A", "Drug B"]])
    assert exp.effect_sizes[idx] == 0.2  # 0.4 - 0.2


def test_single_arm_trial_has_no_ate_ground_truth(tmp_path):
    arms = [make_arm("Drug A", "EXPERIMENTAL")]
    outcomes = [
        make_outcome_measure(
            "Number of Participants with Response",
            "COUNT_OF_PARTICIPANTS",
            "Participants",
            [("Drug A", 10, 50)],
        )
    ]
    exp = build_experiment(tmp_path, make_completed_trial("NCT011", arms, outcomes))
    assert exp.outcome_treatment == []
    assert exp.apo_outcome_treatment == [[exp.outcome_names[0], "Drug A"]]


# -- Continuous outcomes: trial specification in prompts -----------------------


def make_fss_change_experiment(tmp_path):
    """Lithium trial reporting the Fatigue Severity Scale as a change from baseline."""
    arms = [make_arm("Lithium", "EXPERIMENTAL")]
    outcome_measure = make_outcome_measure(
        "Fatigue Severity Scale", "MEAN", "score on a scale", [("Lithium", -11.3, 24)]
    )
    outcome_measure.update(
        {
            "description": (
                "7-item questionnaire assessing fatigue severity. Score range 1-49 "
                "with higher values signifying worse outcome"
            ),
            "timeFrame": "Change from baseline to day 21",
        }
    )
    return build_experiment(
        tmp_path,
        make_completed_trial("NCT05618587", arms, [outcome_measure]),
        require_binary_endpoint=False,
    )


def make_binary_experiment(tmp_path):
    arms = [make_arm("Drug A", "EXPERIMENTAL")]
    outcome_measure = make_outcome_measure(
        "Number of Participants with Response",
        "COUNT_OF_PARTICIPANTS",
        "Participants",
        [("Drug A", 10, 50)],
    )
    outcome_measure["timeFrame"] = "12 weeks"
    return build_experiment(
        tmp_path, make_completed_trial("NCT010", arms, [outcome_measure])
    )


def _user_prompt(exp, prompt_type):
    messages = exp.build_prompt_for_report(
        prompt_type,
        exp.outcome_names[0],
        "reddit",
        "Lithium helped my fatigue a lot.",
        covariate_answers={},
    )
    return messages[-1]["content"]


def test_continuous_experiment_exposes_unit(tmp_path):
    exp = make_fss_change_experiment(tmp_path)
    outcome = exp.outcome_names[0]
    assert exp.outcome_unit(outcome) == "score on a scale"
    assert exp.avg_potential_outcomes == [-11.3]  # label side is untouched


def test_continuous_sample_ty_prompt_states_trial_specification(tmp_path):
    prompt = _user_prompt(make_fss_change_experiment(tmp_path), "sample_ty")
    assert "Units: score on a scale" in prompt
    assert "Timeframe: Change from baseline to day 21" in prompt
    assert "Valid range" not in prompt  # parsed range is for the filter only
    assert "ABSOLUTE value at follow-up or a CHANGE FROM" in prompt
    assert "outcome_basis" in prompt


def test_continuous_ty_filter_prompt_asks_whether_value_is_estimable(tmp_path):
    prompt = _user_prompt(make_fss_change_experiment(tmp_path), "ty_filter")
    assert "enough information to estimate" in prompt
    assert "Timeframe: Change from baseline to day 21" in prompt
    assert "opposite category" not in prompt  # the binary-only guideline


def test_binary_prompts_get_trial_metadata_but_keep_their_questions(tmp_path):
    exp = make_binary_experiment(tmp_path)
    sample_ty = _user_prompt(exp, "sample_ty")
    ty_filter = _user_prompt(exp, "ty_filter")
    for prompt in (sample_ty, ty_filter):
        assert "Units: Participants" in prompt
        assert "Timeframe: 12 weeks" in prompt
        assert "outcome_basis" not in prompt
    assert "Options: ['No', 'Yes']" in sample_ty
    assert "opposite category" in ty_filter
    assert "enough information to estimate" not in ty_filter


def test_outcome_unit_survives_the_yaml_round_trip(tmp_path):
    # Experiments are persisted by create_study and reloaded by estimate_ate.
    exp = make_fss_change_experiment(tmp_path)
    exp._drugbank_names = {"Lithium": []}
    path = tmp_path / "experiment.yaml"

    exp.to_yaml(str(path))
    loaded = Experiment.from_yaml(str(path))

    outcome = loaded.outcome_names[0]
    assert loaded.outcome_unit(outcome) == "score on a scale"


@pytest.mark.parametrize(
    "prompt_type", ["sample_ty", "ty_filter", "relevance", "conditionals"]
)
def test_missing_descriptions_are_omitted_not_rendered_as_none(tmp_path, prompt_type):
    # CT.gov outcome measures and arms often have no description; the templates
    # used to print the literal "None" for them.
    exp = make_binary_experiment(tmp_path)
    exp.treatment_common_names = {"reddit": {"Drug A": ["drug a"]}}
    assert exp.outcome_desc[exp.outcome_names[0]] is None
    assert exp.treatment_desc["Drug A"] is None

    prompt = _user_prompt(exp, prompt_type)

    assert "Description: None" not in prompt
    assert ": None" not in prompt
    assert "Number of Participants with Response" in prompt
    if prompt_type in ("ty_filter", "relevance"):
        assert "Treatment: " in prompt and "Common names: ['drug a']" in prompt
