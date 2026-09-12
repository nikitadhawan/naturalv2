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


def test_change_from_baseline_basis_changes_reddit_prompt(tmp_path):
    arms = [make_arm("Lithium", "EXPERIMENTAL")]
    outcome_measure = make_outcome_measure(
        "Fatigue Severity Scale",
        "MEAN",
        "score on a scale",
        [("Lithium", -11.3, 24)],
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
    exp = build_experiment(
        tmp_path,
        make_completed_trial("NCT05618587", arms, [outcome_measure]),
        require_binary_endpoint=False,
    )
    outcome = exp.outcome_names[0]

    assert exp.outcome_value_bases[outcome] == "change_from_baseline"
    assert exp.avg_potential_outcomes == [-11.3]

    messages = exp.build_prompt_for_report(
        "sample_ty",
        outcome,
        "reddit",
        "Lithium helped my fatigue a lot.",
        covariate_answers={},
    )
    user_prompt = messages[-1]["content"]
    assert "Value basis: change_from_baseline" in user_prompt
    assert "A negative number means the outcome decreased" in user_prompt
    assert "Do not return an absolute follow-up score" in user_prompt


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
