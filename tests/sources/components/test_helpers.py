import pytest

from naturalv2.sources.components.helpers import (
    build_treatment_automaton,
    canonicalize_reports_for_matching,
    normalize_text_for_matching,
)


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ("  Metformin—XR 500mg  ", "metformin-xr 500 mg"),
        ("Crème™ α-tocopherol\xa0150µg", "creme alpha-tocopherol 150 ug"),
        ("co\u200doperate", "cooperate"),
        ("β-blocker", "beta-blocker"),
        ("Blinatumomab 60 µg/m²/d Step", "blinatumomab 60 ug/m 2/d step"),
        ("Libexin® 100 mg Tablets", "libexin 100 mg tablets"),
        ("B+½ OMV (Group II)", "b+1\u20442 omv (group ii)"),
        ("Arm I (AC-->WP)", "arm i (ac-->wp)"),
        ("Part 1: Ixazomib 0.25 mg/m^2", "part 1: ixazomib 0.25 mg/m 2"),
        ("  ^ ", ""),
    ],
)
def test_normalize_text_for_matching_expected_transformations(raw_text, expected):
    assert normalize_text_for_matching(raw_text) == expected


def test_normalize_text_for_matching_inserts_digit_letter_spaces():
    assert normalize_text_for_matching("Metformin10mg") == "metformin 10 mg"


def test_canonicalize_for_matching_collapses_connectors_with_mapping():
    text = "alpha-tocopherol 400mg/day"

    canonical_text, index_map = canonicalize_reports_for_matching(text)

    assert canonical_text == "alpha tocopherol 400 mg day"
    assert index_map == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        20,
        21,
        22,
        23,
        24,
        25,
    ]


def test_canonicalize_for_matching_inserts_spaces_for_letter_digit_boundaries():
    canonical_text, index_map = canonicalize_reports_for_matching("B12 and vitaminB6")

    assert canonical_text == "B 12 and vitaminB 6"
    # Space inserted between B and 1 should map back to the digit index.
    assert canonical_text[1] == " " and index_map[1] == 1
    # Space before the trailing dose should map back to the original index too.
    assert canonical_text[-2] == " " and index_map[-2] == 16
    assert canonical_text[-1] == "6" and index_map[-1] == 16


def test_canonicalize_for_matching_ignores_repeated_connectors():
    canonical_text, index_map = canonicalize_reports_for_matching("aspirin--mg")

    assert canonical_text == "aspirin mg"
    # The single space is sourced from the first hyphen index.
    assert canonical_text[7] == " " and index_map[7] == 7


def test_canonicalize_for_matching_handles_dataset_dose_format():
    normalized = normalize_text_for_matching("Blinatumomab 60 µg/m²/d Step")
    canonical_text, index_map = canonicalize_reports_for_matching(normalized)

    assert normalized == "blinatumomab 60 ug/m 2/d step"
    assert canonical_text == "blinatumomab 60 ug m 2 d step"
    assert index_map == list(range(len(normalized)))


def test_canonicalize_for_matching_retains_non_connector_symbols():
    normalized = normalize_text_for_matching("Arm I (AC-->WP)")
    canonical_text, _ = canonicalize_reports_for_matching(normalized)

    assert normalized == "arm i (ac-->wp)"
    assert canonical_text == "arm i (ac >wp)"


def test_canonicalize_for_matching_handles_empty_input():
    canonical_text, index_map = canonicalize_reports_for_matching("")
    assert canonical_text == ""
    assert index_map == []


def _find_matches(automaton, raw_text):
    normalized = normalize_text_for_matching(raw_text)
    canonical_text, _ = canonicalize_reports_for_matching(normalized)
    return {match for _, match in automaton.iter(canonical_text)}


def test_build_treatment_automaton_matches_normalized_variants():
    aliases = [
        "Metformin XR 500mg",
        "Alpha-tocopherol (oral capsule)",
        "N-acetyl cysteine",
        "Libexin® 100 mg Tablets",
        "topiramate",
        "",
        "   ",
    ]
    automaton = build_treatment_automaton(aliases)

    text = (
        "Metformin—XR 500 mg tablets were given. "
        "Alpha tocopherol oral capsule and N acetyl-cysteine were provided. "
        "Libexin 100 mg tablets were dispensed."
        "I take mytopiramate twice a day."
    )

    matches = _find_matches(automaton, text)
    assert matches == {
        "alpha tocopherol",
        "libexin 100 mg tablets",
        "metformin xr 500 mg",
        "n acetyl cysteine",
        "topiramate",
    }


def test_build_treatment_automaton_handles_parenthetical_aliases_and_skips_empty():
    aliases = ["Aspirin (Oral Tablet)", "®"]
    automaton = build_treatment_automaton(aliases)

    text = "Patient started aspirin (oral tablet) today; aspirin therapy continues."

    matches = _find_matches(automaton, text)
    assert matches == {"aspirin", "aspirin (oral tablet)"}
