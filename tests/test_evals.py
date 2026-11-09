import json
from pathlib import Path

import pytest

from misterio.agent import run
from misterio.contracts import Job, JobStatus
from misterio.llm import FakeLlm
from misterio.models import MODEL_OPTIONS

CASES_PATH = Path(__file__).parents[1] / "evals" / "chastushki.json"
CASES = json.loads(CASES_PATH.read_text(encoding="utf-8"))
FIXTURE_CASES = [case for case in CASES if "fake_output" in case]
PREFERENCES_PATH = Path(__file__).parents[1] / "evals" / "chastushki_preferences.json"
PREFERENCES = json.loads(PREFERENCES_PATH.read_text(encoding="utf-8"))
PREFERENCE_CASES_PATH = (
    Path(__file__).parents[1] / "evals" / "chastushki_preference_cases.json"
)
PREFERENCE_CASES = json.loads(PREFERENCE_CASES_PATH.read_text(encoding="utf-8"))
NATURALNESS_PATH = Path(__file__).parents[1] / "evals" / "chastushki_naturalness.json"
NATURALNESS_CASES = json.loads(NATURALNESS_PATH.read_text(encoding="utf-8"))
MODEL_CONFIGURATIONS_PATH = (
    Path(__file__).parents[1] / "evals" / "chastushki_model_configurations.json"
)
MODEL_CONFIGURATIONS = json.loads(MODEL_CONFIGURATIONS_PATH.read_text(encoding="utf-8"))


def test_chastushki_eval_set_has_varied_topics() -> None:
    assert len(CASES) >= 30
    assert len({case["brief"] for case in CASES}) == len(CASES)
    assert all(case["brief"] and case["expected_topic"] for case in CASES)


def test_chastushki_preferences_are_real_improvements() -> None:
    from misterio.chastushki import quality_issues

    assert len(PREFERENCES) >= 5
    rejected_issues = []
    for pair in PREFERENCES:
        rejected_issues.extend(quality_issues(pair["rejected"]))
        assert quality_issues(pair["chosen"]) == []
        assert pair["expected_topic"].casefold() in pair["chosen"].casefold()
        assert pair["reasons"]
    assert rejected_issues


def test_preference_cases_do_not_overlap_control_topics() -> None:
    control_briefs = {case["brief"] for case in CASES}
    control_topics = {case["expected_topic"] for case in CASES}
    preference_briefs = {case["brief"] for case in PREFERENCE_CASES}
    preference_topics = {case["expected_topic"] for case in PREFERENCE_CASES}

    assert len(PREFERENCE_CASES) == 50
    assert len(preference_briefs) == len(PREFERENCE_CASES)
    assert preference_briefs.isdisjoint(control_briefs)
    assert preference_topics.isdisjoint(control_topics)
    assert all(case["brief"] and case["expected_topic"] for case in PREFERENCE_CASES)


def test_naturalness_seed_has_distinct_confirmed_examples() -> None:
    labels = [case["natural"] for case in NATURALNESS_CASES]

    assert len(NATURALNESS_CASES) >= 15
    assert len({case["id"] for case in NATURALNESS_CASES}) == len(NATURALNESS_CASES)
    assert set(labels) == {True, False}
    assert all(case["text"] for case in NATURALNESS_CASES)
    assert all(case["review_status"] == "confirmed" for case in NATURALNESS_CASES)


def test_model_configurations_cover_registered_models_and_current_skill() -> None:
    manifest_path = (
        Path(__file__).parents[1] / "skills" / "chastushki" / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_parameters = {
        key: manifest[key]
        for key in ("temperature", "top_p", "max_tokens", "candidate_count")
    }

    assert {config["profile"] for config in MODEL_CONFIGURATIONS} == set(MODEL_OPTIONS)
    for config in MODEL_CONFIGURATIONS:
        assert config["model"] == MODEL_OPTIONS[config["profile"]].model
        assert config["parameters"] == expected_parameters
        assert config["status"] and config["rationale"]


def test_rhyme_score_prefers_chosen_preference_examples() -> None:
    from misterio.chastushki import rhyme_score

    preferred = sum(
        rhyme_score(pair["chosen"]) > rhyme_score(pair["rejected"])
        for pair in PREFERENCES
    )

    assert preferred / len(PREFERENCES) >= 0.75


def test_rhyme_score_components_on_preference_examples() -> None:
    from misterio.chastushki import rhyme_score

    configurations = {
        "tail": {"include_rhythm": False, "include_repetition": False},
        "tail_and_rhythm": {"include_rhythm": True, "include_repetition": False},
        "tail_and_repetition": {
            "include_rhythm": False,
            "include_repetition": True,
        },
        "all": {"include_rhythm": True, "include_repetition": True},
    }
    accuracy = {
        name: sum(
            rhyme_score(pair["chosen"], **options)
            > rhyme_score(pair["rejected"], **options)
            for pair in PREFERENCES
        )
        / len(PREFERENCES)
        for name, options in configurations.items()
    }

    assert accuracy["all"] >= 0.75
    assert accuracy["all"] >= accuracy["tail"]


@pytest.mark.parametrize(
    "case",
    FIXTURE_CASES,
    ids=[case["expected_topic"] for case in FIXTURE_CASES],
)
def test_chastushki_contract_fixture_preserves_topic(case: dict[str, str]) -> None:
    job = Job(
        id=f"eval-{case['expected_topic']}",
        status=JobStatus.queued,
        brief=case["brief"],
        skill_id="chastushki",
    )

    artifact = run(job, FakeLlm(text=case["fake_output"]))

    assert case["expected_topic"].casefold() in artifact.markdown.casefold()
    assert artifact.sources == []

    stanzas = artifact.markdown.split("\n\n")
    assert len(stanzas) == 2
    assert all(len(stanza.splitlines()) == 4 for stanza in stanzas)
    assert all(
        3 <= len(line.split()) <= 7
        for stanza in stanzas
        for line in stanza.splitlines()
    )
