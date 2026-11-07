import json
from pathlib import Path

import pytest

from misterio.agent import run
from misterio.contracts import Job, JobStatus
from misterio.llm import FakeLlm

CASES_PATH = Path(__file__).parents[1] / "evals" / "chastushki.json"
CASES = json.loads(CASES_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES, ids=[case["expected_topic"] for case in CASES])
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
