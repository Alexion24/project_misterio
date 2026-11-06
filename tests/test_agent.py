import pytest

from misterio.agent import run
from misterio.contracts import Job, JobStatus, LlmCallSpec
from misterio.llm import FakeLlm


class SpyKnowledge:
    def search(self, query: str, brand_id: str | None, k: int) -> list:
        raise AssertionError("search should not be called")


class CapturingLlm:
    def __init__(self) -> None:
        self.spec: LlmCallSpec | None = None

    def complete(self, spec: LlmCallSpec) -> str:
        self.spec = spec
        return "fake-llm-output"


def test_run_uses_fake_llm_and_skips_search() -> None:
    job = Job(
        id="job-1",
        status=JobStatus.queued,
        brief="про трактор",
        skill_id="chastushki",
    )
    llm = CapturingLlm()
    artifact = run(job, llm, knowledge=SpyKnowledge())
    assert artifact.markdown == "fake-llm-output"
    assert artifact.sources == []
    assert job.skill_version == "0.3.0"
    assert llm.spec is not None
    assert llm.spec.temperature == 0.7


def test_run_requires_skill_id() -> None:
    job = Job(id="job-1", status=JobStatus.queued, brief="про трактор")
    with pytest.raises(ValueError):
        run(job, FakeLlm())
