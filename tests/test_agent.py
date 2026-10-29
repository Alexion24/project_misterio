import pytest

from misterio.agent import run
from misterio.contracts import Job, JobStatus
from misterio.llm import FakeLlm


class SpyKnowledge:
    def search(self, query: str, brand_id: str | None, k: int) -> list:
        raise AssertionError("search should not be called")


def test_run_uses_fake_llm_and_skips_search() -> None:
    job = Job(
        id="job-1",
        status=JobStatus.queued,
        brief="про трактор",
        skill_id="chastushki",
    )
    artifact = run(job, FakeLlm(text="fake-llm-output"), knowledge=SpyKnowledge())
    assert artifact.markdown == "fake-llm-output"
    assert artifact.sources == []
    assert job.skill_version == "0.1.0"


def test_run_requires_skill_id() -> None:
    job = Job(id="job-1", status=JobStatus.queued, brief="про трактор")
    with pytest.raises(ValueError):
        run(job, FakeLlm())
