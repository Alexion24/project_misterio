import pytest

from misterio.agent import run
from misterio.contracts import DocumentChunk, Job, JobStatus, LlmCallSpec
from misterio.llm import FakeLlm


class SpyKnowledge:
    def search(self, query: str, brand_id: str | None, k: int) -> list:
        raise AssertionError("search should not be called")


class CapturingLlm:
    def __init__(self) -> None:
        self.specs: list[LlmCallSpec] = []

    def complete(self, spec: LlmCallSpec) -> str:
        self.specs.append(spec)
        return "fake-llm-output"


class SequenceLlm:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.specs: list[LlmCallSpec] = []

    def complete(self, spec: LlmCallSpec) -> str:
        self.specs.append(spec)
        return next(self.responses)


VALID_CHASTUSHKI = """По деревне трактор мчится,
Пыль клубится за трубой.
Петя думал прокатиться —
Еле выбрался домой.

Трактор едет по дороге,
Пыль летит над бугром.
Куры спрятались в тревоге —
Кот командует двором!"""


def test_run_uses_fake_llm_and_skips_search() -> None:
    job = Job(
        id="job-1",
        status=JobStatus.queued,
        brief="про трактор",
        skill_id="chastushki",
    )
    llm = CapturingLlm()
    artifact = run(job, llm, knowledge=SpyKnowledge(), model="qwen3:8b")
    assert artifact.markdown == "fake-llm-output"
    assert artifact.sources == []
    assert job.skill_version == "0.6.0"
    assert len(llm.specs) == 6
    assert all(spec.temperature == 0.8 for spec in llm.specs[:3])
    assert llm.specs[3].temperature == 0.0
    assert llm.specs[3].max_tokens == 10
    assert llm.specs[4].temperature == 0.2
    assert llm.specs[-1].top_p == 0.8
    assert "Кандидат 3" in llm.specs[3].prompt


def test_run_edits_only_selected_candidate_and_rejects_invalid_edit() -> None:
    job = Job(
        id="job-2",
        status=JobStatus.queued,
        brief="про трактор",
        skill_id="chastushki",
    )
    llm = SequenceLlm(
        ["плохой первый", VALID_CHASTUSHKI, "плохой третий", "2", "сломано"]
    )

    artifact = run(job, llm)

    assert artifact.markdown == VALID_CHASTUSHKI
    assert "плохой первый" not in llm.specs[4].prompt
    assert VALID_CHASTUSHKI in llm.specs[4].prompt


def test_run_requires_skill_id() -> None:
    job = Job(id="job-1", status=JobStatus.queued, brief="про трактор")
    with pytest.raises(ValueError):
        run(job, FakeLlm())


def test_retrieval_context_is_included_and_diagnostics_are_recorded(
    tmp_path,
) -> None:
    skill_dir = tmp_path / "retrieval"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("Используй контекст", encoding="utf-8")
    (skill_dir / "manifest.json").write_text(
        '{"id":"retrieval","version":"1","input_schema":{},'
        '"output_schema":{},"needs_retrieval":true}',
        encoding="utf-8",
    )

    class Knowledge:
        def search(self, query: str, brand_id: str | None, k: int) -> list:
            return [
                DocumentChunk(
                    text="Проверенный факт",
                    source="facts.md",
                    locator="facts.md#0",
                    score=1,
                )
            ]

    llm = CapturingLlm()
    artifact = run(
        Job(
            id="retrieval", status=JobStatus.queued, brief="тема", skill_id="retrieval"
        ),
        llm,
        knowledge=Knowledge(),
        skills_dir=tmp_path,
    )

    assert "Проверенный факт" in llm.specs[0].prompt
    assert "facts.md#0" in llm.specs[0].prompt
    assert artifact.sources[0].source == "facts.md"
    assert artifact.diagnostics.candidates[0].text == "fake-llm-output"
