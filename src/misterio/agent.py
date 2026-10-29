from pathlib import Path
from typing import Protocol

from misterio.contracts import DocumentChunk, GenerationArtifact, Job, LlmCallSpec
from misterio.llm import Llm
from misterio.skills import load_skill


class Knowledge(Protocol):
    def search(
        self, query: str, brand_id: str | None, k: int
    ) -> list[DocumentChunk]: ...


def run(
    job: Job,
    llm: Llm,
    *,
    knowledge: Knowledge | None = None,
    skills_dir: Path | None = None,
    model: str = "fake",
) -> GenerationArtifact:
    if not job.skill_id:
        raise ValueError("job.skill_id is required")
    manifest, folder = load_skill(job.skill_id, skills_dir)
    job.skill_version = manifest.version

    sources: list[DocumentChunk] = []
    if manifest.needs_retrieval:
        if knowledge is None:
            raise ValueError("skill requires retrieval but no knowledge was provided")
        sources = knowledge.search(job.brief, job.brand_id, k=5)

    skill_md = (folder / "SKILL.md").read_text(encoding="utf-8")
    prompt = f"{skill_md}\n\n{job.brief}"
    text = llm.complete(LlmCallSpec(model=model, prompt=prompt))
    return GenerationArtifact(markdown=text, sources=sources)
