from pathlib import Path
from typing import Protocol

from misterio.chastushki import quality_issues, rhyme_score
from misterio.contracts import (
    CandidateDiagnostic,
    DocumentChunk,
    GenerationArtifact,
    GenerationDiagnostics,
    Job,
    LlmCallSpec,
)
from misterio.llm import Llm
from misterio.skills import load_skill


class Knowledge(Protocol):
    def search(
        self, query: str, brand_id: str | None, k: int
    ) -> list[DocumentChunk]: ...


def _source_context(sources: list[DocumentChunk]) -> str:
    if not sources:
        return ""
    chunks = "\n\n".join(
        f"Источник: {chunk.source} ({chunk.locator})\n{chunk.text}" for chunk in sources
    )
    return f"\n\nКонтекст из базы знаний:\n{chunks}"


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
    context = _source_context(sources)
    prompt = f"{skill_md}\n\nТема пользователя: {job.brief}{context}"

    candidates = [
        llm.complete(
            LlmCallSpec(
                model=model,
                prompt=prompt,
                temperature=manifest.temperature,
                top_p=manifest.top_p,
                max_tokens=manifest.max_tokens,
            )
        )
        for _ in range(manifest.candidate_count)
    ]
    diagnostics = GenerationDiagnostics()
    text = candidates[0]
    if len(candidates) > 1:
        uses_quality = manifest.quality_policy == "chastushki"
        issues = [
            quality_issues(candidate) if uses_quality else []
            for candidate in candidates
        ]
        rhyme_scores = [
            rhyme_score(candidate) if uses_quality else 0.0 for candidate in candidates
        ]
        eligible = [
            index
            for index, candidate_issues in enumerate(issues)
            if not candidate_issues
        ]
        eligible = eligible or list(range(len(candidates)))
        diagnostics.candidates = [
            CandidateDiagnostic(
                text=candidate,
                quality_issues=issues[index],
                rhyme_score=rhyme_scores[index],
                eligible=index in eligible,
            )
            for index, candidate in enumerate(candidates)
        ]
        selected_index = max(
            eligible,
            key=lambda index: (-len(issues[index]), rhyme_scores[index]),
        )
        diagnostics.selection_reason = "local quality and rhyme score"
        diagnostics.selected_index = selected_index
        selected = candidates[selected_index]
        text = selected
        if not uses_quality:
            return GenerationArtifact(
                markdown=text, sources=sources, diagnostics=diagnostics
            )
        if issues[selected_index]:
            repaired = llm.complete(
                LlmCallSpec(
                    model=model,
                    prompt=(
                        f"{skill_md}\n\nТема пользователя: {job.brief}{context}\n\n"
                        "Исправь только перечисленные ошибки: "
                        f"{'; '.join(issues[selected_index])}. "
                        "Верни только итоговые частушки.\n\n"
                        f"Текст:\n{text}"
                    ),
                    temperature=0.2,
                    top_p=manifest.top_p,
                    max_tokens=manifest.max_tokens,
                )
            )
            variants = [selected, text, repaired]
            text = max(
                variants,
                key=lambda variant: (
                    -len(quality_issues(variant)),
                    rhyme_score(variant),
                ),
            )
    if not diagnostics.candidates:
        diagnostics.candidates = [CandidateDiagnostic(text=text)]
    return GenerationArtifact(markdown=text, sources=sources, diagnostics=diagnostics)
