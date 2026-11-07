import re
from pathlib import Path
from typing import Protocol

from misterio.chastushki import quality_issues, rhyme_score
from misterio.contracts import DocumentChunk, GenerationArtifact, Job, LlmCallSpec
from misterio.llm import Llm
from misterio.skills import load_skill


class Knowledge(Protocol):
    def search(
        self, query: str, brand_id: str | None, k: int
    ) -> list[DocumentChunk]: ...


def _candidate_number(answer: str, count: int) -> int | None:
    match = re.search(r"\b([1-6])\b", answer)
    if match and int(match.group(1)) <= count:
        return int(match.group(1)) - 1
    return None


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
    prompt = f"{skill_md}\n\nТема пользователя: {job.brief}"
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
        numbered = "\n\n".join(
            f"Кандидат {index + 1} (рифма {rhyme_scores[index]:.2f}):\n"
            f"{candidates[index]}"
            for index in eligible
        )
        selection = llm.complete(
            LlmCallSpec(
                model=model,
                prompt=(
                    f"Тема пользователя: {job.brief}\n\n"
                    "Выбери лучший черновик по смыслу, естественности, ритму, "
                    "рифме и юмору. Учитывай указанную автоматическую оценку "
                    "рифмы. Ответь только номером кандидата.\n\n"
                    f"{numbered}"
                ),
                temperature=0.0,
                top_p=manifest.top_p,
                max_tokens=10,
            )
        )
        selected_index = _candidate_number(selection, len(candidates))
        if selected_index not in eligible:
            selected_index = max(
                eligible,
                key=lambda index: (-len(issues[index]), rhyme_scores[index]),
            )
        selected = candidates[selected_index]
        text = llm.complete(
            LlmCallSpec(
                model=model,
                prompt=(
                    f"{skill_md}\n\nТема пользователя: {job.brief}\n\n"
                    "Отредактируй только этот черновик. Исправь смысл, ритм, "
                    "рифму и неестественные фразы. Не добавляй новые строфы. "
                    "Верни только итоговые частушки.\n\n"
                    f"Черновик:\n{selected}"
                ),
                temperature=0.2,
                top_p=manifest.top_p,
                max_tokens=manifest.max_tokens,
            )
        )
        if not uses_quality:
            return GenerationArtifact(markdown=text, sources=sources)
        final_issues = quality_issues(text)
        if final_issues and not quality_issues(selected):
            text = selected
        elif final_issues:
            repaired = llm.complete(
                LlmCallSpec(
                    model=model,
                    prompt=(
                        f"{skill_md}\n\nТема пользователя: {job.brief}\n\n"
                        "Исправь только перечисленные ошибки: "
                        f"{'; '.join(final_issues)}. "
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
        elif rhyme_score(text) < rhyme_score(selected):
            text = selected
    return GenerationArtifact(markdown=text, sources=sources)
