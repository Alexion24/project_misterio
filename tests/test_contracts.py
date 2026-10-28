import pytest
from pydantic import ValidationError

from misterio.contracts import (
    DocumentChunk,
    GenerationArtifact,
    Job,
    LlmCallSpec,
    SkillManifest,
)


def test_models_from_valid_dicts() -> None:
    job = Job.model_validate(
        {
            "id": "job-1",
            "status": "queued",
            "brief": "про трактор",
            "skill_id": "chastushki",
            "skill_version": "0.1.0",
            "brand_id": "local",
        }
    )
    assert job.brief == "про трактор"

    manifest = SkillManifest.model_validate(
        {
            "id": "chastushki",
            "version": "0.1.0",
            "input_schema": {"type": "object"},
            "output_schema": {"type": "object"},
        }
    )
    assert manifest.needs_retrieval is False

    chunk = DocumentChunk.model_validate(
        {
            "text": "seed",
            "source": "kb/seed/a.md",
            "locator": "kb/seed/a.md#0",
            "score": 0.9,
        }
    )
    artifact = GenerationArtifact.model_validate(
        {"markdown": "ok", "sources": [chunk.model_dump()]}
    )
    spec = LlmCallSpec.model_validate({"model": "fake", "prompt": "hi"})
    assert artifact.markdown == "ok"
    assert spec.temperature == 0.0


def test_job_rejects_missing_brief() -> None:
    with pytest.raises(ValidationError):
        Job.model_validate({"id": "job-1", "status": "queued"})


def test_document_chunk_rejects_non_numeric_score() -> None:
    with pytest.raises(ValidationError):
        DocumentChunk.model_validate(
            {
                "text": "seed",
                "source": "kb/seed/a.md",
                "locator": "kb/seed/a.md#0",
                "score": "high",
            }
        )
