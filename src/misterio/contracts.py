from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(StrEnum):
    queued = "queued"
    running = "running"
    ready = "ready"
    failed = "failed"


class Job(BaseModel):
    id: str
    status: JobStatus
    brief: str
    skill_id: str | None = None
    skill_version: str | None = None
    brand_id: str | None = None


class SkillManifest(BaseModel):
    id: str
    version: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    needs_retrieval: bool = False


class DocumentChunk(BaseModel):
    text: str
    source: str
    locator: str
    score: float


class GenerationArtifact(BaseModel):
    markdown: str
    sources: list[DocumentChunk] = Field(default_factory=list)


class LlmCallSpec(BaseModel):
    model: str
    prompt: str
    temperature: float = 0.0
