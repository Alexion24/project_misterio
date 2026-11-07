from enum import StrEnum
from typing import Any, Literal

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
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    max_tokens: int = Field(default=500, ge=1)
    candidate_count: int = Field(default=1, ge=1, le=6)
    quality_policy: Literal["none", "chastushki"] = "none"


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
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    max_tokens: int = Field(default=500, ge=1)
