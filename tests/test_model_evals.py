import os

import pytest

from misterio.agent import run
from misterio.chastushki import quality_issues
from misterio.contracts import Job, JobStatus
from misterio.llm import OpenAiCompatibleLlm, Settings


@pytest.mark.model_eval
@pytest.mark.skipif(
    os.getenv("MISTERIO_RUN_MODEL_EVALS") != "1",
    reason="set MISTERIO_RUN_MODEL_EVALS=1 to evaluate the configured model",
)
@pytest.mark.parametrize("brief", ["про трактор", "про самовар", "про программиста"])
def test_real_model_generates_valid_chastushki(brief: str) -> None:
    settings = Settings()
    artifact = run(
        Job(
            id=f"model-eval-{brief}",
            status=JobStatus.queued,
            brief=brief,
            skill_id="chastushki",
        ),
        OpenAiCompatibleLlm(settings),
        model=settings.llm_model,
    )

    assert not quality_issues(artifact.markdown)
    assert brief.removeprefix("про ").casefold() in artifact.markdown.casefold()
