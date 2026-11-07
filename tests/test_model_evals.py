import json
import os
from pathlib import Path
from time import perf_counter

import pytest

from misterio.agent import run
from misterio.chastushki import quality_issues
from misterio.contracts import Job, JobStatus
from misterio.llm import LlmError, OpenAiCompatibleLlm, Settings


@pytest.mark.model_eval
@pytest.mark.skipif(
    os.getenv("MISTERIO_RUN_MODEL_EVALS") != "1",
    reason="set MISTERIO_RUN_MODEL_EVALS=1 to evaluate the configured model",
)
def test_real_model_chastushki_report() -> None:
    settings = Settings()
    settings.llm_timeout_s = float(os.getenv("MISTERIO_EVAL_TIMEOUT_S", "120"))
    llm = OpenAiCompatibleLlm(settings)
    results = []
    started = perf_counter()
    for brief in ["про трактор", "про самовар", "про программиста"]:
        case_started = perf_counter()
        try:
            artifact = run(
                Job(
                    id=f"model-eval-{brief}",
                    status=JobStatus.queued,
                    brief=brief,
                    skill_id="chastushki",
                ),
                llm,
                model=settings.llm_model,
            )
        except LlmError as error:
            results.append(
                {
                    "brief": brief,
                    "passed": False,
                    "error": str(error),
                    "elapsed_s": round(perf_counter() - case_started, 3),
                }
            )
            continue
        issues = quality_issues(artifact.markdown)
        topic_present = (
            brief.removeprefix("про ").casefold() in artifact.markdown.casefold()
        )
        results.append(
            {
                "brief": brief,
                "passed": not issues and topic_present,
                "quality_issues": issues,
                "topic_present": topic_present,
                "elapsed_s": round(perf_counter() - case_started, 3),
                "artifact": artifact.model_dump(mode="json"),
            }
        )

    passed = sum(case["passed"] for case in results)
    report = {
        "model": settings.llm_model,
        "api_style": settings.llm_api_style,
        "passed": passed,
        "total": len(results),
        "pass_rate": passed / len(results),
        "elapsed_s": round(perf_counter() - started, 3),
        "cases": results,
    }
    report_dir = Path(os.getenv("MISTERIO_EVALS_DIR", "data/evals"))
    report_dir.mkdir(parents=True, exist_ok=True)
    filename = settings.llm_model.replace("/", "_").replace(":", "_") + ".json"
    (report_dir / filename).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    assert len(results) == 3
