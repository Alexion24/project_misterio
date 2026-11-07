import json
import os
from pathlib import Path
from time import perf_counter

import pytest

from misterio.agent import run
from misterio.chastushki import quality_issues
from misterio.contracts import Job, JobStatus
from misterio.llm import LlmError
from misterio.models import MODEL_OPTIONS, create_llm

CASES_PATH = Path(__file__).parents[1] / "evals" / "chastushki.json"
CASES = json.loads(CASES_PATH.read_text(encoding="utf-8"))


@pytest.mark.model_eval
@pytest.mark.skipif(
    os.getenv("MISTERIO_RUN_MODEL_EVALS") != "1",
    reason="set MISTERIO_RUN_MODEL_EVALS=1 to evaluate the configured model",
)
def test_real_model_chastushki_report(request: pytest.FixtureRequest) -> None:
    profile_key = request.config.getoption("--model-profile")
    option = MODEL_OPTIONS[profile_key]
    llm, model = create_llm(
        profile_key,
        timeout_s=float(os.getenv("MISTERIO_EVAL_TIMEOUT_S", "120")),
    )
    repetitions = int(os.getenv("MISTERIO_EVAL_REPETITIONS", "3"))
    min_pass_rate = float(os.getenv("MISTERIO_EVAL_MIN_PASS_RATE", "0.8"))
    results = []
    started = perf_counter()
    for case in CASES:
        for repetition in range(1, repetitions + 1):
            brief = case["brief"]
            case_started = perf_counter()
            try:
                artifact = run(
                    Job(
                        id=f"model-eval-{case['expected_topic']}-{repetition}",
                        status=JobStatus.queued,
                        brief=brief,
                        skill_id="chastushki",
                    ),
                    llm,
                    model=model,
                )
            except LlmError as error:
                results.append(
                    {
                        "brief": brief,
                        "repetition": repetition,
                        "passed": False,
                        "error": str(error),
                        "elapsed_s": round(perf_counter() - case_started, 3),
                    }
                )
                continue
            issues = quality_issues(artifact.markdown)
            topic_present = (
                case["expected_topic"].casefold() in artifact.markdown.casefold()
            )
            results.append(
                {
                    "brief": brief,
                    "repetition": repetition,
                    "passed": not issues and topic_present,
                    "quality_issues": issues,
                    "topic_present": topic_present,
                    "elapsed_s": round(perf_counter() - case_started, 3),
                    "artifact": artifact.model_dump(mode="json"),
                }
            )

    passed = sum(case["passed"] for case in results)
    pass_rate = passed / len(results)
    report = {
        "model": model,
        "api_style": option.api_style,
        "repetitions": repetitions,
        "min_pass_rate": min_pass_rate,
        "passed": passed,
        "total": len(results),
        "pass_rate": pass_rate,
        "elapsed_s": round(perf_counter() - started, 3),
        "cases": results,
    }
    report_dir = Path(os.getenv("MISTERIO_EVALS_DIR", "data/evals"))
    report_dir.mkdir(parents=True, exist_ok=True)
    filename = model.replace("/", "_").replace(":", "_") + ".json"
    (report_dir / filename).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    assert pass_rate >= min_pass_rate
