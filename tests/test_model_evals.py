import json
import os
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import pytest

from misterio.agent import run
from misterio.chastushki import quality_issues
from misterio.contracts import Job, JobStatus
from misterio.llm import LlmError
from misterio.models import MODEL_OPTIONS, create_llm
from misterio.skills import load_skill

CASES_PATH = Path(__file__).parents[1] / "evals" / "chastushki.json"
CASES = json.loads(CASES_PATH.read_text(encoding="utf-8"))


def _summary(results: list[dict[str, Any]]) -> dict[str, int | float]:
    passed = sum(result["passed"] for result in results)
    total = len(results)
    return {
        "passed": passed,
        "total": total,
        "pass_rate": passed / total if total else 0.0,
        "mean_elapsed_s": (
            round(sum(result["elapsed_s"] for result in results) / total, 3)
            if total
            else 0.0
        ),
    }


def _aggregates(
    results: list[dict[str, Any]], model: str, parameters: dict[str, Any]
) -> dict[str, Any]:
    by_topic: dict[str, list[dict[str, Any]]] = {}
    error_types: Counter[str] = Counter()
    for result in results:
        by_topic.setdefault(result["brief"], []).append(result)
        if result.get("error"):
            error_types["request_error"] += 1
        if result.get("topic_present") is False:
            error_types["topic_missing"] += 1
        error_types.update(result.get("quality_issues", []))

    summary = _summary(results)
    return {
        "by_model": {model: summary},
        "by_parameters": [{"parameters": parameters, **summary}],
        "by_topic": {
            topic: _summary(topic_results) for topic, topic_results in by_topic.items()
        },
        "by_error_type": dict(sorted(error_types.items())),
    }


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
    manifest, _ = load_skill("chastushki")
    parameters = {
        "temperature": manifest.temperature,
        "top_p": manifest.top_p,
        "max_tokens": manifest.max_tokens,
        "candidate_count": manifest.candidate_count,
    }
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
        "aggregates": _aggregates(results, model, parameters),
        "cases": results,
    }
    report_dir = Path(os.getenv("MISTERIO_EVALS_DIR", "data/evals"))
    report_dir.mkdir(parents=True, exist_ok=True)
    filename = model.replace("/", "_").replace(":", "_") + ".json"
    (report_dir / filename).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    assert pass_rate >= min_pass_rate


def test_eval_aggregates_group_results_and_error_types() -> None:
    results = [
        {
            "brief": "про трактор",
            "passed": True,
            "quality_issues": [],
            "topic_present": True,
            "elapsed_s": 1.0,
        },
        {
            "brief": "про трактор",
            "passed": False,
            "quality_issues": ["weak rhyme"],
            "topic_present": False,
            "elapsed_s": 3.0,
        },
        {
            "brief": "про кота",
            "passed": False,
            "error": "timeout",
            "elapsed_s": 2.0,
        },
    ]
    parameters = {"temperature": 0.8, "candidate_count": 2}

    aggregates = _aggregates(results, "cloud-model", parameters)

    assert aggregates["by_model"]["cloud-model"] == {
        "passed": 1,
        "total": 3,
        "pass_rate": 1 / 3,
        "mean_elapsed_s": 2.0,
    }
    assert aggregates["by_parameters"] == [
        {
            "parameters": parameters,
            "passed": 1,
            "total": 3,
            "pass_rate": 1 / 3,
            "mean_elapsed_s": 2.0,
        }
    ]
    assert aggregates["by_topic"]["про трактор"] == {
        "passed": 1,
        "total": 2,
        "pass_rate": 0.5,
        "mean_elapsed_s": 2.0,
    }
    assert aggregates["by_error_type"] == {
        "request_error": 1,
        "topic_missing": 1,
        "weak rhyme": 1,
    }


def test_eval_summary_handles_no_results() -> None:
    assert _summary([]) == {
        "passed": 0,
        "total": 0,
        "pass_rate": 0.0,
        "mean_elapsed_s": 0.0,
    }
