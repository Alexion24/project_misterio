import json
import os
import re
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
PREFERENCES_PATH = Path(__file__).parents[1] / "evals" / "chastushki_preferences.json"
PREFERENCES = json.loads(PREFERENCES_PATH.read_text(encoding="utf-8"))


def _case_slice(cases: list[dict[str, Any]], offset: int, limit: int) -> list:
    if offset < 0 or limit < 0:
        raise ValueError("case offset and limit must be non-negative")
    return cases[offset : offset + limit if limit else None]


def _exclude_prompt_examples(
    cases: list[dict[str, Any]], skill_md: str
) -> tuple[list[dict[str, Any]], list[str]]:
    example_briefs = set(re.findall(r"^Тема:\s*(.+)$", skill_md, re.MULTILINE))
    excluded = [case["brief"] for case in cases if case["brief"] in example_briefs]
    return [case for case in cases if case["brief"] not in example_briefs], excluded


def _select_few_shot_pairs(
    preferences: list[dict[str, Any]],
    excluded_topics: set[str],
    count: int,
    topics: list[str],
) -> list[dict[str, Any]]:
    eligible = [
        pair
        for pair in preferences
        if pair["expected_topic"] not in excluded_topics
        and not pair.get("human_review_notes")
    ]
    if topics:
        if count:
            raise ValueError("set either few-shot count or explicit topics")
        by_topic = {pair["expected_topic"]: pair for pair in eligible}
        if len(topics) != len(set(topics)):
            raise ValueError("few-shot topics must be unique")
        unavailable = [topic for topic in topics if topic not in by_topic]
        if unavailable:
            raise ValueError(f"unavailable few-shot topics: {', '.join(unavailable)}")
        return [by_topic[topic] for topic in topics]
    if not 0 <= count <= len(eligible):
        raise ValueError(f"requested {count} few-shot pairs, found {len(eligible)}")
    return eligible[:count]


def _few_shot_addendum(pairs: list[dict[str, Any]]) -> str:
    examples = "\n\n".join(
        f"Плохой вариант:\n{pair['rejected']}\n\n"
        f"Исправленный вариант:\n{pair['chosen']}"
        for pair in pairs
    )
    return f"\n\n## Контрастные примеры\n\n{examples}" if examples else ""


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
    cases_path = Path(os.getenv("MISTERIO_EVAL_CASES_PATH", str(CASES_PATH)))
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    manifest, skill_folder = load_skill("chastushki")
    skill_md = (skill_folder / "SKILL.md").read_text(encoding="utf-8")
    cases, excluded_prompt_examples = _exclude_prompt_examples(cases, skill_md)
    case_offset = int(os.getenv("MISTERIO_EVAL_CASE_OFFSET", "0"))
    case_limit = int(os.getenv("MISTERIO_EVAL_CASE_LIMIT", "0"))
    cases = _case_slice(cases, case_offset, case_limit)
    candidate_count = int(
        os.getenv("MISTERIO_EVAL_CANDIDATE_COUNT", str(manifest.candidate_count))
    )
    temperature = float(
        os.getenv("MISTERIO_EVAL_TEMPERATURE", str(manifest.temperature))
    )
    top_p = float(os.getenv("MISTERIO_EVAL_TOP_P", str(manifest.top_p)))
    raw_topics = os.getenv("MISTERIO_EVAL_FEW_SHOT_TOPICS", "")
    topics = [topic.strip() for topic in raw_topics.split(",")] if raw_topics else []
    few_shot_pairs = _select_few_shot_pairs(
        PREFERENCES,
        {case["expected_topic"] for case in cases},
        int(os.getenv("MISTERIO_EVAL_FEW_SHOT_COUNT", "0")),
        topics,
    )
    prompt_addendum = _few_shot_addendum(few_shot_pairs)
    few_shot_count = len(few_shot_pairs)
    parameters = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": manifest.max_tokens,
        "candidate_count": candidate_count,
        "few_shot_count": few_shot_count,
        "few_shot_topics": [pair["expected_topic"] for pair in few_shot_pairs],
    }
    results = []
    started = perf_counter()
    for case in cases:
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
                    candidate_count=candidate_count,
                    temperature=temperature,
                    top_p=top_p,
                    prompt_addendum=prompt_addendum,
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
        "cases_path": str(cases_path),
        "excluded_prompt_examples": excluded_prompt_examples,
        "case_offset": case_offset,
        "case_limit": case_limit or None,
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
    if candidate_count != manifest.candidate_count:
        filename = (
            filename.removesuffix(".json") + f"-candidates-{candidate_count}.json"
        )
    if temperature != manifest.temperature or top_p != manifest.top_p:
        filename = filename.removesuffix(".json") + (
            f"-temperature-{temperature:g}-top-p-{top_p:g}.json"
        )
    if few_shot_count:
        filename = filename.removesuffix(".json") + f"-few-shot-{few_shot_count}.json"
    if case_offset:
        filename = filename.removesuffix(".json") + f"-offset-{case_offset}.json"
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


def test_few_shot_addendum_excludes_eval_topics() -> None:
    preferences = [
        {"expected_topic": "кот", "rejected": "bad cat", "chosen": "good cat"},
        {"expected_topic": "пёс", "rejected": "bad dog", "chosen": "good dog"},
    ]

    addendum = _few_shot_addendum(_select_few_shot_pairs(preferences, {"кот"}, 1, []))

    assert "good dog" in addendum
    assert "good cat" not in addendum


def test_few_shot_pairs_follow_explicit_order() -> None:
    topics = ["метел", "колодец", "дрон", "будильник"]

    pairs = _select_few_shot_pairs(PREFERENCES, set(), 0, topics)

    assert [pair["expected_topic"] for pair in pairs] == topics
    with pytest.raises(ValueError, match="unavailable few-shot topics: колодец"):
        _select_few_shot_pairs(PREFERENCES, {"колодец"}, 0, topics)


def test_case_slice_selects_non_overlapping_block() -> None:
    cases = [{"id": index} for index in range(10)]

    assert _case_slice(cases, offset=5, limit=3) == [
        {"id": 5},
        {"id": 6},
        {"id": 7},
    ]


def test_prompt_examples_are_excluded_from_eval_cases() -> None:
    cases = [
        {"brief": "про трактор"},
        {"brief": "про рыбака"},
    ]

    filtered, excluded = _exclude_prompt_examples(cases, "Тема: про трактор")

    assert filtered == [{"brief": "про рыбака"}]
    assert excluded == ["про трактор"]
