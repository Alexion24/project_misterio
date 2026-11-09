import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from misterio.contracts import LlmCallSpec
from misterio.llm import FakeLlm
from misterio.models import create_llm

CASES_PATH = Path(__file__).parents[1] / "evals" / "chastushki_naturalness.json"
CASES = json.loads(CASES_PATH.read_text(encoding="utf-8"))
FEW_SHOT_IDS = {
    "natural-tractor",
    "natural-sandy-corner",
    "unnatural-fishing-net",
    "unnatural-seasoning-mouse",
}


def _naturalness_prompt(text: str) -> str:
    return f"""Оцени только естественность русского языка в частушке.
Считай текст неестественным, если в нём есть грамматически неверные сочетания,
случайные слова или фразы без понятного смысла. Не оценивай качество рифмы,
юмор и соответствие жанру.

Ответь ровно одним словом: NATURAL или UNNATURAL.

Текст:
{text}"""


def _few_shot_naturalness_prompt(text: str) -> str:
    examples = [case for case in CASES if case["id"] in FEW_SHOT_IDS]
    rendered = "\n\n".join(
        f"Текст:\n{case['text']}\nОтвет: "
        f"{'NATURAL' if case['natural'] else 'UNNATURAL'}"
        for case in examples
    )
    return f"""Оцени естественность русского языка по примерам ниже.
Неестественны грамматически неверные сочетания, случайные слова и фразы без
понятного смысла. Не оценивай качество рифмы и юмор.

{rendered}

Текст:
{text}
Ответ:"""


def _parse_naturalness(response: str) -> bool:
    label = response.strip().upper()
    if not re.fullmatch(r"(?:UN)?NATURAL", label):
        raise ValueError(f"unexpected naturalness label: {response!r}")
    return label == "NATURAL"


def _accuracy(results: list[dict[str, Any]]) -> float | None:
    if not results:
        return None
    return sum(result["correct"] for result in results) / len(results)


def _evaluate_cases(
    llm: Any,
    model: str,
    cases: list[dict[str, Any]],
    prompt: Callable[[str], str],
) -> list[dict[str, Any]]:
    results = []
    for case in cases:
        response = llm.complete(
            LlmCallSpec(
                model=model,
                prompt=prompt(case["text"]),
                temperature=0.0,
                max_tokens=10,
            )
        )
        try:
            predicted = _parse_naturalness(response)
        except ValueError as error:
            results.append(
                {
                    "id": case["id"],
                    "review_status": case["review_status"],
                    "expected": case["natural"],
                    "response": response,
                    "error": str(error),
                    "correct": False,
                }
            )
            continue
        results.append(
            {
                "id": case["id"],
                "review_status": case["review_status"],
                "expected": case["natural"],
                "predicted": predicted,
                "correct": predicted == case["natural"],
            }
        )
    return results


@pytest.mark.model_eval
@pytest.mark.skipif(
    os.getenv("MISTERIO_RUN_NATURALNESS_EVAL") != "1",
    reason="set MISTERIO_RUN_NATURALNESS_EVAL=1 to run the naturalness judge",
)
def test_real_model_naturalness_judge(request: pytest.FixtureRequest) -> None:
    profile_key = request.config.getoption("--model-profile")
    llm, model = create_llm(
        profile_key,
        timeout_s=float(os.getenv("MISTERIO_EVAL_TIMEOUT_S", "120")),
    )
    results = _evaluate_cases(llm, model, CASES, _naturalness_prompt)

    confirmed = [r for r in results if r["review_status"] == "confirmed"]
    provisional = [r for r in results if r["review_status"] == "provisional"]
    report = {
        "model": model,
        "accuracy": _accuracy(results),
        "confirmed_accuracy": _accuracy(confirmed),
        "provisional_accuracy": _accuracy(provisional),
        "cases": results,
    }
    report_dir = Path(os.getenv("MISTERIO_EVALS_DIR", "data/evals"))
    report_dir.mkdir(parents=True, exist_ok=True)
    filename = model.replace("/", "_").replace(":", "_")
    (report_dir / f"{filename}-naturalness.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    assert report["accuracy"] is not None
    assert report["accuracy"] >= float(
        os.getenv("MISTERIO_NATURALNESS_MIN_ACCURACY", "0.8")
    )


@pytest.mark.model_eval
@pytest.mark.skipif(
    os.getenv("MISTERIO_RUN_NATURALNESS_EVAL") != "1",
    reason="set MISTERIO_RUN_NATURALNESS_EVAL=1 to run the naturalness judge",
)
def test_real_model_few_shot_naturalness_judge(
    request: pytest.FixtureRequest,
) -> None:
    profile_key = request.config.getoption("--model-profile")
    llm, model = create_llm(
        profile_key,
        timeout_s=float(os.getenv("MISTERIO_EVAL_TIMEOUT_S", "120")),
    )
    holdout = [case for case in CASES if case["id"] not in FEW_SHOT_IDS]
    results = _evaluate_cases(llm, model, holdout, _few_shot_naturalness_prompt)
    accuracy = _accuracy(results)
    report = {
        "model": model,
        "few_shot_ids": sorted(FEW_SHOT_IDS),
        "accuracy": accuracy,
        "cases": results,
    }
    report_dir = Path(os.getenv("MISTERIO_EVALS_DIR", "data/evals"))
    report_dir.mkdir(parents=True, exist_ok=True)
    filename = model.replace("/", "_").replace(":", "_")
    (report_dir / f"{filename}-naturalness-few-shot.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    assert accuracy is not None
    assert accuracy >= float(os.getenv("MISTERIO_NATURALNESS_MIN_ACCURACY", "0.8"))


@pytest.mark.parametrize(
    ("response", "expected"),
    [("NATURAL", True), ("UNNATURAL\n", False), ("natural", True)],
)
def test_parse_naturalness(response: str, expected: bool) -> None:
    assert _parse_naturalness(response) is expected


def test_parse_naturalness_rejects_explanations() -> None:
    with pytest.raises(ValueError, match="unexpected naturalness label"):
        _parse_naturalness("NATURAL: хороший текст")


def test_accuracy_handles_empty_results() -> None:
    assert _accuracy([]) is None


def test_evaluate_cases_records_protocol_error() -> None:
    results = _evaluate_cases(FakeLlm("explanation"), "fake", CASES[:1], str)

    assert results[0]["correct"] is False
    assert "unexpected naturalness label" in results[0]["error"]


def test_few_shot_prompt_excludes_holdout_text() -> None:
    holdout = next(case for case in CASES if case["id"] not in FEW_SHOT_IDS)

    prompt = _few_shot_naturalness_prompt(holdout["text"])

    assert prompt.count(holdout["text"]) == 1
