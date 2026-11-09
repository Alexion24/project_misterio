import json
from pathlib import Path
from runpy import run_path

import pytest

build_review = run_path(
    str(Path(__file__).parents[1] / "scripts" / "prepare_blind_model_review.py")
)["build_review"]
render_markdown = run_path(
    str(Path(__file__).parents[1] / "scripts" / "prepare_blind_model_review.py")
)["render_markdown"]
render_html = run_path(
    str(Path(__file__).parents[1] / "scripts" / "prepare_blind_model_review.py")
)["render_html"]


def _write_report(path: Path, model: str, briefs: list[str]) -> None:
    cases = [
        {
            "brief": brief,
            "repetition": 1,
            "artifact": {"markdown": brief},
        }
        for brief in briefs
    ]
    path.write_text(
        json.dumps({"model": model, "cases": cases}, ensure_ascii=False),
        encoding="utf-8",
    )


def test_build_review_separates_blind_form_and_key(tmp_path: Path) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    _write_report(left, "left-model", ["про кота", "про пса"])
    _write_report(right, "right-model", ["про кота", "про пса"])

    review, key = build_review(left, right, seed=7)

    assert len(review) == 2
    assert all("a_model" not in case and "b_model" not in case for case in review)
    assert set(key["answers"]) == {"case-01", "case-02"}


def test_build_review_rejects_mismatched_cases(tmp_path: Path) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    _write_report(left, "left-model", ["про кота"])
    _write_report(right, "right-model", ["про пса"])

    with pytest.raises(ValueError, match="same successful cases"):
        build_review(left, right, seed=7)


def test_build_review_uses_explicit_condition_labels(tmp_path: Path) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    _write_report(left, "same-model", ["про кота"])
    _write_report(right, "same-model", ["про кота"])

    _, key = build_review(
        left,
        right,
        seed=7,
        left_label="two candidates",
        right_label="four candidates",
    )

    labels = set(key["answers"]["case-01"].values())
    assert labels == {"two candidates", "four candidates"}


def test_render_markdown_shows_text_and_review_controls(tmp_path: Path) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    _write_report(left, "left-model", ["про кота"])
    _write_report(right, "right-model", ["про кота"])
    review, _ = build_review(left, right, seed=7)

    markdown = render_markdown(review)

    assert "## case-01 — про кота" in markdown
    assert "### Вариант A" in markdown
    assert "[ ] A  [ ] B  [ ] Ничья  [ ] Оба плохие" in markdown
    assert "**Заметка:**" in markdown


def test_render_html_embeds_review_and_interactive_controls(tmp_path: Path) -> None:
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    _write_report(left, "left-model", ["про кота"])
    _write_report(right, "right-model", ["про кота"])
    review, _ = build_review(left, right, seed=7)

    html = render_html(review)

    assert "про кота" in html
    assert 'type="radio"' in html
    assert "localStorage" in html
    assert "Скачать ответы JSON" in html
