from pathlib import Path

import pytest
from typer.testing import CliRunner

import misterio.cli as cli
from misterio.cli import app
from misterio.llm import LlmError

runner = CliRunner()


def test_skills_ls_prints_chastushki() -> None:
    result = runner.invoke(app, ["skills", "ls"])
    assert result.exit_code == 0, result.output
    assert "chastushki" in result.output


def test_run_writes_markdown(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "про трактор",
            "--skill",
            "chastushki",
            "--fake",
            "--artifacts-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    files = list(tmp_path.glob("*.md"))
    assert len(files) == 1
    assert files[0].read_text(encoding="utf-8") == "fake-llm-output"


def test_run_unknown_skill_exits_with_error(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["run", "x", "--skill", "nope", "--fake", "--artifacts-dir", str(tmp_path)],
    )
    assert result.exit_code == 1
    assert "nope" in result.output
    assert list(tmp_path.glob("*.md")) == []


def test_run_llm_error_exits_with_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise LlmError("LLM недоступен")

    monkeypatch.setattr(cli, "run_job", boom)
    result = runner.invoke(
        app,
        ["run", "x", "--skill", "chastushki", "--fake", "--artifacts-dir", str(tmp_path)],
    )
    assert result.exit_code == 1
    assert "LLM недоступен" in result.output
    assert list(tmp_path.glob("*.md")) == []
