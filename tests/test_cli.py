from pathlib import Path

import pytest
from typer.testing import CliRunner

import misterio.cli as cli
from misterio.cli import app
from misterio.llm import FakeLlm, LlmError

runner = CliRunner()


def test_skills_ls_prints_chastushki() -> None:
    result = runner.invoke(app, ["skills", "ls"])
    assert result.exit_code == 0, result.output
    assert "chastushki" in result.output


def test_kb_ingest_indexes_seed_files(tmp_path: Path) -> None:
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    (seed_dir / "facts.md").write_text("известный факт", encoding="utf-8")
    db_path = tmp_path / "kb.sqlite3"

    result = runner.invoke(
        app,
        ["kb", "ingest", "--seed-dir", str(seed_dir), "--db-path", str(db_path)],
    )

    assert result.exit_code == 0, result.output
    assert "1" in result.output
    assert db_path.exists()


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


@pytest.mark.parametrize(
    ("model_args", "expected"), [([], "deepseek"), (["--model", "mimo"], "mimo")]
)
def test_run_uses_selected_model_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_args: list[str],
    expected: str,
) -> None:
    selected: list[str] = []

    def fake_create_llm(profile_key: str) -> tuple[FakeLlm, str]:
        selected.append(profile_key)
        return FakeLlm(), profile_key

    monkeypatch.setattr(cli, "create_llm", fake_create_llm)
    result = runner.invoke(
        app,
        [
            "run",
            "topic",
            "--skill",
            "chastushki",
            *model_args,
            "--artifacts-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert selected == [expected]


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
        [
            "run",
            "x",
            "--skill",
            "chastushki",
            "--fake",
            "--artifacts-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 1
    assert "LLM недоступен" in result.output
    assert list(tmp_path.glob("*.md")) == []
