import subprocess
from pathlib import Path

import pytest

import misterio.models as models
from misterio.models import (
    MODEL_OPTIONS,
    STOP_COMMAND,
    DockerModelSwitcher,
    ModelSwitchError,
)


def test_switch_stops_both_backends_before_starting_selected_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "MISTERIO_LLM_API_KEY=secret\nMISTERIO_LLM_MODEL=old\n",
        encoding="utf-8",
    )
    commands: list[tuple[str, ...]] = []

    def record(command: tuple[str, ...], **kwargs: object) -> None:
        commands.append(command)

    monkeypatch.setattr(models.subprocess, "run", record)
    monkeypatch.setattr(models.os, "environ", {})

    DockerModelSwitcher(tmp_path).switch("gigachat")

    assert commands == [
        STOP_COMMAND,
        MODEL_OPTIONS["gigachat"].start_command,
    ]
    settings = env_path.read_text(encoding="utf-8")
    assert "MISTERIO_LLM_API_KEY=secret" in settings
    assert MODEL_OPTIONS["gigachat"].model in settings


def test_switch_restores_env_when_docker_start_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / ".env"
    original = b"MISTERIO_LLM_API_KEY=secret\nMISTERIO_LLM_MODEL=old\n"
    env_path.write_bytes(original)
    calls = 0

    def fail_second(command: tuple[str, ...], **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(models.subprocess, "run", fail_second)

    with pytest.raises(ModelSwitchError):
        DockerModelSwitcher(tmp_path).switch("qwen")

    assert env_path.read_bytes() == original
