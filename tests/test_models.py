import subprocess
from pathlib import Path

import pytest

import misterio.models as models
from misterio.llm import LlmError, Settings
from misterio.models import (
    DEFAULT_MODEL_KEY,
    MODEL_OPTIONS,
    STOP_COMMAND,
    ModelProfileSwitcher,
    ModelSwitchError,
    create_llm,
)


def test_default_model_is_mistral() -> None:
    assert DEFAULT_MODEL_KEY == "mistral"
    assert MODEL_OPTIONS[DEFAULT_MODEL_KEY].model == "mistral-large-3:675b"


def test_cloud_profiles_share_api_key() -> None:
    settings = Settings(llm_api_key="shared-secret")

    _, mistral_model = create_llm("mistral", settings)
    _, mimo_model = create_llm("mimo", settings)

    assert mistral_model == "mistral-large-3:675b"
    assert mimo_model == "mimo-v2.6-flash"


def test_cloud_profile_requires_api_key() -> None:
    with pytest.raises(LlmError, match="MISTERIO_LLM_API_KEY"):
        create_llm("mistral", Settings(llm_api_key=""))


def test_local_profile_does_not_require_api_key() -> None:
    _, model = create_llm("qwen", Settings(llm_api_key=""))

    assert model == "qwen3:8b"


def test_cloud_switch_does_not_call_docker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[tuple[str, ...]] = []

    def record(command: tuple[str, ...], **kwargs: object) -> None:
        commands.append(command)

    monkeypatch.setattr(models.subprocess, "run", record)

    switcher = ModelProfileSwitcher(tmp_path)
    switcher.switch("mimo")

    assert commands == []
    assert switcher.active_key() == "mimo"


def test_local_switch_stops_both_backends_before_starting_selected_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[tuple[str, ...]] = []

    def record(command: tuple[str, ...], **kwargs: object) -> None:
        commands.append(command)

    monkeypatch.setattr(models.subprocess, "run", record)

    switcher = ModelProfileSwitcher(tmp_path)
    switcher.switch("gigachat")

    assert commands == [STOP_COMMAND, MODEL_OPTIONS["gigachat"].start_command]
    assert switcher.active_key() == "gigachat"


def test_switch_keeps_active_model_when_docker_start_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def fail_second(command: tuple[str, ...], **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(models.subprocess, "run", fail_second)
    switcher = ModelProfileSwitcher(tmp_path)

    with pytest.raises(ModelSwitchError):
        switcher.switch("qwen")

    assert switcher.active_key() == DEFAULT_MODEL_KEY
