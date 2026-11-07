import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
from typing import Literal, Protocol

from misterio.llm import LlmConfig, LlmError, OpenAiCompatibleLlm, Settings


@dataclass(frozen=True)
class ModelOption:
    key: str
    label: str
    base_url: str
    model: str
    api_style: Literal["openai", "ollama"]
    start_command: tuple[str, ...] | None
    requires_api_key: bool = False

    def llm_config(self) -> LlmConfig:
        return LlmConfig(
            base_url=self.base_url,
            model=self.model,
            api_style=self.api_style,
            requires_api_key=self.requires_api_key,
        )


DEFAULT_MODEL_KEY = "mistral"


MODEL_OPTIONS = {
    "mistral": ModelOption(
        key="mistral",
        label="Mistral Large 3",
        base_url="https://ollama.com/v1",
        model="mistral-large-3:675b",
        api_style="ollama",
        start_command=None,
        requires_api_key=True,
    ),
    "mimo": ModelOption(
        key="mimo",
        label="MiMo V2.6 Flash",
        base_url="https://ollama.com/v1",
        model="mimo-v2.6-flash",
        api_style="ollama",
        start_command=None,
        requires_api_key=True,
    ),
    "qwen": ModelOption(
        key="qwen",
        label="Qwen 3 · 8B",
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3:8b",
        api_style="ollama",
        start_command=("docker", "compose", "up", "-d", "ollama"),
    ),
    "gigachat": ModelOption(
        key="gigachat",
        label="GigaChat 3.1 · Q4",
        base_url="http://127.0.0.1:8080/v1",
        model="ai-sage/GigaChat3.1-10B-A1.8B-GGUF:Q4_K_M",
        api_style="openai",
        start_command=(
            "docker",
            "compose",
            "--profile",
            "gigachat",
            "up",
            "-d",
            "gigachat",
        ),
    ),
}

STOP_COMMAND = (
    "docker",
    "compose",
    "--profile",
    "gigachat",
    "stop",
    "ollama",
    "ollama-pull",
    "gigachat",
)
MODEL_SWITCH_LOCK = Lock()


class ModelSwitchError(RuntimeError):
    """Model backend could not be switched."""


class ModelSwitcher(Protocol):
    def active_key(self) -> str: ...

    def switch(self, key: str) -> None: ...


class ModelProfileSwitcher:
    def __init__(self, project_dir: Path) -> None:
        self._project_dir = project_dir
        self._active_key = DEFAULT_MODEL_KEY

    def active_key(self) -> str:
        return self._active_key

    def switch(self, key: str) -> None:
        option = MODEL_OPTIONS.get(key)
        if option is None:
            raise ValueError(f"unknown model: {key}")

        with MODEL_SWITCH_LOCK:
            if option.start_command is None:
                self._active_key = key
                return
            try:
                self._run(STOP_COMMAND)
                self._run(option.start_command)
            except (OSError, subprocess.SubprocessError) as exc:
                raise ModelSwitchError("Docker model switch failed") from exc
            self._active_key = key

    def _run(self, command: tuple[str, ...]) -> None:
        subprocess.run(
            command,
            cwd=self._project_dir,
            check=True,
            capture_output=True,
            text=True,
            timeout=1_800,
        )


def create_llm(
    profile_key: str = DEFAULT_MODEL_KEY,
    settings: Settings | None = None,
    timeout_s: float | None = None,
) -> tuple[OpenAiCompatibleLlm, str]:
    option = MODEL_OPTIONS.get(profile_key)
    if option is None:
        raise ValueError(f"unknown model: {profile_key}")
    active_settings = settings or Settings()
    if option.requires_api_key and not active_settings.llm_api_key:
        raise LlmError("MISTERIO_LLM_API_KEY is required for cloud models")
    config = option.llm_config()
    if timeout_s is not None:
        config = replace(config, timeout_s=timeout_s)
    return OpenAiCompatibleLlm(config, active_settings.llm_api_key), option.model
