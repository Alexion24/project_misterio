import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Protocol

from misterio.llm import Settings


@dataclass(frozen=True)
class ModelOption:
    key: str
    label: str
    base_url: str
    model: str
    start_command: tuple[str, ...]


MODEL_OPTIONS = {
    "qwen": ModelOption(
        key="qwen",
        label="Qwen 3 · 8B",
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3:8b",
        start_command=("docker", "compose", "up", "-d", "ollama"),
    ),
    "gigachat": ModelOption(
        key="gigachat",
        label="GigaChat 3.1 · Q4",
        base_url="http://127.0.0.1:8080/v1",
        model="ai-sage/GigaChat3.1-10B-A1.8B-GGUF:Q4_K_M",
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


class DockerModelSwitcher:
    def __init__(self, project_dir: Path) -> None:
        self._project_dir = project_dir
        self._env_path = project_dir / ".env"
        self._active_key = self._detect_active_key()

    def active_key(self) -> str:
        return self._active_key

    def _detect_active_key(self) -> str:
        settings = Settings()
        for option in MODEL_OPTIONS.values():
            if settings.llm_model == option.model:
                return option.key
        return "qwen"

    def switch(self, key: str) -> None:
        option = MODEL_OPTIONS.get(key)
        if option is None:
            raise ValueError(f"unknown model: {key}")

        with MODEL_SWITCH_LOCK:
            original_env = (
                self._env_path.read_bytes() if self._env_path.exists() else None
            )
            try:
                self._run(STOP_COMMAND)
                self._write_settings(option)
                self._run(option.start_command)
            except (OSError, subprocess.SubprocessError) as exc:
                self._restore_settings(original_env)
                raise ModelSwitchError("Docker model switch failed") from exc
            os.environ["MISTERIO_LLM_BASE_URL"] = option.base_url
            os.environ["MISTERIO_LLM_MODEL"] = option.model
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

    def _write_settings(self, option: ModelOption) -> None:
        values = {
            "MISTERIO_LLM_BASE_URL": option.base_url,
            "MISTERIO_LLM_MODEL": option.model,
        }
        lines = (
            self._env_path.read_text(encoding="utf-8").splitlines()
            if self._env_path.exists()
            else []
        )
        seen: set[str] = set()
        updated: list[str] = []
        for line in lines:
            name, separator, _ = line.partition("=")
            if separator and name in values:
                updated.append(f"{name}={values[name]}")
                seen.add(name)
            else:
                updated.append(line)
        updated.extend(
            f"{name}={value}" for name, value in values.items() if name not in seen
        )

        self._replace_settings(("\n".join(updated) + "\n").encode())

    def _restore_settings(self, content: bytes | None) -> None:
        if content is None:
            self._env_path.unlink(missing_ok=True)
            return
        self._replace_settings(content)

    def _replace_settings(self, content: bytes) -> None:
        file_descriptor, temp_name = tempfile.mkstemp(prefix="misterio-env-")
        temp_path = Path(temp_name)
        try:
            with os.fdopen(file_descriptor, "wb") as temp_file:
                temp_file.write(content)
            os.replace(temp_path, self._env_path)
        finally:
            temp_path.unlink(missing_ok=True)
