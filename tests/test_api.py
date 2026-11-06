import logging
from pathlib import Path

from fastapi.testclient import TestClient

from misterio.api import create_app
from misterio.contracts import LlmCallSpec
from misterio.llm import FakeLlm, LlmError
from misterio.models import ModelSwitchError

ORIGIN_HEADERS = {"Origin": "http://testserver"}
MODEL_HEADERS = {
    "Origin": "http://testserver",
    "X-Misterio-Action": "switch-model",
}


class FailingLlm:
    def complete(self, spec: LlmCallSpec) -> str:
        raise LlmError("connection failed: secret-internal-url")


class FakeModelSwitcher:
    def __init__(self, active: str = "qwen", fail: bool = False) -> None:
        self.active = active
        self.fail = fail
        self.calls: list[str] = []

    def active_key(self) -> str:
        return self.active

    def switch(self, key: str) -> None:
        self.calls.append(key)
        if self.fail:
            raise ModelSwitchError("docker details")
        self.active = key


def test_form_lists_skills() -> None:
    response = TestClient(create_app(FakeLlm())).get("/")

    assert response.status_code == 200
    assert "chastushki" in response.text
    assert "Задача" in response.text
    assert 'id="generation-status"' in response.text
    assert 'aria-label="Генерация результата"' in response.text
    assert "Qwen 3" in response.text
    assert "GigaChat 3.1" in response.text


def test_model_switch_uses_allowlisted_key() -> None:
    switcher = FakeModelSwitcher()
    client = TestClient(create_app(FakeLlm(), model_switcher=switcher))

    response = client.post(
        "/model",
        data={"model": "gigachat"},
        headers=MODEL_HEADERS,
    )

    assert response.status_code == 200
    assert switcher.calls == ["gigachat"]
    assert "Модель переключена" in response.text


def test_model_switch_rejects_unknown_key() -> None:
    switcher = FakeModelSwitcher()
    client = TestClient(create_app(FakeLlm(), model_switcher=switcher))

    response = client.post(
        "/model",
        data={"model": "arbitrary-command"},
        headers=MODEL_HEADERS,
    )

    assert response.status_code == 400
    assert switcher.calls == []
    assert "Неизвестная модель" in response.text


def test_model_switch_hides_docker_error() -> None:
    switcher = FakeModelSwitcher(fail=True)
    client = TestClient(create_app(FakeLlm(), model_switcher=switcher))

    response = client.post(
        "/model",
        data={"model": "qwen"},
        headers=MODEL_HEADERS,
    )

    assert response.status_code == 503
    assert "Проверьте Docker" in response.text
    assert "docker details" not in response.text


def test_model_switch_requires_ui_header() -> None:
    switcher = FakeModelSwitcher()
    client = TestClient(create_app(FakeLlm(), model_switcher=switcher))

    response = client.post("/model", data={"model": "qwen"})

    assert response.status_code == 403
    assert switcher.calls == []


def test_form_generates_displays_and_saves_result(tmp_path: Path) -> None:
    client = TestClient(create_app(FakeLlm("готовая частушка"), artifacts_dir=tmp_path))

    response = client.post(
        "/",
        data={"brief": "про трактор", "skill": "chastushki"},
        headers=ORIGIN_HEADERS,
    )

    assert response.status_code == 200
    assert "готовая частушка" in response.text
    assert "про трактор" in response.text
    artifacts = list(tmp_path.glob("*.md"))
    assert len(artifacts) == 1
    assert artifacts[0].read_text(encoding="utf-8") == "готовая частушка"


def test_form_rejects_unknown_skill() -> None:
    response = TestClient(create_app(FakeLlm())).post(
        "/",
        data={"brief": "тема", "skill": "unknown"},
        headers=ORIGIN_HEADERS,
    )

    assert response.status_code == 400
    assert "Неизвестный навык" in response.text


def test_form_hides_llm_error_details_and_logs_exception(
    caplog,
    tmp_path: Path,
) -> None:
    client = TestClient(create_app(FailingLlm(), artifacts_dir=tmp_path))

    with caplog.at_level(logging.ERROR, logger="misterio.api"):
        response = client.post(
            "/",
            data={"brief": "про трактор", "skill": "chastushki"},
            headers=ORIGIN_HEADERS,
        )

    assert response.status_code == 503
    assert "Не удалось получить ответ от модели" in response.text
    assert "secret-internal-url" not in response.text
    assert "LLM generation failed" in caplog.text
    assert "secret-internal-url" in caplog.text
    assert list(tmp_path.glob("*.md")) == []


def test_generation_rejects_cross_origin_request(tmp_path: Path) -> None:
    client = TestClient(create_app(FakeLlm(), artifacts_dir=tmp_path))

    response = client.post(
        "/",
        data={"brief": "про трактор", "skill": "chastushki"},
        headers={"Origin": "https://example.test"},
    )

    assert response.status_code == 403
    assert list(tmp_path.glob("*.md")) == []


def test_generation_rejects_oversized_brief(tmp_path: Path) -> None:
    client = TestClient(create_app(FakeLlm(), artifacts_dir=tmp_path))

    response = client.post(
        "/",
        data={"brief": "я" * 2_001, "skill": "chastushki"},
        headers=ORIGIN_HEADERS,
    )

    assert response.status_code == 400
    assert list(tmp_path.glob("*.md")) == []
