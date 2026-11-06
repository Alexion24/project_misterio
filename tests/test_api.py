import logging
from pathlib import Path

from fastapi.testclient import TestClient

from misterio.api import create_app
from misterio.contracts import LlmCallSpec
from misterio.llm import FakeLlm, LlmError


class FailingLlm:
    def complete(self, spec: LlmCallSpec) -> str:
        raise LlmError("connection failed: secret-internal-url")


def test_form_lists_skills() -> None:
    response = TestClient(create_app(FakeLlm())).get("/")

    assert response.status_code == 200
    assert "chastushki" in response.text
    assert "Задача" in response.text
    assert 'id="generation-status"' in response.text
    assert 'aria-label="Генерация результата"' in response.text


def test_form_generates_displays_and_saves_result(tmp_path: Path) -> None:
    client = TestClient(create_app(FakeLlm("готовая частушка"), artifacts_dir=tmp_path))

    response = client.post("/", data={"brief": "про трактор", "skill": "chastushki"})

    assert response.status_code == 200
    assert "готовая частушка" in response.text
    assert "про трактор" in response.text
    artifacts = list(tmp_path.glob("*.md"))
    assert len(artifacts) == 1
    assert artifacts[0].read_text(encoding="utf-8") == "готовая частушка"


def test_form_rejects_unknown_skill() -> None:
    response = TestClient(create_app(FakeLlm())).post(
        "/", data={"brief": "тема", "skill": "unknown"}
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
            "/", data={"brief": "про трактор", "skill": "chastushki"}
        )

    assert response.status_code == 503
    assert "Не удалось получить ответ от модели" in response.text
    assert "secret-internal-url" not in response.text
    assert "LLM generation failed" in caplog.text
    assert "secret-internal-url" in caplog.text
    assert list(tmp_path.glob("*.md")) == []
