import logging
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from misterio.agent import run as run_job
from misterio.contracts import Job, JobStatus
from misterio.kb import LocalKb
from misterio.llm import Llm, LlmError, LlmTimeoutError
from misterio.models import (
    MODEL_OPTIONS,
    ModelProfileSwitcher,
    ModelSwitcher,
    ModelSwitchError,
    create_llm,
)
from misterio.skills import DEFAULT_SKILLS_DIR, list_skills

templates = Jinja2Templates(directory=DEFAULT_SKILLS_DIR.parent / "templates")
logger = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parents[2]
MAX_BRIEF_LENGTH = 2_000


def create_app(
    llm: Llm | None = None,
    model: str | None = None,
    artifacts_dir: Path = Path("data/artifacts"),
    model_switcher: ModelSwitcher | None = None,
    knowledge: LocalKb | None = None,
) -> FastAPI:
    app = FastAPI(title="Misterio")
    switcher = model_switcher or ModelProfileSwitcher(PROJECT_DIR)

    @app.get("/", response_class=HTMLResponse)
    def form(request: Request) -> HTMLResponse:
        return _page(request, switcher=switcher)

    @app.post("/model", response_class=HTMLResponse)
    def switch_model(
        request: Request,
        selected_model: str = Form(alias="model"),
    ) -> Response:
        enhanced_request = request.headers.get("x-misterio-action") == "switch-model"
        if (
            not _is_same_origin(request)
            or not enhanced_request
            and request.headers.get("sec-fetch-site") != "same-origin"
        ):
            return _page(
                request,
                switcher=switcher,
                error="Переключение модели отклонено.",
                status_code=403,
            )
        if selected_model not in MODEL_OPTIONS:
            return _page(
                request,
                switcher=switcher,
                error="Неизвестная модель.",
                status_code=400,
            )
        try:
            switcher.switch(selected_model)
        except ModelSwitchError:
            logger.exception("Model switch failed", extra={"model": selected_model})
            return _page(
                request,
                switcher=switcher,
                error="Не удалось переключить модель. Проверьте Docker.",
                status_code=503,
            )
        if not enhanced_request:
            return RedirectResponse(url="/", status_code=303)
        return _page(
            request,
            switcher=switcher,
            notice="Модель переключена. Первый ответ может занять больше времени.",
        )

    @app.post("/", response_class=HTMLResponse)
    def generate(
        request: Request,
        brief: str = Form(),
        skill: str = Form(),
    ) -> HTMLResponse:
        if not _is_same_origin(request):
            return _page(
                request,
                switcher=switcher,
                brief=brief,
                selected_skill=skill,
                error="Запрос отклонён.",
                status_code=403,
            )
        if len(brief) > MAX_BRIEF_LENGTH:
            return _page(
                request,
                switcher=switcher,
                brief=brief,
                selected_skill=skill,
                error=f"Задача должна быть короче {MAX_BRIEF_LENGTH} символов.",
                status_code=400,
            )
        known = {manifest.id for manifest in list_skills()}
        if skill not in known:
            return _page(
                request,
                switcher=switcher,
                brief=brief,
                selected_skill=skill,
                error=f"Неизвестный навык: {skill}",
                status_code=400,
            )

        job = Job(
            id=uuid4().hex,
            status=JobStatus.queued,
            brief=brief,
            skill_id=skill,
        )
        try:
            active_llm, active_model = _resolve_llm(llm, model, switcher.active_key())
            artifact = run_job(
                job, active_llm, model=active_model, knowledge=knowledge or LocalKb()
            )
        except LlmError as error:
            logger.exception(
                "LLM generation failed",
                extra={"job_id": job.id, "skill_id": skill},
            )
            return _page(
                request,
                switcher=switcher,
                brief=brief,
                selected_skill=skill,
                error=(
                    str(error)
                    if isinstance(error, LlmTimeoutError)
                    else (
                        "Не удалось получить ответ от модели. Попробуйте ещё раз позже."
                    )
                ),
                status_code=503,
            )
        except ValueError:
            logger.exception(
                "Job generation failed",
                extra={"job_id": job.id, "skill_id": skill},
            )
            return _page(
                request,
                switcher=switcher,
                brief=brief,
                selected_skill=skill,
                error="Не удалось выполнить задачу.",
                status_code=500,
            )
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / f"{job.id}.md").write_text(artifact.markdown, encoding="utf-8")
        (artifacts_dir / f"{job.id}.json").write_text(
            artifact.model_dump_json(indent=2), encoding="utf-8"
        )
        return _page(
            request,
            switcher=switcher,
            brief=brief,
            selected_skill=skill,
            result=artifact.markdown,
        )

    return app


def _is_same_origin(request: Request) -> bool:
    origin = request.headers.get("origin")
    expected = str(request.base_url).rstrip("/")
    return origin == expected or (
        origin is None and request.headers.get("sec-fetch-site") == "same-origin"
    )


def _resolve_llm(
    llm: Llm | None,
    model: str | None,
    profile_key: str,
) -> tuple[Llm, str]:
    if llm is not None:
        return llm, model or "fake"
    active_llm, active_model = create_llm(profile_key)
    return active_llm, model or active_model


def _page(
    request: Request,
    *,
    switcher: ModelSwitcher,
    brief: str = "",
    selected_skill: str = "",
    result: str | None = None,
    error: str | None = None,
    notice: str | None = None,
    status_code: int = 200,
) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "skills": list_skills(),
            "models": MODEL_OPTIONS.values(),
            "active_model": switcher.active_key(),
            "brief": brief,
            "selected_skill": selected_skill,
            "result": result,
            "error": error,
            "notice": notice,
        },
        status_code=status_code,
    )


app = create_app()
