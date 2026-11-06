import logging
from uuid import uuid4

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from misterio.agent import run as run_job
from misterio.contracts import Job, JobStatus
from misterio.llm import Llm, LlmError, OpenAiCompatibleLlm, Settings
from misterio.skills import DEFAULT_SKILLS_DIR, list_skills

templates = Jinja2Templates(directory=DEFAULT_SKILLS_DIR.parent / "templates")
logger = logging.getLogger(__name__)


def create_app(llm: Llm | None = None, model: str | None = None) -> FastAPI:
    app = FastAPI(title="Misterio")

    @app.get("/", response_class=HTMLResponse)
    def form(request: Request) -> HTMLResponse:
        return _page(request)

    @app.post("/", response_class=HTMLResponse)
    def generate(
        request: Request,
        brief: str = Form(),
        skill: str = Form(),
    ) -> HTMLResponse:
        known = {manifest.id for manifest in list_skills()}
        if skill not in known:
            return _page(
                request,
                brief=brief,
                selected_skill=skill,
                error=f"Неизвестный навык: {skill}",
                status_code=400,
            )

        active_llm, active_model = _resolve_llm(llm, model)
        job = Job(
            id=uuid4().hex,
            status=JobStatus.queued,
            brief=brief,
            skill_id=skill,
        )
        try:
            artifact = run_job(job, active_llm, model=active_model)
        except LlmError:
            logger.exception(
                "LLM generation failed",
                extra={"job_id": job.id, "skill_id": skill},
            )
            return _page(
                request,
                brief=brief,
                selected_skill=skill,
                error="Не удалось получить ответ от модели. Попробуйте ещё раз позже.",
                status_code=503,
            )
        except ValueError:
            logger.exception(
                "Job generation failed",
                extra={"job_id": job.id, "skill_id": skill},
            )
            return _page(
                request,
                brief=brief,
                selected_skill=skill,
                error="Не удалось выполнить задачу.",
                status_code=500,
            )
        return _page(
            request,
            brief=brief,
            selected_skill=skill,
            result=artifact.markdown,
        )

    return app


def _resolve_llm(llm: Llm | None, model: str | None) -> tuple[Llm, str]:
    if llm is not None:
        return llm, model or "fake"
    settings = Settings()
    return OpenAiCompatibleLlm(settings), model or settings.llm_model


def _page(
    request: Request,
    *,
    brief: str = "",
    selected_skill: str = "",
    result: str | None = None,
    error: str | None = None,
    status_code: int = 200,
) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "skills": list_skills(),
            "brief": brief,
            "selected_skill": selected_skill,
            "result": result,
            "error": error,
        },
        status_code=status_code,
    )


app = create_app()
