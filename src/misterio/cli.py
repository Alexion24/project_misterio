from pathlib import Path
from typing import Annotated
from uuid import uuid4

import typer

from misterio.agent import run as run_job
from misterio.contracts import Job, JobStatus
from misterio.llm import FakeLlm, LlmError, OpenAiCompatibleLlm, Settings
from misterio.skills import list_skills

app = typer.Typer(no_args_is_help=True)
skills_app = typer.Typer(no_args_is_help=True)
app.add_typer(skills_app, name="skills")


def _fail(message: str) -> typer.Exit:
    typer.echo(f"Ошибка: {message}", err=True)
    return typer.Exit(1)


@skills_app.command("ls")
def skills_ls() -> None:
    for s in list_skills():
        typer.echo(f"{s.id}\t{s.version}")


@app.command("run")
def run_cmd(
    brief: Annotated[str, typer.Argument(help="Текст задачи")],
    skill: Annotated[str, typer.Option("--skill")],
    fake: Annotated[bool, typer.Option("--fake")] = False,
    artifacts_dir: Annotated[Path, typer.Option("--artifacts-dir")] = Path("data/artifacts"),
) -> None:
    known = {s.id for s in list_skills()}
    if skill not in known:
        raise _fail(f"неизвестный навык '{skill}'. Доступно: {', '.join(sorted(known))}")

    if fake:
        llm, model = FakeLlm(), "fake"
    else:
        settings = Settings()
        llm, model = OpenAiCompatibleLlm(settings), settings.llm_model

    job = Job(id=uuid4().hex, status=JobStatus.queued, brief=brief, skill_id=skill)
    try:
        artifact = run_job(job, llm, model=model)
    except LlmError as e:
        raise _fail(str(e)) from e

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / f"{job.id}.md"
    path.write_text(artifact.markdown, encoding="utf-8")
    typer.echo(str(path))
