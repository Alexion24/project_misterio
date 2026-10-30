# Misterio

Локальный контент-агент: навыки в `skills/`, KB в `data/`, LLM через OpenAI-compatible (local/free). План: `.cursor/plans/plan_a_prime_local.md`.

## Setup

```bash
uv sync
```

Требуется Python 3.12+.

## Структура

- `src/misterio/` — один пакет (contracts, agent, kb, llm, cli — по шагам плана)
- `skills/` — навыки (папки)
- `kb/seed/` — стартовый корпус KB
- `evals/` — брифы для проверок
- `data/` — sqlite, индекс, артефакты (не в git)

## CLI

```bash
uv run misterio skills ls
uv run misterio run "про трактор" --skill chastushki --fake
```

Артефакт пишется в `data/artifacts/`. Без `--fake` используется OpenAI-compatible LLM из Settings (`MISTERIO_LLM_BASE_URL`, `MISTERIO_LLM_API_KEY`, `MISTERIO_LLM_MODEL`).
