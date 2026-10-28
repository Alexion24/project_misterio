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

Команды CLI появятся на шаге S5.
