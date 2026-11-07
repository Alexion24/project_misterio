# Misterio

Локальный агент для text-to-text навыков. Сейчас доступен навык генерации
частушек; LLM может работать через Ollama Cloud либо локально через Ollama или
`llama.cpp`.

## Быстрый старт

Нужны Python 3.12+, [uv](https://docs.astral.sh/uv/), Docker с Compose,
NVIDIA Container Toolkit и Git Bash.

```bash
uv sync
cp .env.example .env
# Укажите MISTERIO_LLM_API_KEY в .env
./mist ui
```

Откройте `http://127.0.0.1:8000`. По умолчанию используется облачный профиль
Mistral Large 3.

## Основные команды

```bash
./mist ui                  # локальный интерфейс
./mist run "про картошку"  # навык chastushki из терминала
./mist qwen                # запустить локальный Qwen
./mist giga                # запустить локальный GigaChat
./mist pull qwen           # скачать только Qwen
./mist pull giga           # скачать только GigaChat
./mist pull all            # скачать обе модели без запуска
./mist status              # состояние LLM-контейнеров
./mist down                # остановить LLM-контейнеры
./mist test                # тесты
./mist lint                # Ruff
```

Полный список: `./mist help`.

## Модели

Реестр содержит четыре профиля:

- `mistral` — Mistral Large 3 в Ollama Cloud, production-профиль по умолчанию;
- `mimo` — MiMo V2.6 Flash в Ollama Cloud, экономичный ручной fallback;
- `qwen` — локальный `qwen3:8b` через Ollama;
- `gigachat` — локальный GigaChat 3.1 Lightning Q4_K_M через `llama.cpp`.

Выбор облачного профиля хранится в памяти процесса и не запускает Docker. После
перезапуска снова выбирается `mistral`; автоматического fallback между моделями
нет. Для локальных профилей переключатель сначала останавливает оба
LLM-сервиса, затем запускает выбранный, поэтому они не занимают GPU
одновременно.

Скачать GigaChat заранее, без запуска сервера:

```bash
./mist pull giga
```

Обе модели можно скачать заранее, не запуская LLM-серверы:

```bash
./mist pull all
```

Для загрузки Qwen сервис Ollama временно запускается как зависимость
`ollama-pull` и останавливается сразу после скачивания. GigaChat загружается без
запуска `llama.cpp`.

В `.env` хранится только `MISTERIO_LLM_API_KEY`, необходимый облачным профилям.
Локальным backend ключ не передаётся. URL, API-стиль и точные model ID заданы в
реестре профилей, а не в окружении.

## CLI

```bash
uv run misterio skills ls
uv run misterio kb ingest
uv run misterio run "про трактор" --skill chastushki
uv run misterio run "про трактор" --skill chastushki --model mimo
```

Результаты сохраняются в `data/artifacts/`. Локальные данные из `data/` не
попадают в Git.

## Разработка

```bash
./mist test
./mist lint
uv run pre-commit install
```

Архитектура проекта описана в `docs/architecture.md`.
