# Misterio

Локальный агент для text-to-text навыков. Сейчас доступен навык генерации
частушек; LLM запускается локально через Ollama или `llama.cpp`.

## Быстрый старт

Нужны Python 3.12+, [uv](https://docs.astral.sh/uv/), Docker с Compose,
NVIDIA Container Toolkit и Git Bash.

```bash
uv sync
cp .env.example .env
./mist pull qwen
./mist up
./mist ui
```

Откройте `http://127.0.0.1:8000`. Первый запуск скачивает модель и может занять
несколько минут.

## Основные команды

```bash
./mist ui                  # локальный интерфейс
./mist run "про картошку"  # навык chastushki из терминала
./mist qwen                # остановить GigaChat и включить Qwen
./mist giga                # остановить Qwen и включить GigaChat
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

По умолчанию используется `qwen3:8b` через Ollama. `./mist giga` запускает
GigaChat 3.1 Lightning Q4_K_M через `llama.cpp`. Переключатель сначала
останавливает оба LLM-сервиса и только затем запускает выбранный, поэтому модели
не занимают GPU одновременно. Его загрузка вынесена в отдельный контейнер
`gigachat-pull`. Он сохраняет GGUF в отдельный model volume, а сервер открывает
конкретный локальный файл и не обращается к сети.

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

Активные `MISTERIO_LLM_BASE_URL` и `MISTERIO_LLM_MODEL` сохраняются в локальном
`.env`, который исключён из Git. Ключ API переключатель не читает и не меняет.

## CLI

```bash
uv run misterio skills ls
uv run misterio kb ingest
uv run misterio run "про трактор" --skill chastushki
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
