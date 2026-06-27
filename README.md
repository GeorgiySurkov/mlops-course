# MLOps AITH 2026 — курсовой проект

Сервис классификации текстов (sentiment) на ClearML. Главное — жизненный цикл модели
и инфраструктура, а не метрики качества. Полные требования по этапам:
[`docs/hw-task/Курсовой проект – MLOps AITH 2026.md`](docs/hw-task/Курсовой%20проект%20%E2%80%93%20MLOps%20AITH%202026.md).

## Этап 0 — подготовка инфраструктуры

Развернуть self-hosted ClearML Server, настроить SDK, поднять ClearML Agent на очереди
`students` и убедиться, что задача выполняется именно агентом.

### Предпосылки

- Docker (Docker Desktop на macOS) с ~8 ГБ памяти для VM
- [uv](https://docs.astral.sh/uv/) — управление Python-окружением

### Запуск

```bash
make install        # uv sync -> .venv с ClearML SDK
make server-up      # поднять стек ClearML (UI на http://localhost:8080, ~1-2 мин)
# войти в UI как admin / admin1234 (fixed-admin из infra/clearml-server/apiserver.conf),
# Settings -> Workspace -> Create new credentials, скопировать блок
make credentials    # uv run clearml-init: вставить ключи -> ~/clearml.conf
make agent          # в отдельном терминале: агент на очереди students (foreground)
make smoke          # отправить smoke-задачу и проверить выполнение агентом
```

`make help` — список целей.

### Проверка (критерии этапа 0)

- В UI (Workers & Queues) виден агент-воркер.
- Задача `stage0-smoke` (проект «MLOps Course Sentiment») уходит в очередь `students`.
- Задача выполняется агентом (не локально): статус queued → running, в логе строка
  «Hello from the ClearML agent…».

### Структура

| Путь | Назначение |
| --- | --- |
| `pyproject.toml`, `uv.lock` | зависимости Stage 0 (uv) |
| `infra/clearml-server/` | docker-compose стек ClearML + fixed-admin (`apiserver.conf`) |
| `infra/agent/run-agent.sh` | запуск ClearML Agent через `uvx` |
| `scripts/smoke_task.py` | smoke-задача для проверки удалённого исполнения |
| `config/` | загрузка конфигурации (yaml + env) |
| `.env.example`, `clearml.conf.example` | шаблоны (реальные файлы не коммитятся) |
