"""Stage 3: publish the best trained model to the ClearML Model Registry.

Picks the highest-f1 task behind a quality gate (absolute floor, and it must beat
the current production model), then demotes any previous production model.
"""
from __future__ import annotations

import re
import sys

from clearml import Model, Task

from config import cfg


def _metric(task: Task, series: str) -> float:
    try:
        return float(task.get_reported_scalars()["metrics"][series]["y"][-1])
    except (KeyError, IndexError, TypeError):
        return -1.0


def _model_f1(model: Model) -> float:
    try:
        return float(model.get_metadata("f1") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _next_version(project: str) -> str:
    versions = []
    for m in Model.query_models(project_name=project) or []:
        for tag in m.tags or []:
            match = re.fullmatch(r"v(\d+)", tag)
            if match:
                versions.append(int(match.group(1)))
    return f"v{max(versions) + 1 if versions else 1}"


def main() -> None:
    project = cfg.clearml.project_name

    tasks = Task.get_tasks(
        project_name=project,
        task_name=cfg.training.task_name,  # matched as a regex (train-sentiment-*)
        task_filter={"status": ["completed"]},
    )
    if not tasks:
        sys.exit("no completed training tasks found — run Stage 2 first")

    best = max(tasks, key=lambda t: _metric(t, "f1"))
    best_f1 = _metric(best, "f1")
    print(f"best task: '{best.name}' (id={best.id}) f1={best_f1:.4f}")

    if best_f1 < cfg.registry.min_f1:
        sys.exit(f"quality gate failed: f1={best_f1:.4f} < min_f1={cfg.registry.min_f1}")

    current = list(Model.query_models(project_name=project, tags=[cfg.registry.tag]) or [])
    for m in current:
        if best_f1 <= _model_f1(m):
            print(f"current production model {m.id} f1={_model_f1(m):.4f} >= candidate; nothing to do")
            return

    output_models = best.get_models().get("output", [])
    if not output_models:
        sys.exit("best task has no registered output model (Stage 2 should create one)")
    model = output_models[-1]

    for m in current:  # drop the production tag from the old model(s) so only one stays current
        m.tags = [t for t in (m.tags or []) if t != cfg.registry.tag]
        print(f"demoted previous production model {m.id}")

    version = _next_version(project)
    model.tags = list(dict.fromkeys((model.tags or []) + [cfg.registry.tag, version]))
    model.set_metadata("f1", f"{best_f1:.4f}", v_type="float")
    model.set_metadata("accuracy", f"{_metric(best, 'accuracy'):.4f}", v_type="float")
    if not model.published:
        model.publish()

    print(f"PUBLISHED model id={model.id} version={version} tag='{cfg.registry.tag}' f1={best_f1:.4f}")
    print(f"  Stage 4 deploy:  clearml-serving --id <SERVICE_ID> model add ... --model-id {model.id}")


if __name__ == "__main__":
    main()
