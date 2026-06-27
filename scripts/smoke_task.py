"""Stage 0 smoke test: enqueue a trivial Task to the students queue and let the
agent run it, which proves remote (not local) execution.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from clearml import Task  # noqa: E402

from config import cfg  # noqa: E402


def main() -> None:
    task = Task.init(
        project_name=cfg.clearml.project_name,
        task_name="stage0-smoke",
        task_type=Task.TaskTypes.custom,
    )
    # Enqueue and stop locally; the agent re-runs the script from the top.
    task.execute_remotely(queue_name=cfg.clearml.queue_name, exit_process=True)

    # Everything below runs on the agent only.
    import platform

    logger = task.get_logger()
    logger.report_text(f"Hello from the ClearML agent running on {platform.node()}")
    logger.report_single_value(name="smoke_ok", value=1)
    print("Stage 0 smoke task executed by the agent.")


if __name__ == "__main__":
    main()
