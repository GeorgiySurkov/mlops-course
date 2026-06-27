"""Stage 2: train a TF-IDF + LogReg sentiment classifier as a ClearML Task,
run remotely by the agent on the students queue.

The dataset is resolved from ClearML by project and name, so there is no hardcoded dataset_id.
"""
from __future__ import annotations

import argparse

from clearml import Dataset, OutputModel, Task

from config import cfg
from model import HParams, build_pipeline, confusion_figure, evaluate


def parse_args() -> argparse.Namespace:
    t = cfg.training
    p = argparse.ArgumentParser(description="Train a TF-IDF + LogReg sentiment classifier")
    p.add_argument("--name", default="baseline", help="experiment name (task name suffix)")
    p.add_argument("--max-features", type=int, default=t.max_features)
    p.add_argument("--ngram-min", type=int, default=t.ngram_min)
    p.add_argument("--ngram-max", type=int, default=t.ngram_max)
    p.add_argument("--C", type=float, default=t.C)
    p.add_argument("--max-iter", type=int, default=t.max_iter)
    p.add_argument("--solver", default=t.solver)
    p.add_argument("--local", action="store_true", help="run locally instead of enqueuing")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    task = Task.init(
        project_name=cfg.clearml.project_name,
        task_name=f"{cfg.training.task_name}-{args.name}",
        task_type=Task.TaskTypes.training,
        output_uri=True,  # upload model and artifacts to the ClearML file server
        auto_connect_frameworks={"scikit": False, "joblib": False},  # we register the model explicitly
        reuse_last_task_id=False,
    )

    # Hand off to the agent unless --local. Everything below this runs on the agent.
    if not args.local:
        task.execute_remotely(queue_name=cfg.clearml.queue_name, exit_process=True)

    import joblib
    import pandas as pd

    logger = task.get_logger()

    dataset = Dataset.get(
        dataset_project=cfg.clearml.dataset_project,
        dataset_name=cfg.clearml.dataset_name,
        only_completed=True,
        alias="training_data",
    )
    data_dir = dataset.get_local_copy()
    train_df = pd.read_csv(f"{data_dir}/train.csv")
    test_df = pd.read_csv(f"{data_dir}/test.csv")
    task.set_parameter("dataset/id", dataset.id)
    task.set_parameter("dataset/version", str(dataset.version))

    hp = HParams(
        max_features=args.max_features,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        C=args.C,
        max_iter=args.max_iter,
        solver=args.solver,
    )
    pipeline = build_pipeline(hp)
    pipeline.fit(train_df["text"], train_df["label"])

    y_true = test_df["label"]
    y_pred = pipeline.predict(test_df["text"])
    metrics = evaluate(y_true, y_pred)

    logger.report_single_value("accuracy", metrics["accuracy"])
    logger.report_single_value("f1", metrics["f1"])
    logger.report_scalar("metrics", "accuracy", value=metrics["accuracy"], iteration=0)
    logger.report_scalar("metrics", "f1", value=metrics["f1"], iteration=0)

    class_names = [cfg.labels[i] for i in sorted(cfg.labels)]
    fig = confusion_figure(y_true, y_pred, class_names, f"Confusion Matrix ({args.name})")
    logger.report_matplotlib_figure(
        title="Confusion Matrix", series=args.name, figure=fig, iteration=0, report_image=True
    )

    print(f"[{args.name}] accuracy={metrics['accuracy']:.4f} f1={metrics['f1']:.4f}")

    model_path = "model.pkl"
    joblib.dump(pipeline, model_path)
    task.upload_artifact("model", artifact_object=model_path)

    output_model = OutputModel(
        task=task,
        name=cfg.clearml.model_name,
        framework="ScikitLearn",
        tags=["sentiment", "tfidf-logreg", args.name],
        comment=f"TF-IDF + LogReg ({args.name})",
        label_enumeration={v: k for k, v in cfg.labels.items()},
    )
    output_model.update_weights(weights_filename=model_path, auto_delete_file=False)
    output_model.set_metadata("accuracy", f"{metrics['accuracy']:.4f}", v_type="float")
    output_model.set_metadata("f1", f"{metrics['f1']:.4f}", v_type="float")

    print(f"[{args.name}] registered draft model id={output_model.id}")
    task.close()


if __name__ == "__main__":
    main()
