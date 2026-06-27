"""Typed application config.

Static values come from config/config.yaml; server hosts can be overridden via
env or .env. Credentials stay in ~/clearml.conf, never here. Dataset IDs are
resolved at runtime by name, never stored here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import yaml

try:  # .env is optional; the SDK also reads ~/clearml.conf
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

CONFIG_PATH = Path(__file__).parent / "config.yaml"


@dataclass
class ClearMLConfig:
    project_name: str
    queue_name: str
    dataset_project: str
    dataset_name: str
    model_name: str
    api_host: str
    web_host: str
    files_host: str


@dataclass
class DataConfig:
    hf_dataset: str
    train_per_class: int
    test_per_class: int
    random_seed: int


@dataclass
class TrainingConfig:
    task_name: str
    max_features: int
    ngram_min: int
    ngram_max: int
    C: float
    max_iter: int
    solver: str


@dataclass
class RegistryConfig:
    min_f1: float
    tag: str


@dataclass
class ServingConfig:
    endpoint: str
    version: int
    port: int


@dataclass
class AppConfig:
    clearml: ClearMLConfig
    data: DataConfig
    training: TrainingConfig
    registry: RegistryConfig
    serving: ServingConfig
    labels: Dict[int, str]


def load_config(path: Path = CONFIG_PATH) -> AppConfig:
    raw = yaml.safe_load(path.read_text())
    c = raw["clearml"]
    clearml_cfg = ClearMLConfig(
        project_name=c["project_name"],
        queue_name=c["queue_name"],
        dataset_project=c["dataset_project"],
        dataset_name=c["dataset_name"],
        model_name=c["model_name"],
        api_host=os.getenv("CLEARML_API_HOST", "http://localhost:8008"),
        web_host=os.getenv("CLEARML_WEB_HOST", "http://localhost:8080"),
        files_host=os.getenv("CLEARML_FILES_HOST", "http://localhost:8081"),
    )
    data_cfg = DataConfig(**raw["data"])
    training_cfg = TrainingConfig(**raw["training"])
    registry_cfg = RegistryConfig(**raw["registry"])
    s = raw["serving"]
    serving_cfg = ServingConfig(endpoint=s["endpoint"], version=int(s["version"]), port=int(s["port"]))
    labels = {int(k): str(v) for k, v in raw["labels"].items()}
    return AppConfig(
        clearml=clearml_cfg,
        data=data_cfg,
        training=training_cfg,
        registry=registry_cfg,
        serving=serving_cfg,
        labels=labels,
    )


cfg = load_config()
