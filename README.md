# MLOps AITH 2026 coursework

A small text sentiment classifier taken through a full ML lifecycle on a self-hosted
ClearML. This is a course lab, so the infrastructure and the lifecycle are the point,
not model accuracy.

The goal: version the data, train on a remote agent, log everything, register the best
model, serve it over HTTP, and put a small UI in front.

## What was done

Everything runs on a self-hosted ClearML server with an agent picking up jobs from the
`students` queue.

The data is a balanced IMDB subset, versioned as a ClearML Dataset. Training runs as a
ClearML Task on the agent: a TF-IDF + LogReg pipeline, with hyperparameters, metrics and a
confusion matrix logged and the model saved as an artifact. A couple of experiments run
with different hyperparameters, and the best result is published to the ClearML Model
Registry behind a quality gate (an f1 floor, plus it has to beat the current production
model). From the registry the model is deployed to an inference endpoint through the
official ClearML Serving. A Streamlit UI sits in front and talks to that endpoint over
HTTP only, without loading the model itself.

For reproducibility, train and serve pin the same numpy/scikit-learn versions so the
pickled pipeline loads cleanly in the inference container.

## Quick start

```bash
make install        # uv sync -> .venv with the ClearML SDK
make server-up      # ClearML stack, web UI on http://localhost:8080 (first boot ~1-2 min)
# in the web UI (admin / admin1234) create new credentials and copy the block
make credentials    # uv run clearml-init: paste the keys -> ~/clearml.conf
make agent          # separate terminal: agent on the students queue (foreground)
make smoke          # enqueue a smoke task and check the agent runs it
```

Then run the rest in order: `make dataset`, `make experiments`, `make register`,
`make serve`, `make ui`. Run `make help` for the full list of targets.

## Layout

| Path | Purpose |
| --- | --- |
| `config/` | config loading (yaml + env) |
| `prepare_data.py`, `create_dataset.py` | build the IMDB subset and register it as a Dataset |
| `model.py`, `train.py`, `tests/` | pure ML logic, the training Task, offline tests |
| `register_model.py` | publish the best model to the registry |
| `serving/`, `scripts/deploy_serving.sh` | serving preprocessing and the deploy script |
| `ui/app.py` | Streamlit front end |
| `infra/` | docker-compose stacks for the ClearML server and serving |
| `.env.example`, `clearml.conf.example` | templates, real files are not committed |
