"""ClearML Serving preprocessing for the TF-IDF + LogReg sentiment pipeline.

clearml-serving (engine sklearn) loads the pickled Pipeline and calls .predict()
itself, so this class only shapes the input and output. preprocess turns the request
into a list of raw strings (TfidfVectorizer wants an iterable of documents, not a
numeric array); postprocess maps numeric predictions to labels.

The file runs inside the inference container, so it must be self-contained: no project
imports.
"""
from typing import Any

LABELS = {0: "negative", 1: "positive"}


class Preprocess:
    def preprocess(self, body: dict, state: dict, collect_custom_statistics_fn=None) -> Any:
        text = body.get("text", "")
        if isinstance(text, str):
            text = [text]
        return list(text)

    def postprocess(self, data: Any, state: dict, collect_custom_statistics_fn=None) -> dict:
        preds = data.tolist() if hasattr(data, "tolist") else list(data)
        if len(preds) == 1:
            label_id = int(preds[0])
            return {"label": LABELS.get(label_id, str(label_id)), "label_id": label_id}
        return {
            "predictions": [
                {"label": LABELS.get(int(p), str(p)), "label_id": int(p)} for p in preds
            ]
        }
