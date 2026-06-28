"""Stage 5: Streamlit UI for the sentiment classifier.

Calls the ClearML Serving endpoint over HTTP only, it never loads the model.
"""
from __future__ import annotations

import pathlib
import sys
import time

import requests
import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from config import cfg  # noqa: E402

st.set_page_config(page_title="Sentiment Classifier", page_icon="🎬")
st.title("Sentiment Classifier")
st.caption("TF-IDF + Logistic Regression served via ClearML Serving (HTTP only)")

with st.sidebar:
    st.header("Endpoint")
    endpoint_url = st.text_input("Serving URL", value=cfg.ui.default_serving_url)
    timeout = st.number_input("Timeout (s)", min_value=1, max_value=60, value=cfg.ui.request_timeout)
    if st.button("Health check"):
        try:
            t0 = time.perf_counter()
            resp = requests.post(endpoint_url, json={"text": "ping"}, timeout=timeout)
            resp.raise_for_status()
            st.success(f"Endpoint OK ({(time.perf_counter() - t0) * 1000:.0f} ms)")
        except requests.exceptions.RequestException as exc:
            st.error(f"Endpoint unreachable: {exc}")

text = st.text_area(
    "Text to classify",
    height=140,
    placeholder="e.g. This movie was absolutely fantastic!",
)

if st.button("Predict", type="primary"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        try:
            t0 = time.perf_counter()
            resp = requests.post(endpoint_url, json={"text": text}, timeout=timeout)
            latency_ms = (time.perf_counter() - t0) * 1000
            resp.raise_for_status()
            data = resp.json()

            col1, col2 = st.columns(2)
            col1.metric("Label", str(data.get("label", data)))
            col2.metric("Latency", f"{latency_ms:.0f} ms")
            with st.expander("Raw response"):
                st.json(data)
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot reach the endpoint at {endpoint_url}. Is clearml-serving up (Stage 4)?")
        except requests.exceptions.Timeout:
            st.error(f"Request timed out after {timeout}s.")
        except requests.exceptions.HTTPError as exc:
            st.error(f"HTTP {exc.response.status_code}: {exc.response.text[:300]}")
        except ValueError:
            st.error("Endpoint returned a non-JSON response.")
        except requests.exceptions.RequestException as exc:
            st.error(f"Request failed: {exc}")
