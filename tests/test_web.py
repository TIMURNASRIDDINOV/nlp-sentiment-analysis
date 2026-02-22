"""Tests for the FastAPI web layer (web.py).

Uses the FastAPI TestClient with a DummyPipeline so that no real
model download is needed.
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from web import create_app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _DummyPipeline:
    """Minimal pipeline mock — returns static results."""

    def __call__(self, texts, top_k=None):
        if top_k is not None and top_k > 1:
            return [
                [
                    {"label": "POSITIVE", "score": 0.95},
                    {"label": "NEGATIVE", "score": 0.05},
                ]
                for _ in texts
            ]
        return [{"label": "POSITIVE", "score": 0.95} for _ in texts]


@pytest.fixture()
def client():
    """Create a TestClient with a mocked pipeline."""
    with patch("web.build_pipeline", return_value=_DummyPipeline()):
        app = create_app(model="test-model", device=-1)
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# Health / info endpoints
# ---------------------------------------------------------------------------

def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["model"] == "test-model"


def test_models(client):
    res = client.get("/models")
    assert res.status_code == 200
    body = res.json()
    assert body["current_model"] == "test-model"


# ---------------------------------------------------------------------------
# POST /analyze
# ---------------------------------------------------------------------------

def test_analyze_single_text(client):
    res = client.post("/analyze", json={"texts": ["Hello world"]})
    assert res.status_code == 200
    body = res.json()
    assert len(body["results"]) == 1
    assert body["results"][0]["sentiments"][0]["label"] == "POSITIVE"


def test_analyze_batch(client):
    res = client.post(
        "/analyze", json={"texts": ["Good", "Bad", "Okay"], "top_k": 1}
    )
    assert res.status_code == 200
    assert len(res.json()["results"]) == 3


def test_analyze_top_k(client):
    res = client.post("/analyze", json={"texts": ["Test"], "top_k": 2})
    assert res.status_code == 200
    sentiments = res.json()["results"][0]["sentiments"]
    assert len(sentiments) == 2


def test_analyze_empty_texts(client):
    res = client.post("/analyze", json={"texts": []})
    assert res.status_code == 422  # validation error


def test_analyze_blank_text(client):
    res = client.post("/analyze", json={"texts": ["   "]})
    assert res.status_code == 422


def test_analyze_missing_body(client):
    res = client.post("/analyze")
    assert res.status_code == 422


def test_analyze_text_too_long(client):
    long_text = "a" * 6000
    res = client.post("/analyze", json={"texts": [long_text]})
    assert res.status_code == 422


def test_analyze_returns_elapsed(client):
    res = client.post("/analyze", json={"texts": ["Fast"]})
    assert "elapsed_ms" in res.json()


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

def test_index_serves_html(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
