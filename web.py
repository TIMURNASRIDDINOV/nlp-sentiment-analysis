"""FastAPI web server for the NLP Sentiment application.

Provides REST endpoints that expose the same sentiment analysis
functionality available through the CLI.

Usage (standalone)::

    uvicorn web:create_app --factory --host 127.0.0.1 --port 5000

In practice this module is started via ``main.py --mode web``.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from cli import analyze_texts, build_pipeline
from config import settings
from modeling import analyze_modeling

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    """POST /analyze request body."""
    texts: List[str] = Field(
        ...,
        min_length=1,
        description="List of texts to analyse.",
        json_schema_extra={"example": ["I love this product!", "Terrible experience."]},
    )
    top_k: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of labels to return per text.",
    )

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: list[str]) -> list[str]:
        if len(v) > settings.max_batch_size:
            raise ValueError(
                f"Batch size {len(v)} exceeds maximum of {settings.max_batch_size}."
            )
        for i, text in enumerate(v):
            stripped = text.strip()
            if not stripped:
                raise ValueError(f"Text at index {i} is empty.")
            if len(stripped) > settings.max_text_length:
                raise ValueError(
                    f"Text at index {i} exceeds maximum length of "
                    f"{settings.max_text_length} characters."
                )
        return v


class SentimentResult(BaseModel):
    """Single label prediction."""
    label: str
    score: float


class TextResult(BaseModel):
    """Result for one input text."""
    text: str
    sentiments: List[SentimentResult]


class AnalyzeResponse(BaseModel):
    """POST /analyze response body."""
    results: List[TextResult]
    model: str
    elapsed_ms: float


class HealthResponse(BaseModel):
    """GET /health response body."""
    status: str
    model: str
    device: int


class ModelInfo(BaseModel):
    """GET /models response body."""
    current_model: str
    device: int
    top_k_supported: bool


# -- Modeling ---------------------------------------------------------------

class ModelingRequest(BaseModel):
    """POST /analyze_modeling request body."""
    texts: List[str] = Field(..., min_length=1)
    method: str = Field(default="tfidf") # "tfidf" or "bow"

class ModelingResultRow(BaseModel):
    text: str
    vector: List[float]

class ModelingResponse(BaseModel):
    method: str
    features: List[str]
    data: List[ModelingResultRow]


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app(
    model: str | None = None,
    device: int | None = None,
) -> FastAPI:
    """Build and return the FastAPI application.

    Args:
        model: Hugging Face model id (defaults to config/settings).
        device: ``-1`` for CPU, ``0+`` for CUDA device index.
    """
    model = model or settings.model
    device = device if device is not None else settings.device

    # -- Lifespan: load model on startup ------------------------------------
    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        log.info("Loading model '%s' on device %d …", model, device)
        try:
            app.state.nlp = build_pipeline(model, device)
            log.info("Model loaded successfully.")
        except Exception as exc:
            log.error("Failed to load model '%s': %s", model, exc)
            raise RuntimeError(f"Model load failed: {exc}") from exc
        yield

    app = FastAPI(
        title="NLP Sentiment API",
        description="Lightweight sentiment analysis powered by Hugging Face Transformers.",
        version="1.0.0",
        lifespan=_lifespan,
    )

    # -- CORS ----------------------------------------------------------------
    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- State ---------------------------------------------------------------
    app.state.nlp = None  # set by lifespan
    app.state.model_name = model
    app.state.device = device
    app.state.top_k_supported = True

    # -- Middleware: request timing ------------------------------------------
    @app.middleware("http")
    async def _timeout_middleware(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        elapsed = time.time() - start
        response.headers["X-Process-Time-Ms"] = f"{elapsed * 1000:.1f}"
        return response

    # -- Routes --------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    async def _serve_index():
        index = STATIC_DIR / "index.html"
        if not index.is_file():
            raise HTTPException(status_code=404, detail="Frontend not found.")
        return FileResponse(index, media_type="text/html")

    @app.get("/modeling", include_in_schema=False)
    async def _serve_modeling():
        page = STATIC_DIR / "modeling.html"
        if not page.is_file():
            raise HTTPException(status_code=404, detail="Modeling UI not found.")
        return FileResponse(page, media_type="text/html")

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def _health():
        """Health-check endpoint."""
        return HealthResponse(
            status="ok" if app.state.nlp is not None else "loading",
            model=app.state.model_name,
            device=app.state.device,
        )

    @app.get("/models", response_model=ModelInfo, tags=["ops"])
    async def _models():
        """Return information about the currently loaded model."""
        return ModelInfo(
            current_model=app.state.model_name,
            device=app.state.device,
            top_k_supported=app.state.top_k_supported,
        )

    @app.post("/analyze", response_model=AnalyzeResponse, tags=["analysis"])
    async def _analyze(body: AnalyzeRequest):
        """Analyse sentiment for one or more texts.

        Accepts a JSON body with ``texts`` (list of strings) and an optional
        ``top_k`` parameter.  Returns labelled sentiment predictions with
        confidence scores.
        """
        if app.state.nlp is None:
            raise HTTPException(status_code=503, detail="Model is still loading.")

        start = time.time()
        try:
            raw_results, effective_top_k = analyze_texts(
                app.state.nlp, body.texts, body.top_k
            )
        except Exception as exc:
            log.exception("Inference error")
            raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

        if effective_top_k < body.top_k:
            app.state.top_k_supported = False

        # Normalise results into a uniform list-of-lists shape.
        results: list[TextResult] = []
        for text, raw in zip(body.texts, raw_results):
            if isinstance(raw, dict):
                sentiments = [SentimentResult(label=raw["label"], score=round(raw["score"], 4))]
            else:
                sentiments = [
                    SentimentResult(label=r["label"], score=round(r["score"], 4))
                    for r in raw
                ]
            results.append(TextResult(text=text, sentiments=sentiments))

        elapsed_ms = (time.time() - start) * 1000
        return AnalyzeResponse(
            results=results,
            model=app.state.model_name,
            elapsed_ms=round(elapsed_ms, 1),
        )

    @app.post("/analyze_modeling", response_model=ModelingResponse, tags=["modeling"])
    async def _modeling(body: ModelingRequest):
        """Perform BoW or TF-IDF modeling on a list of documents."""
        try:
            result = analyze_modeling(body.texts, method=body.method)
            return ModelingResponse(**result)
        except Exception as exc:
            log.exception("Modeling error")
            raise HTTPException(status_code=500, detail=f"Modeling failed: {exc}")

    # -- Static files (CSS, JS) — mounted last so routes take priority ------
    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app
