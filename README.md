# NLP Sentiment Analyzer

> Interactive sentiment analysis tool — CLI **and** Web — powered by Hugging Face Transformers.

---

## Project Overview

A **Python application** that performs real-time sentiment analysis on user-provided text. It wraps the Hugging Face `transformers` sentiment-analysis pipeline and exposes it through **two modes**:

| Mode              | Command                     | Description                                                             |
| ----------------- | --------------------------- | ----------------------------------------------------------------------- |
| **CLI** (default) | `python main.py`            | Interactive REPL — type text, get predictions in the terminal.          |
| **Web**           | `python main.py --mode web` | FastAPI server with a browser UI + REST API at `http://localhost:5000`. |

The default model is **DistilBERT fine-tuned on SST-2** (`distilbert-base-uncased-finetuned-sst-2-english`), but any compatible Hugging Face model can be swapped in via a CLI flag or environment variable.

---

## Architecture & Code Structure

```
NLP/
├── main.py                 # Entry point — mode selection (cli / web)
├── cli.py                  # CLI REPL logic, pipeline helpers, formatting
├── web.py                  # FastAPI application factory & REST endpoints
├── config.py               # Centralised settings (env vars + defaults)
├── static/
│   ├── index.html          # Web UI — single-page frontend
│   ├── style.css           # Styling (light/dark theme, responsive)
│   └── app.js              # Frontend logic (vanilla JS, no build step)
├── tests/
│   ├── test_main.py        # Unit tests for CLI functions
│   ├── test_web.py         # API endpoint tests (FastAPI TestClient)
│   └── test_integration.py # Entry-point / arg-parsing tests
├── Dockerfile              # Multi-stage production image
├── docker-compose.yml      # One-command startup
├── requirements.txt        # Runtime dependencies
├── requirements-dev.txt    # Dev/test dependencies
├── .env.example            # Environment variable template
└── README.md
```

### Module Breakdown

#### `cli.py` — Core Analysis Engine

| Function                                | Purpose                                                                                      |
| --------------------------------------- | -------------------------------------------------------------------------------------------- |
| `build_pipeline(model, device)`         | Initialises the HF `pipeline("sentiment-analysis", ...)` with model ID and device (CPU/GPU). |
| `analyze_texts(nlp, texts, top_k)`      | Runs inference on a batch. Gracefully falls back to top-1 if `top_k` is unsupported.         |
| `format_outputs(texts, results, top_k)` | Converts raw pipeline output into human-readable strings.                                    |
| `iter_inputs(batch_size)`               | Generator — reads stdin lines, buffers into batches, yields each batch.                      |
| `run_repl(nlp, top_k, batch_size)`      | Main REPL loop — read → analyse → print.                                                     |

#### `web.py` — FastAPI Server

| Endpoint   | Method | Description                                                               |
| ---------- | ------ | ------------------------------------------------------------------------- |
| `/`        | GET    | Serves the HTML frontend.                                                 |
| `/analyze` | POST   | Accepts JSON `{ "texts": [...], "top_k": 1 }`, returns sentiment results. |
| `/health`  | GET    | Health-check (status, model, device).                                     |
| `/models`  | GET    | Info about the currently loaded model.                                    |
| `/docs`    | GET    | Auto-generated OpenAPI / Swagger documentation.                           |

#### `config.py` — Configuration

All settings are read from environment variables (with sensible defaults) and exposed as a frozen `Settings` dataclass. Optionally loads a `.env` file via `python-dotenv`.

#### `main.py` — Orchestrator

Parses CLI arguments, selects mode, and delegates to either `cli.run_repl()` or `uvicorn.run(web.create_app())`.

### Key Design Decisions

- **Two modes, one entry-point** — `python main.py` defaults to CLI (backward-compatible); add `--mode web` for the server.
- **Shared analysis core** — both modes call the same `analyze_texts()` function from `cli.py`.
- **Graceful `top_k` fallback** — older `transformers` versions raise `TypeError`; the code catches this and degrades to top-1.
- **Batched processing** — `--batch-size` in CLI / JSON array in API — single forward pass for throughput.
- **Model loaded once** — in web mode the pipeline is created at startup and shared across requests.
- **No external CDN** — the frontend is fully self-contained (vanilla HTML/CSS/JS).

---

## Tech Stack

| Component         | Details                                                                    |
| ----------------- | -------------------------------------------------------------------------- |
| Language          | Python 3.10+                                                               |
| NLP Framework     | Hugging Face `transformers` ≥ 4.36.0                                       |
| Inference Backend | PyTorch (CPU or CUDA)                                                      |
| Web Framework     | FastAPI + Uvicorn                                                          |
| Frontend          | Vanilla HTML / CSS / JS (no build step)                                    |
| Default Model     | `distilbert-base-uncased-finetuned-sst-2-english` (SST-2 binary sentiment) |
| Testing           | `pytest` + `httpx` (FastAPI TestClient)                                    |
| Containerisation  | Docker + Docker Compose                                                    |

---

## Quick Start

### 1. Create & activate virtual environment

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
# venv\Scripts\activate       # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch installation can vary by platform/CUDA version. If `pip install` fails, follow the [official PyTorch instructions](https://pytorch.org/get-started/locally/) first, then re-run the command above.

### 3a. Run — CLI mode (default)

```bash
python main.py
```

```
Enter text (or 'exit'): I love this project
I love this project
  POSITIVE (0.999)
```

### 3b. Run — Web mode

```bash
python main.py --mode web
```

Open **http://localhost:5000** in your browser. The Swagger docs are at **http://localhost:5000/docs**.

---

## CLI Options

| Flag           | Type  | Default                                           | Applies to | Description                                           |
| -------------- | ----- | ------------------------------------------------- | ---------- | ----------------------------------------------------- |
| `--mode`       | `str` | `cli`                                             | both       | `cli` or `web`.                                       |
| `--model`      | `str` | `distilbert-base-uncased-finetuned-sst-2-english` | both       | Any HF model ID compatible with `sentiment-analysis`. |
| `--device`     | `int` | `-1`                                              | both       | `-1` CPU, `0` first GPU, `1` second GPU, …            |
| `--top-k`      | `int` | `1`                                               | both       | Number of labels to return per input.                 |
| `--batch-size` | `int` | `1`                                               | cli        | Lines to buffer before inference.                     |
| `--host`       | `str` | `127.0.0.1`                                       | web        | Bind address.                                         |
| `--port`       | `int` | `5000`                                            | web        | Server port.                                          |

### Examples

```bash
# CLI — top-2 labels, batch of 3, GPU
python main.py --top-k 2 --batch-size 3 --device 0

# Web — custom port, bind to all interfaces
python main.py --mode web --host 0.0.0.0 --port 8080
```

---

## API Reference

### `POST /analyze`

Analyse sentiment for one or more texts.

**Request:**

```json
{
  "texts": ["I love this!", "Terrible experience."],
  "top_k": 2
}
```

**Response:**

```json
{
  "results": [
    {
      "text": "I love this!",
      "sentiments": [
        { "label": "POSITIVE", "score": 0.9998 },
        { "label": "NEGATIVE", "score": 0.0002 }
      ]
    },
    {
      "text": "Terrible experience.",
      "sentiments": [
        { "label": "NEGATIVE", "score": 0.9995 },
        { "label": "POSITIVE", "score": 0.0005 }
      ]
    }
  ],
  "model": "distilbert-base-uncased-finetuned-sst-2-english",
  "elapsed_ms": 42.3
}
```

**curl example:**

```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great product!", "Worst service ever"], "top_k": 1}'
```

### `GET /health`

```bash
curl http://localhost:5000/health
# → { "status": "ok", "model": "distilbert-...", "device": -1 }
```

### `GET /models`

```bash
curl http://localhost:5000/models
# → { "current_model": "distilbert-...", "device": -1, "top_k_supported": true }
```

---

## Configuration

All settings can be controlled via environment variables. Copy `.env.example` to `.env` and edit:

```bash
cp .env.example .env
```

| Variable              | Default                                           | Description                            |
| --------------------- | ------------------------------------------------- | -------------------------------------- |
| `NLP_MODEL`           | `distilbert-base-uncased-finetuned-sst-2-english` | Model identifier                       |
| `NLP_DEVICE`          | `-1`                                              | Compute device                         |
| `NLP_TOP_K`           | `1`                                               | Default top-k                          |
| `NLP_HOST`            | `127.0.0.1`                                       | Web server host                        |
| `NLP_PORT`            | `5000`                                            | Web server port                        |
| `NLP_CORS_ORIGINS`    | `*`                                               | CORS allowed origins (comma-separated) |
| `NLP_MAX_BATCH_SIZE`  | `64`                                              | Max texts per request                  |
| `NLP_MAX_TEXT_LENGTH` | `5000`                                            | Max characters per text                |
| `NLP_REQUEST_TIMEOUT` | `30`                                              | Request timeout (seconds)              |
| `NLP_LOG_LEVEL`       | `INFO`                                            | Logging level                          |
| `NLP_DEBUG`           | `false`                                           | Debug mode                             |

CLI flags override environment variables when provided.

---

## Docker

### Build & run

```bash
docker compose up --build
```

The web UI will be available at **http://localhost:5000**.

### Build manually

```bash
docker build -t nlp-sentiment .
docker run -p 5000:5000 nlp-sentiment
```

### GPU support

Uncomment the `deploy` section in `docker-compose.yml` and ensure you have `nvidia-docker` installed.

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest -v
```

### Test suites

| File                        | Scope                                                            |
| --------------------------- | ---------------------------------------------------------------- |
| `tests/test_main.py`        | Unit tests for `cli.py` functions (analyse, format, edge cases). |
| `tests/test_web.py`         | API endpoint tests via FastAPI TestClient (mock pipeline).       |
| `tests/test_integration.py` | Entry-point arg parsing, exit codes, error handling.             |

All tests use a `DummyPipeline` mock — **no model download required**.

---

## Security Considerations

- **No authentication by default.** Do not expose to the public internet without adding auth.
- Default bind is `127.0.0.1` (localhost only). Use `--host 0.0.0.0` deliberately.
- Input length and batch size are capped via `NLP_MAX_TEXT_LENGTH` / `NLP_MAX_BATCH_SIZE`.
- For production, place behind a reverse proxy (nginx / Caddy) with TLS.
- CORS is set to `*` by default — restrict `NLP_CORS_ORIGINS` in production.

---

## Performance Tips

- **Use GPU** (`--device 0`) for significantly faster inference.
- **Increase batch size** in CLI mode (`--batch-size 8`) for throughput.
- **Send batches** in the API — one request with 10 texts is faster than 10 separate requests.
- The model is loaded once at startup; the first request may be slightly slower (warm-up).
- For high-traffic production use, consider running multiple Uvicorn workers: `uvicorn web:create_app --factory --workers 4`.

---

## Troubleshooting

| Problem                                        | Solution                                                                  |
| ---------------------------------------------- | ------------------------------------------------------------------------- |
| `ModuleNotFoundError: No module named 'torch'` | Install PyTorch separately: https://pytorch.org/get-started/locally/      |
| Port already in use                            | Change port: `--port 8080` or `NLP_PORT=8080`                             |
| Web mode says "fastapi not installed"          | Run `pip install -r requirements.txt`                                     |
| Model download hangs                           | Check internet connection; HF downloads the model on first run (~260 MB). |
| GPU out of memory                              | Switch to CPU (`--device -1`) or use a smaller model.                     |

---

## Migration Guide (from v1 → v2)

If you were using the original single-file version:

1. **`python main.py` still works identically** — CLI mode is the default, no flags changed.
2. Core functions (`analyze_texts`, `format_outputs`, `build_pipeline`) moved from `main.py` → `cli.py`. Update imports if you were importing from `main`.
3. New dependencies added to `requirements.txt` (`fastapi`, `uvicorn`, `python-dotenv`). Re-run `pip install -r requirements.txt`.
4. Tests moved: `test_main.py` now imports from `cli` instead of `main`.

---

## AI Prompt Context

If you are an AI assistant working on this codebase:

- **Entry point:** `main.py` → `main()` — dispatches to CLI or Web based on `--mode`.
- **CLI logic:** `cli.py` — `build_pipeline()`, `analyze_texts()`, `format_outputs()`, `run_repl()`.
- **Web logic:** `web.py` — `create_app()` factory returns a FastAPI application.
- **Config:** `config.py` — `Settings` dataclass, singleton `settings` object.
- **Frontend:** `static/index.html`, `static/style.css`, `static/app.js` — vanilla JS, no build step.
- **Uses Hugging Face `pipeline` API** — not raw model/tokenizer calls.
- **Stateless** — no database, no persistent storage.
- **Exit codes:** `0` = success, `1` = model/server error, `2` = invalid arguments.
- **To run CLI:** `python main.py` — **Web:** `python main.py --mode web` — **Tests:** `pytest`.
- **When extending:** keep CLI and Web layers thin; shared logic lives in `cli.py`.
