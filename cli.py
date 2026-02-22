"""CLI (REPL) mode for interactive sentiment analysis.

This module contains all the original CLI logic extracted from `main.py`
so that the web layer and CLI layer can coexist without duplication.
"""

from __future__ import annotations

import logging
import sys
from typing import Iterable, Sequence, Tuple

from transformers import pipeline

log = logging.getLogger(__name__)

DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def build_pipeline(model: str, device: int):
    """Create and return a Hugging Face sentiment-analysis pipeline.

    Args:
        model: Hugging Face model identifier.
        device: ``-1`` for CPU, ``0+`` for a CUDA device index.

    Returns:
        A ``transformers.Pipeline`` object ready for inference.
    """
    log.info("Loading model '%s' on device %d …", model, device)
    return pipeline("sentiment-analysis", model=model, device=device)


def analyze_texts(nlp, texts: Sequence[str], top_k: int) -> Tuple[list, int]:
    """Run sentiment analysis on a batch of texts.

    Falls back to *top_k=1* automatically when the installed version of
    ``transformers`` does not support the ``top_k`` parameter.

    Returns:
        A tuple ``(results, effective_top_k)``.
    """
    if not texts:
        return [], top_k

    try:
        results = nlp(list(texts), top_k=top_k)
        return results, top_k
    except TypeError:
        results = nlp(list(texts))
        effective_top_k = 1 if top_k > 1 else top_k
        return results, effective_top_k


def format_outputs(texts: Sequence[str], results: list, top_k: int) -> list[str]:
    """Convert raw pipeline results into human-readable strings.

    Each entry looks like::

        <original_text>
          LABEL (0.xxx)
    """
    outputs: list[str] = []

    if top_k <= 1:
        for text, res in zip(texts, results):
            outputs.append(f"{text}\n  {res['label']} ({res['score']:.3f})")
        return outputs

    for text, res_list in zip(texts, results):
        lines = [text]
        for res in res_list:
            lines.append(f"  {res['label']} ({res['score']:.3f})")
        outputs.append("\n".join(lines))

    return outputs


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

def iter_inputs(batch_size: int) -> Iterable[list[str]]:
    """Yield batches of user input lines from stdin.

    Exits on ``exit``, ``quit``, or ``EOF``.
    """
    buffer: list[str] = []
    while True:
        try:
            raw = input("Enter text (or 'exit'): ")
        except EOFError:
            print()
            break

        if not raw.strip():
            continue

        if raw.strip().lower() in {"exit", "quit"}:
            break

        buffer.append(raw)
        if len(buffer) >= batch_size:
            yield buffer
            buffer = []

    if buffer:
        yield buffer


def run_repl(nlp, top_k: int, batch_size: int) -> None:
    """Start the interactive REPL loop.

    Reads lines from stdin, analyses sentiment in batches,
    and prints formatted results to stdout.
    """
    warned = False
    for batch in iter_inputs(batch_size):
        results, effective_top_k = analyze_texts(nlp, batch, top_k)
        if effective_top_k < top_k and not warned:
            log.warning(
                "Installed transformers version does not support --top-k; "
                "returning top-1 only."
            )
            print(
                "Warning: installed transformers version does not support "
                "--top-k; returning top-1 only.",
                file=sys.stderr,
            )
            warned = True
        for output in format_outputs(batch, results, effective_top_k):
            print(output)
