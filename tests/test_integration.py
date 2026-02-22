"""Integration tests — verify CLI entry-point argument parsing and exit codes.

These tests exercise ``main.main()`` without loading a real model by
patching ``cli.build_pipeline``.
"""

from unittest.mock import patch

import pytest

from main import main, parse_args


class _DummyPipeline:
    def __call__(self, texts, top_k=None):
        return [{"label": "POSITIVE", "score": 0.99} for _ in texts]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def test_parse_defaults():
    args = parse_args([])
    assert args.mode == "cli"
    assert args.top_k == 1
    assert args.batch_size == 1


def test_parse_web_mode():
    args = parse_args(["--mode", "web", "--port", "8080"])
    assert args.mode == "web"
    assert args.port == 8080


# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------

def test_invalid_top_k():
    assert main(["--top-k", "0"]) == 2


def test_invalid_batch_size():
    assert main(["--batch-size", "0"]) == 2


def test_model_load_failure():
    with patch("main.build_pipeline", side_effect=RuntimeError("boom")):
        assert main(["--model", "nonexistent"]) == 1
