"""Unit tests for CLI-layer functions (cli.py).

The DummyPipeline mock simulates both top_k-capable and legacy
pipeline behaviour so tests run without downloading real models.
"""

from cli import analyze_texts, format_outputs


class DummyPipeline:
    def __init__(self, supports_top_k: bool = True):
        self.supports_top_k = supports_top_k

    def __call__(self, texts, top_k=None):
        if not self.supports_top_k and top_k is not None and top_k > 1:
            raise TypeError("top_k not supported")

        if top_k is None or top_k == 1:
            return [{"label": "POSITIVE", "score": 0.9} for _ in texts]

        return [
            [
                {"label": "POSITIVE", "score": 0.9},
                {"label": "NEGATIVE", "score": 0.1},
            ]
            for _ in texts
        ]


def test_analyze_texts_top_k_supported():
    nlp = DummyPipeline(supports_top_k=True)
    results, effective_top_k = analyze_texts(nlp, ["hello"], 2)
    assert effective_top_k == 2
    assert isinstance(results[0], list)


def test_analyze_texts_top_k_fallback():
    nlp = DummyPipeline(supports_top_k=False)
    results, effective_top_k = analyze_texts(nlp, ["hello"], 2)
    assert effective_top_k == 1
    assert isinstance(results[0], dict)


def test_analyze_texts_empty():
    nlp = DummyPipeline()
    results, effective_top_k = analyze_texts(nlp, [], 1)
    assert results == []
    assert effective_top_k == 1


def test_format_outputs_top1():
    outputs = format_outputs(["hello"], [{"label": "POSITIVE", "score": 0.9}], 1)
    assert "POSITIVE" in outputs[0]
    assert "0.900" in outputs[0]


def test_format_outputs_top_k():
    results = [
        [
            {"label": "POSITIVE", "score": 0.9},
            {"label": "NEGATIVE", "score": 0.1},
        ]
    ]
    outputs = format_outputs(["great movie"], results, 2)
    assert "POSITIVE" in outputs[0]
    assert "NEGATIVE" in outputs[0]
