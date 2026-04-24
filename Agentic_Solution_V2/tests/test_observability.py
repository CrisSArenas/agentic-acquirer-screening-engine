"""Metrics accumulator and cost tracking tests."""
from __future__ import annotations

import pytest

from acquirer_engine.observability import RunMetrics
from acquirer_engine.schemas import TokenUsage


def test_metrics_accumulates_token_usage() -> None:
    metrics = RunMetrics()
    metrics.record_llm_call(TokenUsage(input_tokens=100, output_tokens=50))
    metrics.record_llm_call(TokenUsage(input_tokens=200, output_tokens=100))

    assert metrics.total_input_tokens == 300
    assert metrics.total_output_tokens == 150
    assert metrics.llm_call_count == 2
    # Cost math: (300 * 3 + 150 * 15) / 1M = 0.003150
    assert metrics.total_cost_usd == pytest.approx(0.00315, abs=1e-6)


def test_metrics_as_dict_ready_for_logging() -> None:
    metrics = RunMetrics()
    metrics.record_llm_call(TokenUsage(input_tokens=1000, output_tokens=500))
    metrics.record_validation_failure()
    metrics.record_validation_repair()

    d = metrics.as_dict()
    assert d["llm_calls"] == 1
    assert d["validation_failures"] == 1
    assert d["validation_repairs"] == 1
    assert d["input_tokens"] == 1000
    assert d["cost_usd"] > 0
