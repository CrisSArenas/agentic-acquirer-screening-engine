"""
Production-grade observability: structured logs, cost tracking, trace spans.

Responsibilities:
  1. Cost tracking: compute USD cost per LLM call from the Anthropic usage
     fields on each response. Aggregate into a RunMetrics accumulator and
     dump to logs at run end.
  2. Distributed-tracing-style spans: every LLM call gets a span with
     acquirer, model, latency, tokens, and cost attributes. Swap in
     OpenTelemetry exporters in production with a one-line change.
  3. Structured logging via structlog. Set LOG_JSON=true for JSON output
     that DataDog, Splunk, or CloudWatch can ingest directly.
"""
from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import structlog

from .config import settings
from .schemas import TokenUsage


# ==============================================================================
# LOGGING SETUP
# ==============================================================================

def configure_logging() -> None:
    """Configure structlog. Call once at app startup."""
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Shared processors
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.log_json:
        # Production: JSON to stdout — DataDog/CloudWatch ingest this directly.
        processors.append(structlog.processors.format_exc_info)
        processors.append(structlog.processors.JSONRenderer())
    else:
        # Dev: pretty console output.
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "acquirer_engine") -> Any:
    """Get a bound logger."""
    return structlog.get_logger(name)


# ==============================================================================
# METRICS ACCUMULATOR
# ==============================================================================

@dataclass
class RunMetrics:
    """Accumulates per-run metrics. One instance per identify-acquirers request.

    Explicitly NOT a global singleton — passing it through the call stack makes
    it testable and keeps concurrent runs isolated."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cost_usd: float = 0.0
    llm_call_count: int = 0
    llm_call_errors: int = 0
    tool_call_count: int = 0
    validation_failures: int = 0
    validation_repairs: int = 0
    cache_hits: int = 0
    start_time: float = field(default_factory=time.perf_counter)

    def record_llm_call(self, usage: TokenUsage) -> None:
        """Record usage from a single LLM call. Called from the Claude wrapper."""
        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens
        self.total_cache_creation_tokens += usage.cache_creation_tokens
        self.total_cache_read_tokens += usage.cache_read_tokens
        self.total_cost_usd += usage.cost_usd
        self.llm_call_count += 1

    def record_error(self) -> None:
        self.llm_call_errors += 1

    def record_validation_failure(self) -> None:
        self.validation_failures += 1

    def record_validation_repair(self) -> None:
        self.validation_repairs += 1

    def record_cache_hit(self) -> None:
        self.cache_hits += 1

    @property
    def duration_seconds(self) -> float:
        return round(time.perf_counter() - self.start_time, 3)

    def as_dict(self) -> dict[str, Any]:
        return {
            "duration_s": self.duration_seconds,
            "llm_calls": self.llm_call_count,
            "llm_errors": self.llm_call_errors,
            "tool_calls": self.tool_call_count,
            "validation_failures": self.validation_failures,
            "validation_repairs": self.validation_repairs,
            "cache_hits": self.cache_hits,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "cache_read_tokens": self.total_cache_read_tokens,
            "cost_usd": round(self.total_cost_usd, 4),
        }


# ==============================================================================
# TRACE SPANS — minimal OpenTelemetry-style context
# ==============================================================================

@contextmanager
def trace_span(name: str, **attributes: Any) -> Iterator[dict[str, Any]]:
    """Minimal span context manager. Logs start/end and computes duration.

    In production, swap this for opentelemetry.trace.get_tracer().start_span().
    The API surface is compatible — same attributes, same context manager shape.
    """
    log = get_logger()
    span_attrs: dict[str, Any] = {"span": name, **attributes}
    start = time.perf_counter()

    log.debug("span_start", **span_attrs)

    try:
        yield span_attrs
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        log.error("span_error", duration_ms=duration_ms, error=str(exc), **span_attrs)
        raise
    else:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        log.debug("span_end", duration_ms=duration_ms, **span_attrs)
