"""
Retry logic for LLM calls. Handles rate limits and transient failures cleanly.

Uses tenacity — the Python standard for retry policies. Specific exception
types, with before-sleep logging so you can see in the logs exactly when and
why a retry happened.

Key design choice: when the API returns an output-tokens-per-minute rate
limit (a sliding 60-second window), a 1-second retry is guaranteed to fail
again because the capacity hasn't reset. We use a longer wait (30s min,
60s max) for 429s specifically, and a short exponential backoff for
transient connection errors.
"""
from __future__ import annotations

from anthropic import APIConnectionError, APIStatusError, RateLimitError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_chain,
    wait_exponential,
    wait_fixed,
)
import logging

from .config import settings

# Exceptions we retry on. Everything else bubbles up immediately.
RETRYABLE_EXCEPTIONS = (
    RateLimitError,
    APIConnectionError,
    APIStatusError,  # 5xx errors
)

# Standard Python logger for tenacity integration (tenacity uses stdlib logging).
_tenacity_logger = logging.getLogger("acquirer_engine.retry")


def _smart_wait(retry_state) -> float:
    """Different backoff based on exception type.

    - RateLimitError (429): 30s, 45s, 60s -- because output-token-per-minute
      limits reset on a 60s window. A 1s retry is guaranteed to fail again.
    - Connection errors: 1s, 2s, 4s -- typically transient network blips.
    """
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    attempt = retry_state.attempt_number

    if isinstance(exc, RateLimitError):
        # Long waits for output-token-per-minute rate limits
        return min(30 * attempt, 60)
    # Fast exponential for transient connection errors
    return min(2 ** (attempt - 1), 8)


# Decorator applied to every LLM call.
llm_retry = retry(
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    wait=_smart_wait,
    stop=stop_after_attempt(settings.retry_max_attempts),
    before_sleep=before_sleep_log(_tenacity_logger, logging.WARNING),
    reraise=True,
)
