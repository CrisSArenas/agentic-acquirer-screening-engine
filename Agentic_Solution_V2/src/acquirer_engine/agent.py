"""
The agent: tool calling, dynamic routing, validation, repair loop.

This module is the production-grade orchestrator. It implements the full
agentic pattern:

  - Tool use with Pydantic-validated input schemas (see tools.py)
  - Dynamic routing: each response's stop_reason drives the next step
  - Structured output validation via Pydantic on tool inputs and final output
  - Per-step error handling: tool failures return error dicts, never raise
    past the agent boundary, so one bad tool call never crashes a run
  - Dependency-injected AsyncAnthropic client (one instance per app, reused
    across every call — avoids per-call TLS handshake and connection-pool churn)
  - Real async parallelism via asyncio.gather + Semaphore, tuned to the
    provider's rate limit tier
  - Per-call cost tracking computed from resp.usage.input_tokens and
    resp.usage.output_tokens, aggregated into RunMetrics
  - Retry with typed backoff (see retry.py): long waits for rate limits,
    short waits for transient connection errors
  - Validation repair loop: on Pydantic failure, send the specific errors
    back to the model with a directive to fix
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import pandas as pd
from anthropic import AsyncAnthropic
from pydantic import ValidationError

from .config import settings
from .observability import RunMetrics, get_logger, trace_span
from .retry import llm_retry
from .schemas import (
    AcquirerRationale,
    RationaleSet,
    ScoredAcquirer,
    TargetProfile,
    TokenUsage,
)
from .scoring import rank_acquirers
from .tools import (
    TOOL_SCHEMAS,
    ToolContext,
    dispatch_tool,
)
from .evidence import build_evidence_packets

log = get_logger("agent")


# ==============================================================================
# PROMPTS
# ==============================================================================

def build_system_prompt(target: TargetProfile) -> str:
    """Dynamic system prompt with target profile + gating rules injected."""
    from .scoring import get_adjacency
    adj = get_adjacency(target.sector)
    adj_list = ", ".join(sorted(adj["adjacent"])) or "(none defined)"

    return f"""You are a senior M&A analyst at William Blair's Investment Banking \
division, drafting a one-page acquirer rationale that a Vice President will hand \
directly to a Managing Director without editing.

TARGET PROFILE
- Sector: {target.sector}
- Enterprise value: ~${target.size_mm:.0f}M
- Size band for analysis: ${target.size_lo_mm:.0f}M - ${target.size_hi_mm:.0f}M
- Adjacent sub-sectors: {adj_list}
- Geography: {target.geography}

IMPORTANT: The user message contains everything you need — the evidence packet, \
the deterministic conviction level, and pre-selected relevant transactions. \
Use tools ONLY if you need an extra computation (e.g. compute_valuation_envelope). \
Do NOT emit a <thinking> block. Emit the JSON directly.

HARD RULES
1. CONVICTION IS DETERMINISTIC. Use the conviction level stated in the user \
message. Never change it. Qualitative concerns go in risk_flags, not conviction.
2. Every precedent reference cites its transaction_id (e.g. MA-2023-0142).
3. No vague marketing language. Banned phrases: "world-class", "track record \
of", "well-positioned", "industry leader", "best-in-class", "proven ability", \
"deep expertise", "robust". Note: "strategic acquirer" is FINE — it's factual \
IB terminology. Use numbers and transaction IDs over adjectives.
4. Numbers over adjectives. Prefer "3 Healthcare Services deals at median \
13.2x EV/EBITDA" over "active healthcare investor at attractive multiples".
5. Honest weakness. If evidence is thin, say so plainly.
6. No hallucinated facts. Only what's in the evidence packet and tool results.
7. Valuation format: "At [X.Xx] median EV/EBITDA on [N] closed deals, a \
target generating ~$[Y]M EBITDA implies ~$[Z]M EV; the range of [low]x-[high]x \
implies a $[low]M-$[high]M envelope." Work forward from multiples to EV.
8. BE CONCISE. Tight prose, no filler, no repetition.
9. Output: ONE valid JSON object, NOTHING ELSE. First character is `{{`, last \
character is `}}`. Schema:

{{
  "acquirer_overview": "2-3 sentences, ~300 chars. Cite deal count, type, sector, recency.",
  "strategic_fit_thesis": "3-4 sentences, ~500 chars. Tie to tags + size history.",
  "precedent_activity": "2-3 bullets (use \\n- separator), ~400 chars total. Each bullet cites its transaction_id.",
  "valuation_context": "2 sentences, ~300 chars. MUST include Nx AND $EV.",
  "risk_flags": "2-3 bullets (use \\n- separator), ~300 chars total.",
  "conviction": {{"level": "High|Medium|Low", "rationale": "1-2 sentences, ~300 chars. Cite which gates passed/failed."}}
}}

Keep total output UNDER 1000 tokens.
"""


USER_PROMPT_TEMPLATE = """Produce the one-page rationale for:

ACQUIRER: {acquirer_name}  (ranked #{rank})
TYPE: {acquirer_type}
DETERMINISTIC CONVICTION (use this exact value): {conviction}
TARGET: {sector} at ~${size:.0f}M EV

==================== EVIDENCE PACKET ====================
{evidence_json}

==================== RELEVANT PRIOR TRANSACTIONS ====================
{transactions_json}

==================== CONVICTION GATE DETAIL ====================
{gate_detail_json}

=========================================================
Emit the JSON rationale now. All the data you need is above. Only call a tool \
if you need to compute an EV envelope from a target EBITDA you're estimating."""


# ==============================================================================
# CLAUDE WRAPPER — retries + observability
# ==============================================================================

@llm_retry
async def _call_claude(
    client: AsyncAnthropic,
    system: str,
    messages: list[dict],
    tools: list[dict],
    metrics: RunMetrics,
    tag: str,
) -> Any:
    """Single Claude call with retry + metrics. Client is INJECTED, not created."""
    with trace_span("llm_call", tag=tag, model=settings.model) as span:
        resp = await client.messages.create(
            model=settings.model,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            system=system,
            tools=tools,
            messages=messages,
        )

        # Real token usage. This is the fix for "cost tracking returns zeros".
        usage = TokenUsage(
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            cache_creation_tokens=getattr(resp.usage, "cache_creation_input_tokens", 0) or 0,
            cache_read_tokens=getattr(resp.usage, "cache_read_input_tokens", 0) or 0,
        )
        metrics.record_llm_call(usage)

        span["tokens_in"] = usage.input_tokens
        span["tokens_out"] = usage.output_tokens
        span["cost_usd"] = usage.cost_usd
        span["stop_reason"] = resp.stop_reason

        log.info(
            "llm_call_completed",
            tag=tag,
            stop_reason=resp.stop_reason,
            tokens_in=usage.input_tokens,
            tokens_out=usage.output_tokens,
            cost_usd=usage.cost_usd,
        )
        return resp


# ==============================================================================
# AGENT LOOP — per-acquirer
# ==============================================================================

MAX_AGENT_ITERATIONS = 4  # Safety cap. With pre-loaded evidence, usually 1-2.


async def _extract_final_json(resp: Any) -> dict | None:
    """Pull the final JSON object from the assistant's text content.

    The model should emit JSON directly. We strip any <thinking> blocks that
    slip through, then locate the JSON object by brace matching. Robust to
    minor formatting quirks (markdown fences, leading prose)."""
    import re

    text_blocks = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    if not text_blocks:
        return None

    raw = "\n".join(text_blocks).strip()

    # Strip <thinking>...</thinking> blocks if present (even partial)
    raw = re.sub(r"<thinking>.*?</thinking>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"<thinking>.*", "", raw, flags=re.DOTALL)  # unclosed thinking

    # Strip markdown code fences if the model added them
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```\s*$", "", raw)

    # Find the first `{` and match braces to find the complete object.
    # More robust than greedy regex — handles nested braces correctly.
    start = raw.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    end = -1
    for i in range(start, len(raw)):
        ch = raw[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end == -1:
        # Truncated JSON — no matching close brace. Nothing to salvage.
        return None

    candidate = raw[start:end]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


async def run_single_acquirer_agent(
    client: AsyncAnthropic,
    ctx: ToolContext,
    scored: ScoredAcquirer,
    rank: int,
    target: TargetProfile,
    metrics: RunMetrics,
) -> AcquirerRationale:
    """Run the tool-calling loop for ONE acquirer. Returns a validated rationale.

    Dynamic routing happens here: on each Claude response we inspect stop_reason
    and decide whether to dispatch tools and loop, or exit and validate.

    Performance optimization: we pre-load the evidence packet, relevant
    transactions, and conviction gate detail into the user prompt. This means
    the model usually completes in ONE iteration (emits JSON directly) instead
    of 3-4 iterations of back-and-forth tool calls. Tools remain available for
    additional computations (e.g. EV envelope) but are rarely needed.
    """
    system = build_system_prompt(target)

    # Pre-compute the data the model needs — avoids 2-3 wasted tool-call round-trips
    from .tools import (
        CheckConvictionGateInput,
        GetEvidencePacketInput,
        GetRelevantTransactionsInput,
        tool_check_conviction_gate,
        tool_get_evidence_packet,
        tool_get_relevant_transactions,
    )

    evidence_dict = await tool_get_evidence_packet(
        ctx, GetEvidencePacketInput(acquirer_name=scored.acquirer_name)
    )
    transactions_dict = await tool_get_relevant_transactions(
        ctx,
        GetRelevantTransactionsInput(
            acquirer_name=scored.acquirer_name,
            target_sector=target.sector,
            target_size_mm=target.size_mm,
            n=3,
        ),
    )
    gate_dict = await tool_check_conviction_gate(
        ctx,
        CheckConvictionGateInput(
            acquirer_name=scored.acquirer_name,
            target_sector=target.sector,
            target_size_mm=target.size_mm,
        ),
    )

    messages: list[dict] = [{
        "role": "user",
        "content": USER_PROMPT_TEMPLATE.format(
            acquirer_name=scored.acquirer_name,
            acquirer_type=scored.packet.acquirer_type,
            rank=rank,
            conviction=scored.conviction,
            sector=target.sector,
            size=target.size_mm,
            evidence_json=json.dumps(evidence_dict, indent=2, default=str),
            transactions_json=json.dumps(transactions_dict, indent=2, default=str),
            gate_detail_json=json.dumps(gate_dict, indent=2, default=str),
        )
    }]

    for iteration in range(MAX_AGENT_ITERATIONS):
        resp = await _call_claude(
            client=client,
            system=system,
            messages=messages,
            tools=TOOL_SCHEMAS,
            metrics=metrics,
            tag=f"rationale:{scored.acquirer_name}:iter{iteration}",
        )

        # DYNAMIC ROUTING
        if resp.stop_reason == "end_turn":
            break

        if resp.stop_reason == "tool_use":
            tool_use_blocks = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
            messages.append({"role": "assistant", "content": resp.content})

            tool_results = []
            for block in tool_use_blocks:
                metrics.tool_call_count += 1
                result = await dispatch_tool(ctx, block.name, block.input)
                log.debug("tool_result", tool=block.name, acquirer=scored.acquirer_name)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, default=str),
                })
            messages.append({"role": "user", "content": tool_results})
            continue

        # Unexpected stop reason — bail
        log.warning("unexpected_stop", stop_reason=resp.stop_reason, acquirer=scored.acquirer_name)
        break

    # Extract + validate final JSON
    raw_json = await _extract_final_json(resp)
    if raw_json is None:
        raise RuntimeError(f"Could not extract rationale JSON for {scored.acquirer_name}")

    raw_json.setdefault("acquirer_name", scored.acquirer_name)
    raw_json.setdefault("rank", rank)

    # Force conviction to the deterministic value. Protects against the model
    # trying to override the gate even with the prompt telling it not to.
    if isinstance(raw_json.get("conviction"), dict):
        raw_json["conviction"]["level"] = scored.conviction

    try:
        return AcquirerRationale.model_validate(raw_json)
    except ValidationError as e:
        metrics.record_validation_failure()
        log.warning(
            "validation_failed_attempting_repair",
            acquirer=scored.acquirer_name,
            errors=e.errors(),
        )
        return await _repair_rationale(
            client=client,
            scored=scored,
            rank=rank,
            target=target,
            bad_json=raw_json,
            validation_error=e,
            metrics=metrics,
        )


async def _repair_rationale(
    client: AsyncAnthropic,
    scored: ScoredAcquirer,
    rank: int,
    target: TargetProfile,
    bad_json: dict,
    validation_error: ValidationError,
    metrics: RunMetrics,
) -> AcquirerRationale:
    """One-shot repair: send the validation error back, ask for a fix.

    Intentionally minimal — we don't need to re-send the full evidence packet,
    just the bad JSON and the specific errors. Lower token cost, faster repair.
    """
    system = build_system_prompt(target)

    # Build a compact error summary the model can act on directly
    error_lines = []
    for err in validation_error.errors():
        loc = ".".join(str(x) for x in err.get("loc", []))
        error_lines.append(f"  - {loc}: {err.get('msg', '')}")
    error_summary = "\n".join(error_lines)

    messages = [
        {
            "role": "user",
            "content": (
                f"Your previous rationale JSON for {scored.acquirer_name} (rank #{rank}, "
                f"conviction: {scored.conviction}) failed Pydantic validation:\n\n"
                f"{error_summary}\n\n"
                f"PREVIOUS JSON:\n{json.dumps(bad_json, indent=2)}\n\n"
                f"Return ONLY the corrected JSON object. Keep the same content, "
                f"just fix the specific errors listed. No <thinking>, no prose. "
                f"First character must be '{{', last character must be '}}'. "
                f"Keep conviction.level = \"{scored.conviction}\"."
            ),
        },
    ]

    resp = await _call_claude(
        client=client, system=system, messages=messages, tools=[],
        metrics=metrics, tag=f"repair:{scored.acquirer_name}",
    )
    repaired = await _extract_final_json(resp)
    if repaired is None:
        raise RuntimeError(f"Repair failed for {scored.acquirer_name}")

    repaired.setdefault("acquirer_name", scored.acquirer_name)
    repaired.setdefault("rank", rank)
    if isinstance(repaired.get("conviction"), dict):
        repaired["conviction"]["level"] = scored.conviction

    metrics.record_validation_repair()
    return AcquirerRationale.model_validate(repaired)


# ==============================================================================
# ORCHESTRATOR — async parallel run for all 10 acquirers
# ==============================================================================

async def identify_acquirers(
    df: pd.DataFrame,
    target: TargetProfile,
    client: AsyncAnthropic | None = None,
    top_n: int = 10,
) -> RationaleSet:
    """Full pipeline: score → top N → parallel per-acquirer agents → validated set.

    Client is optional so tests can inject a mock. In production, an app-level
    client is instantiated once and reused — NEVER per-call."""
    metrics = RunMetrics()

    # Client injection: one instance for the whole run.
    owned_client = client is None
    if client is None:
        client = AsyncAnthropic(api_key=settings.api_key)

    try:
        with trace_span("identify_acquirers", sector=target.sector, size_mm=target.size_mm):
            # Stage 1: deterministic shortlist
            with trace_span("stage_1_scoring"):
                packets = build_evidence_packets(df, target)
                deal_sizes = {
                    str(name): grp["deal_size_mm"].dropna().tolist()
                    for name, grp in df.groupby("acquirer")
                }
                top = rank_acquirers(packets, target, deal_sizes, top_n=top_n)
                log.info(
                    "shortlist_produced",
                    count=len(top),
                    top_3=[r.acquirer_name for r in top[:3]],
                )

            # Stage 2: parallel per-acquirer agents
            # asyncio.gather with Semaphore — real concurrency, bounded by
            # settings.max_concurrent_requests so we stay under the rate limit.
            ctx = ToolContext(df, target)
            semaphore = asyncio.Semaphore(settings.max_concurrent_requests)

            async def _bounded_run(scored: ScoredAcquirer, rank: int) -> AcquirerRationale:
                async with semaphore:
                    return await run_single_acquirer_agent(
                        client=client, ctx=ctx, scored=scored,
                        rank=rank, target=target, metrics=metrics,
                    )

            with trace_span("stage_2_rationales", n_acquirers=len(top)):
                rationales = await asyncio.gather(
                    *[_bounded_run(s, i + 1) for i, s in enumerate(top)],
                    return_exceptions=True,
                )

            # Partition successes and failures
            validated: list[AcquirerRationale] = []
            for r in rationales:
                if isinstance(r, Exception):
                    metrics.record_error()
                    log.error("rationale_failed", error=str(r))
                else:
                    validated.append(r)

            # Re-rank so gaps don't break the contiguity validator
            for new_rank, r in enumerate(sorted(validated, key=lambda x: x.rank), start=1):
                r.rank = new_rank

    finally:
        if owned_client:
            await client.close()

    result = RationaleSet(
        target=target,
        rationales=validated,
        total_cost_usd=round(metrics.total_cost_usd, 4),
        total_duration_seconds=metrics.duration_seconds,
        total_input_tokens=metrics.total_input_tokens,
        total_output_tokens=metrics.total_output_tokens,
        cache_hits=metrics.cache_hits,
    )

    log.info("run_complete", **metrics.as_dict())
    return result
