"""
Tool definitions for the Claude agent.

This is THE module that turns the system from a single-prompt wrapper into an
agentic system. Each tool has:
  - A Pydantic input schema (auto-converted to JSON Schema for Claude)
  - A concrete async implementation
  - A dispatch table used by the agent loop

When Claude decides to call `get_evidence_packet`, we route to the Python
function, execute it, and feed the result back into the message history.
The LLM chooses which tool to call based on what it needs — that's dynamic
routing.
"""
from __future__ import annotations

from typing import Any, Callable, Awaitable

import pandas as pd
from pydantic import BaseModel, Field

from .evidence import build_evidence_packets, select_relevant_transactions
from .schemas import EvidencePacket, RelevantTransaction, TargetProfile
from .scoring import (
    compute_conviction,
    rank_acquirers,
    score_acquirer,
)


# ==============================================================================
# TOOL INPUT SCHEMAS
# ==============================================================================

class ShortlistAcquirersInput(BaseModel):
    """Run Stage 1: score every acquirer and return the top N candidates."""
    target_sector: str = Field(..., description="e.g. 'Healthcare Services'")
    target_size_mm: float = Field(..., gt=0, description="Enterprise value in $M")
    top_n: int = Field(default=10, ge=1, le=25)


class GetEvidencePacketInput(BaseModel):
    """Retrieve the full structured evidence packet for ONE acquirer."""
    acquirer_name: str = Field(..., description="Exact name from the dataset")


class GetRelevantTransactionsInput(BaseModel):
    """Retrieve 2-3 prior transactions most relevant to the target."""
    acquirer_name: str
    target_sector: str
    target_size_mm: float = Field(..., gt=0)
    n: int = Field(default=3, ge=1, le=5)


class ComputeValuationEnvelopeInput(BaseModel):
    """Project EV range from acquirer's closed-deal multiples and a target EBITDA."""
    acquirer_name: str
    target_ebitda_mm: float = Field(..., gt=0, description="Estimated target EBITDA in $M")


class CheckConvictionGateInput(BaseModel):
    """Deterministic conviction lookup for an acquirer. Source of truth."""
    acquirer_name: str
    target_sector: str
    target_size_mm: float = Field(..., gt=0)


# ==============================================================================
# TOOL SCHEMAS EXPORTED TO CLAUDE
# Anthropic expects: [{name, description, input_schema}, ...]
# ==============================================================================

def _schema_for(model: type[BaseModel]) -> dict[str, Any]:
    """Convert a Pydantic model to the JSON Schema Claude expects.

    We strip 'title' because Anthropic's tool schema validator is pickier than
    generic JSON Schema and occasionally rejects them on nested refs."""
    schema = model.model_json_schema()
    schema.pop("title", None)
    # Anthropic tool schemas don't allow $defs at the top level in some versions
    return schema


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "shortlist_acquirers",
        "description": (
            "Run the deterministic scoring engine and return the top N acquirers "
            "ranked against the target. Use this FIRST to get your candidate list."
        ),
        "input_schema": _schema_for(ShortlistAcquirersInput),
    },
    {
        "name": "get_evidence_packet",
        "description": (
            "Fetch the full evidence packet for ONE acquirer: sector breakdown, "
            "deal sizes, closed-deal multiples, rationale tags, recency, close rate."
        ),
        "input_schema": _schema_for(GetEvidencePacketInput),
    },
    {
        "name": "get_relevant_transactions",
        "description": (
            "Fetch 2-3 verbatim prior transactions most relevant to the target. "
            "Use these to populate precedent_activity with transaction-ID citations."
        ),
        "input_schema": _schema_for(GetRelevantTransactionsInput),
    },
    {
        "name": "compute_valuation_envelope",
        "description": (
            "Project an EV range from the acquirer's closed-deal median and range "
            "EV/EBITDA multiples applied to a target EBITDA estimate. "
            "Returns low/median/high EV in $M."
        ),
        "input_schema": _schema_for(ComputeValuationEnvelopeInput),
    },
    {
        "name": "check_conviction_gate",
        "description": (
            "Deterministic conviction level for an acquirer. This is the BINDING "
            "source of truth — set conviction.level to whatever this returns, "
            "regardless of qualitative judgment."
        ),
        "input_schema": _schema_for(CheckConvictionGateInput),
    },
]


# ==============================================================================
# TOOL IMPLEMENTATIONS
# ==============================================================================

class ToolContext:
    """Injected into tool calls. Holds the dataframe + cached packets.

    Passing this explicitly (instead of module globals) makes tools independently
    testable and supports concurrent runs with different datasets."""

    def __init__(self, df: pd.DataFrame, target: TargetProfile):
        self.df = df
        self.target = target
        # Cache: build packets once per run, not per tool call.
        self._packets: list[EvidencePacket] | None = None
        self._deal_sizes: dict[str, list[float]] | None = None

    @property
    def packets(self) -> list[EvidencePacket]:
        if self._packets is None:
            self._packets = build_evidence_packets(self.df, self.target)
        return self._packets

    @property
    def deal_sizes(self) -> dict[str, list[float]]:
        if self._deal_sizes is None:
            self._deal_sizes = {
                str(name): grp["deal_size_mm"].dropna().tolist()
                for name, grp in self.df.groupby("acquirer")
            }
        return self._deal_sizes

    def get_packet(self, name: str) -> EvidencePacket | None:
        for p in self.packets:
            if p.acquirer_name == name:
                return p
        return None


async def tool_shortlist_acquirers(ctx: ToolContext, inp: ShortlistAcquirersInput) -> dict:
    target = TargetProfile(sector=inp.target_sector, size_mm=inp.target_size_mm)
    top = rank_acquirers(ctx.packets, target, ctx.deal_sizes, top_n=inp.top_n)
    return {
        "target": target.model_dump(),
        "results": [
            {
                "rank": i + 1,
                "acquirer_name": r.acquirer_name,
                "total_score": r.total_score,
                "conviction": r.conviction,
                "acquirer_type": r.packet.acquirer_type,
                "total_deals": r.packet.total_deals,
                "components": r.components.model_dump(),
            }
            for i, r in enumerate(top)
        ],
    }


async def tool_get_evidence_packet(ctx: ToolContext, inp: GetEvidencePacketInput) -> dict:
    packet = ctx.get_packet(inp.acquirer_name)
    if packet is None:
        return {"error": f"Acquirer '{inp.acquirer_name}' not found in dataset"}
    return packet.model_dump(exclude={"acquirer_type"} if False else None)


async def tool_get_relevant_transactions(
    ctx: ToolContext, inp: GetRelevantTransactionsInput
) -> dict:
    target = TargetProfile(sector=inp.target_sector, size_mm=inp.target_size_mm)
    txs = select_relevant_transactions(ctx.df, inp.acquirer_name, target, n=inp.n)
    return {"transactions": [t.model_dump() for t in txs]}


async def tool_compute_valuation_envelope(
    ctx: ToolContext, inp: ComputeValuationEnvelopeInput
) -> dict:
    packet = ctx.get_packet(inp.acquirer_name)
    if packet is None:
        return {"error": f"Acquirer '{inp.acquirer_name}' not found"}

    grp = ctx.df[(ctx.df["acquirer"] == inp.acquirer_name) & (ctx.df["outcome"] == "Closed")]
    ev_ebitda_values = grp["ev_ebitda_multiple"].dropna().tolist()

    if not ev_ebitda_values:
        return {
            "error": "No closed-deal EV/EBITDA multiples available for this acquirer",
            "num_closed_deals": 0,
        }

    median_multiple = float(pd.Series(ev_ebitda_values).median())
    low_multiple = float(min(ev_ebitda_values))
    high_multiple = float(max(ev_ebitda_values))

    return {
        "target_ebitda_mm": inp.target_ebitda_mm,
        "num_closed_deals": len(ev_ebitda_values),
        "median_ev_ebitda_multiple": round(median_multiple, 2),
        "multiple_range": [round(low_multiple, 2), round(high_multiple, 2)],
        "implied_ev_mm": {
            "at_median": round(inp.target_ebitda_mm * median_multiple, 1),
            "low": round(inp.target_ebitda_mm * low_multiple, 1),
            "high": round(inp.target_ebitda_mm * high_multiple, 1),
        },
    }


async def tool_check_conviction_gate(
    ctx: ToolContext, inp: CheckConvictionGateInput
) -> dict:
    packet = ctx.get_packet(inp.acquirer_name)
    if packet is None:
        return {"error": f"Acquirer '{inp.acquirer_name}' not found"}

    target = TargetProfile(sector=inp.target_sector, size_mm=inp.target_size_mm)
    level = compute_conviction(packet, target)

    # Also return the specific gate-pass/fail booleans so the LLM can cite them.
    from .scoring import get_adjacency
    adj = get_adjacency(target.sector)
    exact = packet.sector_distribution.get(target.sector, 0)
    adjacent = sum(packet.sector_distribution.get(s, 0) for s in adj["adjacent"])

    return {
        "conviction_level": level,
        "gate_a_sector": {
            "exact_deals": exact,
            "adjacent_deals": adjacent,
            "passes": (exact >= 2) or (adjacent >= 4 and exact >= 1),
        },
        "gate_b_size_band": {
            "deals_in_band": packet.deal_size_stats.deals_in_target_band,
            "passes": packet.deal_size_stats.deals_in_target_band >= 1,
        },
        "gate_c_recency": {
            "most_recent_year": packet.most_recent_deal_year,
            "passes": packet.most_recent_deal_year >= 2022,
        },
    }


# ==============================================================================
# DISPATCH TABLE — agent loop calls this
# ==============================================================================

ToolFn = Callable[[ToolContext, BaseModel], Awaitable[dict]]

TOOL_REGISTRY: dict[str, tuple[type[BaseModel], ToolFn]] = {
    "shortlist_acquirers": (ShortlistAcquirersInput, tool_shortlist_acquirers),  # type: ignore[dict-item]
    "get_evidence_packet": (GetEvidencePacketInput, tool_get_evidence_packet),  # type: ignore[dict-item]
    "get_relevant_transactions": (GetRelevantTransactionsInput, tool_get_relevant_transactions),  # type: ignore[dict-item]
    "compute_valuation_envelope": (ComputeValuationEnvelopeInput, tool_compute_valuation_envelope),  # type: ignore[dict-item]
    "check_conviction_gate": (CheckConvictionGateInput, tool_check_conviction_gate),  # type: ignore[dict-item]
}


async def dispatch_tool(ctx: ToolContext, name: str, raw_input: dict) -> dict:
    """Validate tool input with Pydantic, then call the implementation.

    If validation fails, return an error dict that the LLM can recover from —
    never raise, never let a bad tool call crash the whole agent."""
    if name not in TOOL_REGISTRY:
        return {"error": f"Unknown tool: {name}"}

    input_model, fn = TOOL_REGISTRY[name]
    try:
        validated = input_model.model_validate(raw_input)
    except Exception as e:
        return {"error": f"Invalid arguments for {name}: {e}"}

    return await fn(ctx, validated)
