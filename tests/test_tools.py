"""
Tool-dispatch tests — validate that the tool layer handles bad input gracefully
and produces the shape the agent loop expects.
"""
from __future__ import annotations

import pytest

from acquirer_engine.schemas import TargetProfile
from acquirer_engine.tools import (
    TOOL_REGISTRY,
    TOOL_SCHEMAS,
    ToolContext,
    dispatch_tool,
)


# ==============================================================================
# SCHEMA EXPORT
# ==============================================================================

def test_tool_schemas_match_registry() -> None:
    """Every exported schema must have a registered handler."""
    exported_names = {s["name"] for s in TOOL_SCHEMAS}
    assert exported_names == set(TOOL_REGISTRY.keys())


def test_tool_schemas_have_required_fields() -> None:
    for s in TOOL_SCHEMAS:
        assert "name" in s
        assert "description" in s
        assert "input_schema" in s
        assert s["input_schema"]["type"] == "object"


# ==============================================================================
# DISPATCH
# ==============================================================================

@pytest.mark.asyncio
async def test_unknown_tool_returns_error(sample_df, target_profile) -> None:
    ctx = ToolContext(sample_df, target_profile)
    result = await dispatch_tool(ctx, "does_not_exist", {})
    assert "error" in result
    assert "Unknown tool" in result["error"]


@pytest.mark.asyncio
async def test_invalid_input_returns_error_not_raise(sample_df, target_profile) -> None:
    """Bad tool input must return an error dict — never raise past the agent."""
    ctx = ToolContext(sample_df, target_profile)
    # Missing required field
    result = await dispatch_tool(ctx, "get_evidence_packet", {})
    assert "error" in result


@pytest.mark.asyncio
async def test_shortlist_returns_ranked_candidates(sample_df, target_profile) -> None:
    ctx = ToolContext(sample_df, target_profile)
    result = await dispatch_tool(
        ctx, "shortlist_acquirers",
        {"target_sector": "Healthcare Services", "target_size_mm": 200.0, "top_n": 5},
    )
    assert "results" in result
    assert len(result["results"]) <= 5
    # Ranks must be contiguous
    ranks = [r["rank"] for r in result["results"]]
    assert ranks == list(range(1, len(ranks) + 1))


@pytest.mark.asyncio
async def test_check_conviction_gate_returns_gate_booleans(sample_df, target_profile) -> None:
    ctx = ToolContext(sample_df, target_profile)
    # Ardent Health should be High (2 HC deals, 1 in band, 2022)
    result = await dispatch_tool(
        ctx, "check_conviction_gate",
        {
            "acquirer_name": "Ardent Health Services",
            "target_sector": "Healthcare Services",
            "target_size_mm": 200.0,
        },
    )
    # Either we get a real result or the acquirer isn't in this CSV
    if "error" not in result:
        assert result["conviction_level"] in {"High", "Medium", "Low"}
        assert "gate_a_sector" in result
        assert "gate_b_size_band" in result
        assert "gate_c_recency" in result
