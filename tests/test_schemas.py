"""
Schema validation tests — Pydantic validators enforce the hard rules as type
constraints. These tests prove that malformed LLM output is rejected before it
reaches the UI.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from acquirer_engine.schemas import (
    AcquirerRationale,
    Conviction,
    TargetProfile,
    TokenUsage,
)


# ==============================================================================
# TARGET PROFILE
# ==============================================================================

def test_target_profile_size_band_half_to_double() -> None:
    t = TargetProfile(sector="Healthcare Services", size_mm=200.0)
    assert t.size_lo_mm == 100.0
    assert t.size_hi_mm == 400.0


def test_target_profile_rejects_zero_size() -> None:
    with pytest.raises(ValidationError):
        TargetProfile(sector="Healthcare Services", size_mm=0.0)


# ==============================================================================
# ACQUIRER RATIONALE — field validators
# ==============================================================================

def _valid_rationale_kwargs() -> dict:
    return dict(
        acquirer_name="Test Corp",
        rank=1,
        acquirer_overview="Test Corp closed 5 deals between 2020-2023 with a median "
                          "EV/EBITDA of 13.2x and a 100% close rate.",
        strategic_fit_thesis="The acquirer's sector concentration and recurring "
                             "Geographic Expansion tags align with acquiring a regional "
                             "$200M target. Size-band alignment is strong given 3 of 5 "
                             "deals fell in the $100M-$400M window. The target's EBITDA "
                             "margin profile fits the acquirer's prior mid-market pattern.",
        precedent_activity="- Acme Co (2022, $180M, 13.0x EV/EBITDA, Bolt-on) (MA-2022-0145)\n"
                           "- Beta Inc (2023, $220M, 14.2x EV/EBITDA, Platform) (MA-2023-0298)",
        valuation_context="At 13.2x median EV/EBITDA on 5 closed deals, a target "
                          "generating ~$15M EBITDA implies ~$198M EV; range 12.4x-14.2x "
                          "implies $186M-$213M envelope.",
        risk_flags="- Size step-up risk: median deal size $160M vs target $200M.\n"
                   "- Integration: 3 concurrent active processes in the portfolio.",
        conviction=Conviction(
            level="High",
            rationale="All three gating conditions met: 5 sector deals, 3 in band, 2023 recency.",
        ),
    )


def test_valid_rationale_passes() -> None:
    r = AcquirerRationale(**_valid_rationale_kwargs())
    assert r.conviction.level == "High"


def test_precedent_without_transaction_id_rejected() -> None:
    kwargs = _valid_rationale_kwargs()
    kwargs["precedent_activity"] = "- Acme Co (2022, $180M, Bolt-on) — just relevant."
    with pytest.raises(ValidationError, match="transaction_id"):
        AcquirerRationale(**kwargs)


def test_banned_marketing_phrase_rejected() -> None:
    kwargs = _valid_rationale_kwargs()
    kwargs["acquirer_overview"] = (
        "A leading strategic acquirer with a track record of successful "
        "healthcare acquisitions at premier multiples in the space."
    )
    with pytest.raises(ValidationError, match="Marketing language"):
        AcquirerRationale(**kwargs)


def test_valuation_without_multiple_rejected() -> None:
    kwargs = _valid_rationale_kwargs()
    kwargs["valuation_context"] = "The acquirer pays premium prices across the board."
    with pytest.raises(ValidationError, match="multiple"):
        AcquirerRationale(**kwargs)


# ==============================================================================
# TOKEN USAGE — cost tracking that actually works
# ==============================================================================

def test_token_cost_is_nonzero_for_real_usage() -> None:
    """Regression: cost must compute to a real positive USD value for real
    token counts. Catches silent drops back to zero if the pricing constants
    or arithmetic ever get broken."""
    usage = TokenUsage(input_tokens=1000, output_tokens=500)
    cost = usage.cost_usd
    assert cost > 0.0
    # 1000 input @ $3/M + 500 output @ $15/M = 0.003 + 0.0075 = 0.0105
    assert cost == pytest.approx(0.0105, abs=1e-4)


def test_token_cost_zero_only_for_zero_tokens() -> None:
    usage = TokenUsage(input_tokens=0, output_tokens=0)
    assert usage.cost_usd == 0.0
