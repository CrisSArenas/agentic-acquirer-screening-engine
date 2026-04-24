"""
Pydantic schemas — the type-safe contract between every layer.

Every LLM response is validated against these schemas. Field-level validators
enforce the hard output rules (transaction-ID citation, banned marketing
phrases, valuation format). If validation fails, the agent's repair loop
(in agent.py) sends the ValidationError back to the model with a directive
to fix it before giving up.
"""
from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ==============================================================================
# INPUT SCHEMAS — what the user gives us
# ==============================================================================

class TargetProfile(BaseModel):
    """The target company a banker is trying to find buyers for."""

    sector: str = Field(..., min_length=1, description="Target sector, e.g. 'Healthcare Services'")
    size_mm: float = Field(..., gt=0, description="Target enterprise value in $M")
    geography: str = Field(default="Regional", description="Geographic profile")
    profile_notes: str = Field(
        default="Mid-market, private, strong EBITDA margins",
        description="Free-text profile details"
    )

    @property
    def size_lo_mm(self) -> float:
        """Lower bound of size band: 0.5x target EV."""
        return round(self.size_mm * 0.5, 1)

    @property
    def size_hi_mm(self) -> float:
        """Upper bound of size band: 2x target EV."""
        return round(self.size_mm * 2.0, 1)


# ==============================================================================
# EVIDENCE PACKET — what the scoring layer produces for each acquirer
# ==============================================================================

class DealSizeStats(BaseModel):
    min_mm: float
    median_mm: float
    max_mm: float
    deals_in_target_band: int = Field(..., ge=0)
    target_band: str


class ClosedDealMultiples(BaseModel):
    median_ev_ebitda: Optional[float] = None
    median_ev_revenue: Optional[float] = None
    num_closed_deals: int = Field(..., ge=0)


class EvidencePacket(BaseModel):
    """Structured evidence for ONE acquirer. Computed deterministically from CSV."""

    acquirer_name: str
    acquirer_type: Literal["Strategic", "Financial Sponsor"]
    total_deals: int = Field(..., ge=1)
    sector_distribution: dict[str, int]
    sub_sector_distribution: dict[str, int]
    deal_size_stats: DealSizeStats
    closed_deal_multiples: ClosedDealMultiples
    deal_type_mix: dict[str, int]
    geography_mix: dict[str, int]
    top_strategic_rationale_tags: list[dict]  # [{tag: str, count: int}, ...]
    most_recent_deal_year: int
    close_rate: float = Field(..., ge=0.0, le=1.0)


class RelevantTransaction(BaseModel):
    """Full row from the CSV for a prior transaction."""

    transaction_id: str = Field(..., pattern=r"^MA-\d{4}-\d{4}$")
    target_company: str
    sector: str
    sub_sector: str
    deal_year: int
    deal_type: str
    geography: str
    deal_size_mm: Optional[float] = None
    ev_ebitda_multiple: Optional[float] = None
    ev_revenue_multiple: Optional[float] = None
    ebitda_margin_pct: Optional[float] = None
    outcome: str
    strategic_rationale_tags: str
    acquirer_type: str


# ==============================================================================
# SCORING OUTPUT
# ==============================================================================

class ScoreComponents(BaseModel):
    sector: float = Field(..., ge=0.0, le=1.0)
    size: float = Field(..., ge=0.0, le=1.0)
    recency: float = Field(..., ge=0.0, le=1.0)
    close_rate: float = Field(..., ge=0.0, le=1.0)
    volume: float = Field(..., ge=0.0, le=1.0)


class ScoredAcquirer(BaseModel):
    """Output of the deterministic scoring engine."""

    acquirer_name: str
    total_score: float
    components: ScoreComponents
    conviction: Literal["High", "Medium", "Low"]
    packet: EvidencePacket


# ==============================================================================
# RATIONALE — the LLM's final output, strictly validated
# ==============================================================================

BANNED_MARKETING_PHRASES = [
    # Vague marketing adjectives — replace with numbers and transaction refs
    "world-class", "track record of", "well-positioned",
    "industry leader", "best-in-class", "proven ability",
    "deep expertise", "robust",
    # Note: "strategic acquirer" is NOT banned because it's factual IB
    # terminology (Strategic vs Financial Sponsor is a literal taxonomy field
    # in the dataset). "leading" and "premier" were also removed because they
    # commonly appear as descriptive, not promotional, in acquirer context.
]


class Conviction(BaseModel):
    level: Literal["High", "Medium", "Low"]
    # Raised from 500 -> 800 because the model legitimately wants to cite
    # all three gate outcomes + qualitative context. Forcing it below 500
    # triggered unnecessary repair loops that burned tokens + time.
    rationale: str = Field(..., min_length=20, max_length=800)


class AcquirerRationale(BaseModel):
    """The final validated rationale for a single acquirer. Returned to the UI.

    Field validators enforce the HARD RULES from the system prompt as actual
    type constraints. If the LLM violates them, Pydantic raises before the
    response ever leaves the agent."""

    acquirer_name: str
    rank: int = Field(..., ge=1)
    acquirer_overview: str = Field(..., min_length=50, max_length=800)
    strategic_fit_thesis: str = Field(..., min_length=100, max_length=1200)
    precedent_activity: str = Field(..., min_length=50)
    valuation_context: str = Field(..., min_length=50, max_length=800)
    risk_flags: str = Field(..., min_length=50)
    conviction: Conviction

    @field_validator("precedent_activity")
    @classmethod
    def must_cite_transaction_ids(cls, v: str) -> str:
        """Rule 2: Every precedent must cite at least one transaction_id."""
        if not re.search(r"MA-\d{4}-\d{4}", v):
            raise ValueError(
                "precedent_activity must cite at least one transaction_id "
                "(format: MA-YYYY-NNNN)"
            )
        return v

    @field_validator("acquirer_overview", "strategic_fit_thesis", "valuation_context")
    @classmethod
    def no_marketing_language(cls, v: str) -> str:
        """Rule 3: Block banned marketing phrases."""
        lower = v.lower()
        found = [p for p in BANNED_MARKETING_PHRASES if p in lower]
        if found:
            raise ValueError(
                f"Marketing language detected: {found}. "
                "Rewrite with numbers and transaction references."
            )
        return v

    @field_validator("valuation_context")
    @classmethod
    def must_state_multiple_and_ev_range(cls, v: str) -> str:
        """Rule 9: Valuation must cite a multiple AND derive an EV envelope."""
        has_multiple = bool(re.search(r"\d+\.?\d*x", v))
        has_ev = bool(re.search(r"\$\d+", v))
        if not (has_multiple and has_ev):
            raise ValueError(
                "valuation_context must include both a multiple (Nx) "
                "and a dollar EV figure"
            )
        return v


class RationaleSet(BaseModel):
    """The full set of 10 rationales plus run metadata."""

    target: TargetProfile
    rationales: list[AcquirerRationale]
    total_cost_usd: float
    total_duration_seconds: float
    total_input_tokens: int
    total_output_tokens: int
    cache_hits: int = 0

    @model_validator(mode="after")
    def validate_rank_contiguity(self) -> "RationaleSet":
        """Ranks must be 1..N without gaps."""
        ranks = sorted(r.rank for r in self.rationales)
        expected = list(range(1, len(ranks) + 1))
        if ranks != expected:
            raise ValueError(f"Ranks not contiguous: got {ranks}, expected {expected}")
        return self


# ==============================================================================
# TOKEN USAGE — for cost tracking that actually works (fixes the "zeros" gap)
# ==============================================================================

class TokenUsage(BaseModel):
    """Per-call token usage. Summed across calls for total cost."""

    input_tokens: int = Field(..., ge=0)
    output_tokens: int = Field(..., ge=0)
    cache_creation_tokens: int = Field(default=0, ge=0)
    cache_read_tokens: int = Field(default=0, ge=0)

    # Claude Sonnet pricing (per Anthropic public pricing page, in USD per 1M tokens).
    # Update when pricing changes — kept here explicitly so cost math is auditable.
    INPUT_PRICE_PER_MTOK: float = 3.00
    OUTPUT_PRICE_PER_MTOK: float = 15.00
    CACHE_WRITE_PRICE_PER_MTOK: float = 3.75
    CACHE_READ_PRICE_PER_MTOK: float = 0.30

    @property
    def cost_usd(self) -> float:
        """Compute actual dollar cost for this call. Never returns 0 spuriously."""
        return round(
            (self.input_tokens / 1_000_000) * self.INPUT_PRICE_PER_MTOK
            + (self.output_tokens / 1_000_000) * self.OUTPUT_PRICE_PER_MTOK
            + (self.cache_creation_tokens / 1_000_000) * self.CACHE_WRITE_PRICE_PER_MTOK
            + (self.cache_read_tokens / 1_000_000) * self.CACHE_READ_PRICE_PER_MTOK,
            6,
        )
