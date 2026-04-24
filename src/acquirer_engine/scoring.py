"""
Deterministic scoring and conviction gating.

This is the Stage 1 pipeline: no LLM involvement. Pure functions, fully unit
testable. Given a TargetProfile and a list of EvidencePackets, produces a
ranked list of ScoredAcquirers with deterministic conviction levels.

The dual enforcement of conviction rules — once here in code, once in the
system prompt — is what prevents the "all High conviction" failure mode that
plagues naive LLM implementations.
"""
from __future__ import annotations

import math
from typing import Final

from .schemas import (
    EvidencePacket,
    ScoredAcquirer,
    ScoreComponents,
    TargetProfile,
)


# ==============================================================================
# SECTOR ADJACENCY MAP
# ==============================================================================

SECTOR_ADJACENCY: Final[dict[str, dict[str, set[str]]]] = {
    "Healthcare Services": {
        "adjacent": {"Physician Groups", "Health IT", "Behavioral Health", "Dental"},
        "other": {"Revenue Cycle", "Home Health/Hospice", "Health Insurance"},
    },
    "Health IT": {
        "adjacent": {"Healthcare Services", "Revenue Cycle", "Health Insurance"},
        "other": {"Physician Groups", "Behavioral Health", "Dental"},
    },
    "Physician Groups": {
        "adjacent": {"Healthcare Services", "Behavioral Health", "Dental"},
        "other": {"Health IT", "Home Health/Hospice", "Health Insurance"},
    },
    "Behavioral Health": {
        "adjacent": {"Healthcare Services", "Physician Groups", "Home Health/Hospice"},
        "other": {"Health IT", "Dental", "Health Insurance"},
    },
    "Dental": {
        "adjacent": {"Healthcare Services", "Physician Groups"},
        "other": {"Behavioral Health", "Health IT", "Revenue Cycle"},
    },
    "Home Health/Hospice": {
        "adjacent": {"Healthcare Services", "Behavioral Health", "Physician Groups"},
        "other": {"Health IT", "Health Insurance", "Revenue Cycle"},
    },
    "Revenue Cycle": {
        "adjacent": {"Health IT", "Health Insurance"},
        "other": {"Healthcare Services", "Physician Groups", "Home Health/Hospice"},
    },
    "Health Insurance": {
        "adjacent": {"Health IT", "Revenue Cycle"},
        "other": {"Healthcare Services", "Physician Groups", "Behavioral Health"},
    },
    "Medical Devices": {
        "adjacent": {"Pharma/Biotech"},
        "other": {"Healthcare Services", "Health IT", "Revenue Cycle"},
    },
    "Pharma/Biotech": {
        "adjacent": {"Medical Devices"},
        "other": {"Healthcare Services", "Health IT", "Revenue Cycle"},
    },
}


def get_adjacency(sector: str) -> dict[str, set[str]]:
    """Return adjacency buckets for a sector. Unknown sectors fall back to empty."""
    return SECTOR_ADJACENCY.get(sector, {"adjacent": set(), "other": set()})


# ==============================================================================
# SCORING WEIGHTS
# ==============================================================================

WEIGHTS: Final[dict[str, float]] = {
    "sector": 40.0,
    "size": 25.0,
    "recency": 20.0,
    "close_rate": 10.0,
    "volume": 10.0,
}


# ==============================================================================
# COMPONENT SCORING — pure functions
# ==============================================================================

def score_sector(packet: EvidencePacket, target_sector: str) -> float:
    """Exact = 1.0, adjacent = 0.6, other healthcare = 0.3. Weighted by deal share."""
    adj = get_adjacency(target_sector)
    exact = packet.sector_distribution.get(target_sector, 0)
    adjacent = sum(packet.sector_distribution.get(s, 0) for s in adj["adjacent"])
    other = sum(packet.sector_distribution.get(s, 0) for s in adj["other"])

    if packet.total_deals == 0:
        return 0.0

    score = (exact * 1.0 + adjacent * 0.6 + other * 0.3) / packet.total_deals
    return round(min(score, 1.0), 3)


def score_size(deal_sizes_mm: list[float], target_size_mm: float) -> float:
    """Fraction of deals within 0.5x-2x of target. Decayed scoring outside."""
    if not deal_sizes_mm:
        return 0.0

    total = 0.0
    for s in deal_sizes_mm:
        ratio = s / target_size_mm
        if 0.5 <= ratio <= 2.0:
            total += 1.0
        elif 0.25 <= ratio < 0.5 or 2.0 < ratio <= 4.0:
            total += 0.4
        else:
            total += 0.1
    return round(total / len(deal_sizes_mm), 3)


def score_recency(most_recent_year: int) -> float:
    """More recent = higher score. Step function by cohort."""
    if most_recent_year >= 2022:
        return 1.0
    if most_recent_year >= 2020:
        return 0.6
    if most_recent_year >= 2018:
        return 0.3
    return 0.1


def score_close_rate(close_rate: float) -> float:
    """Bonus for >70% close rate; penalized below 50%."""
    if close_rate >= 0.7:
        return 1.0
    if close_rate >= 0.5:
        return 0.6
    return 0.3


def score_volume(total_deals: int) -> float:
    """Log-scaled so 20-deal firms don't crush 5-deal firms."""
    if total_deals < 2:
        return 0.0
    return round(min(math.log(total_deals) / math.log(15), 1.0), 3)


# ==============================================================================
# TOTAL SCORE
# ==============================================================================

def score_acquirer(
    packet: EvidencePacket,
    target: TargetProfile,
    deal_sizes_mm: list[float],
) -> tuple[float, ScoreComponents]:
    """Compute total score for one acquirer. Pure function."""
    components = ScoreComponents(
        sector=score_sector(packet, target.sector),
        size=score_size(deal_sizes_mm, target.size_mm),
        recency=score_recency(packet.most_recent_deal_year),
        close_rate=score_close_rate(packet.close_rate),
        volume=score_volume(packet.total_deals),
    )

    total = round(
        components.sector * WEIGHTS["sector"]
        + components.size * WEIGHTS["size"]
        + components.recency * WEIGHTS["recency"]
        + components.close_rate * WEIGHTS["close_rate"]
        + components.volume * WEIGHTS["volume"],
        2,
    )
    return total, components


# ==============================================================================
# CONVICTION GATING — deterministic, binding
# ==============================================================================

def compute_conviction(packet: EvidencePacket, target: TargetProfile) -> str:
    """Deterministic conviction level. CANNOT be overridden by qualitative
    judgment. This is the JS-side enforcement — the LLM gets the same rules in
    its system prompt but this function is the source of truth."""
    adj = get_adjacency(target.sector)

    exact_deals = packet.sector_distribution.get(target.sector, 0)
    adjacent_deals = sum(packet.sector_distribution.get(s, 0) for s in adj["adjacent"])
    in_band = packet.deal_size_stats.deals_in_target_band
    recent = packet.most_recent_deal_year >= 2022

    # HIGH: ALL THREE conditions
    high_sector = (exact_deals >= 2) or (adjacent_deals >= 4 and exact_deals >= 1)
    high_ok = high_sector and in_band >= 1 and recent

    # Pure adjacency cap: adjacent deals only, no exact-sector → max Medium
    pure_adjacency = adjacent_deals >= 3 and exact_deals == 0

    if high_ok and not pure_adjacency:
        return "High"
    if high_ok and pure_adjacency:
        return "Medium"

    # Count how many of the 3 core conditions are met (using relaxed sector)
    has_some_sector = (exact_deals >= 1) or (adjacent_deals >= 3)
    conditions_met = sum([has_some_sector, in_band >= 1, recent])

    if conditions_met >= 2:
        return "Medium"
    return "Low"


# ==============================================================================
# RANKING
# ==============================================================================

def rank_acquirers(
    packets: list[EvidencePacket],
    target: TargetProfile,
    deal_sizes_by_acquirer: dict[str, list[float]],
    top_n: int = 10,
) -> list[ScoredAcquirer]:
    """Score and rank all acquirers. Returns top N sorted by total_score desc."""
    scored: list[ScoredAcquirer] = []
    for p in packets:
        if p.total_deals < 2:
            continue  # Min-2-deals filter

        sizes = deal_sizes_by_acquirer.get(p.acquirer_name, [])
        total, components = score_acquirer(p, target, sizes)
        conviction = compute_conviction(p, target)

        scored.append(
            ScoredAcquirer(
                acquirer_name=p.acquirer_name,
                total_score=total,
                components=components,
                conviction=conviction,  # type: ignore[arg-type]
                packet=p,
            )
        )

    scored.sort(key=lambda x: x.total_score, reverse=True)
    return scored[:top_n]
