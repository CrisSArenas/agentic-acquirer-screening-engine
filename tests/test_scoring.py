"""
Scoring unit tests.

The scoring layer is composed of pure functions with no side effects, so it's
testable in isolation with plain fixtures — no API key, no mocks, no network.
This file covers: individual component scores, the composite weighted score,
conviction gating rules, and end-to-end ranking.
"""
from __future__ import annotations

import pytest

from acquirer_engine.scoring import (
    compute_conviction,
    get_adjacency,
    rank_acquirers,
    score_recency,
    score_sector,
    score_size,
    score_volume,
)
from acquirer_engine.schemas import TargetProfile


# ==============================================================================
# ADJACENCY MAP
# ==============================================================================

def test_healthcare_services_adjacency_includes_physician_groups() -> None:
    adj = get_adjacency("Healthcare Services")
    assert "Physician Groups" in adj["adjacent"]
    assert "Health IT" in adj["adjacent"]


def test_unknown_sector_returns_empty_adjacency() -> None:
    adj = get_adjacency("NonExistentSector")
    assert adj == {"adjacent": set(), "other": set()}


# ==============================================================================
# SECTOR SCORING
# ==============================================================================

def test_exact_sector_match_scores_one(healthcare_services_packet) -> None:
    score = score_sector(healthcare_services_packet, "Healthcare Services")
    assert score == 1.0


def test_pure_adjacency_scores_point_six(pure_adjacency_packet) -> None:
    # 3 Dental deals for a Healthcare Services target = 3 * 0.6 / 3 = 0.6
    score = score_sector(pure_adjacency_packet, "Healthcare Services")
    assert score == pytest.approx(0.6, abs=0.01)


# ==============================================================================
# SIZE SCORING
# ==============================================================================

def test_size_in_band_scores_one() -> None:
    # All 3 deals within 0.5x-2.0x of $200M → 1.0
    assert score_size([150.0, 200.0, 300.0], target_size_mm=200.0) == 1.0


def test_size_outside_band_scores_penalty() -> None:
    # All 3 deals at 10x target → 0.1
    assert score_size([2000.0, 2100.0, 2200.0], target_size_mm=200.0) == 0.1


def test_size_empty_returns_zero() -> None:
    assert score_size([], target_size_mm=200.0) == 0.0


# ==============================================================================
# RECENCY SCORING
# ==============================================================================

@pytest.mark.parametrize("year,expected", [
    (2024, 1.0),
    (2022, 1.0),
    (2021, 0.6),
    (2020, 0.6),
    (2019, 0.3),
    (2015, 0.1),
])
def test_recency_step_function(year: int, expected: float) -> None:
    assert score_recency(year) == expected


# ==============================================================================
# VOLUME SCORING
# ==============================================================================

def test_volume_below_min_is_zero() -> None:
    assert score_volume(1) == 0.0


def test_volume_is_log_scaled() -> None:
    # log(15)/log(15) = 1.0 cap
    assert score_volume(15) == 1.0
    # Monotonic increase
    assert score_volume(5) < score_volume(10) < score_volume(15)


# ==============================================================================
# CONVICTION GATING — the critical path
# ==============================================================================

def test_high_conviction_when_all_gates_pass(
    healthcare_services_packet, target_profile,
) -> None:
    assert compute_conviction(healthcare_services_packet, target_profile) == "High"


def test_pure_adjacency_capped_at_medium(
    pure_adjacency_packet, target_profile,
) -> None:
    """3 Dental deals + all in band + recent — should be Medium, not High."""
    assert compute_conviction(pure_adjacency_packet, target_profile) == "Medium"


def test_stale_acquirer_misses_recency_gate(healthcare_services_packet, target_profile) -> None:
    """Fail condition c (recency) → only 2 of 3 conditions → Medium."""
    healthcare_services_packet.most_recent_deal_year = 2019
    assert compute_conviction(healthcare_services_packet, target_profile) == "Medium"


def test_out_of_band_and_stale_is_low(healthcare_services_packet, target_profile) -> None:
    """Fail conditions b AND c → Low."""
    healthcare_services_packet.deal_size_stats.deals_in_target_band = 0
    healthcare_services_packet.most_recent_deal_year = 2017
    assert compute_conviction(healthcare_services_packet, target_profile) == "Low"


# ==============================================================================
# END-TO-END RANKING
# ==============================================================================

def test_rank_acquirers_sorts_by_score(
    healthcare_services_packet, pure_adjacency_packet, target_profile,
) -> None:
    packets = [pure_adjacency_packet, healthcare_services_packet]
    deal_sizes = {
        "Fictional Health Corp": [120.0, 200.0, 300.0],
        "Dental Only Corp": [150.0, 200.0, 250.0],
    }
    top = rank_acquirers(packets, target_profile, deal_sizes, top_n=10)

    # HC Services packet should rank above Dental-only
    assert top[0].acquirer_name == "Fictional Health Corp"
    assert top[0].total_score > top[1].total_score


def test_rank_filters_min_2_deals(target_profile, healthcare_services_packet) -> None:
    healthcare_services_packet.total_deals = 1
    top = rank_acquirers([healthcare_services_packet], target_profile, {}, top_n=10)
    assert top == []
