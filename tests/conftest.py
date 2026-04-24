"""Pytest fixtures shared across tests."""
from __future__ import annotations

import sys
from pathlib import Path

# Make `src/` importable during tests
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import pytest

from acquirer_engine.schemas import (
    ClosedDealMultiples,
    DealSizeStats,
    EvidencePacket,
    TargetProfile,
)


@pytest.fixture
def target_profile() -> TargetProfile:
    return TargetProfile(
        sector="Healthcare Services",
        size_mm=200.0,
        geography="Regional",
    )


@pytest.fixture
def sample_csv_path() -> Path:
    path = ROOT / "data" / "ma_transactions_500.csv"
    if not path.exists():
        pytest.skip("Sample CSV not present")
    return path


@pytest.fixture
def sample_df(sample_csv_path) -> pd.DataFrame:
    return pd.read_csv(sample_csv_path)


@pytest.fixture
def healthcare_services_packet() -> EvidencePacket:
    """An acquirer that should score HIGH: 3 HC Services deals, 2 in band, recent."""
    return EvidencePacket(
        acquirer_name="Fictional Health Corp",
        acquirer_type="Strategic",
        total_deals=3,
        sector_distribution={"Healthcare Services": 3},
        sub_sector_distribution={"Healthcare Services": 3},
        deal_size_stats=DealSizeStats(
            min_mm=120.0, median_mm=200.0, max_mm=300.0,
            deals_in_target_band=2, target_band="$100M-$400M",
        ),
        closed_deal_multiples=ClosedDealMultiples(
            median_ev_ebitda=13.5, median_ev_revenue=2.1, num_closed_deals=3,
        ),
        deal_type_mix={"Strategic Acquisition": 3},
        geography_mix={"Midwest": 2, "Northeast": 1},
        top_strategic_rationale_tags=[{"tag": "Geographic Expansion", "count": 3}],
        most_recent_deal_year=2023,
        close_rate=1.0,
    )


@pytest.fixture
def pure_adjacency_packet() -> EvidencePacket:
    """Dental-only acquirer targeting Healthcare Services → should cap at Medium."""
    return EvidencePacket(
        acquirer_name="Dental Only Corp",
        acquirer_type="Strategic",
        total_deals=3,
        sector_distribution={"Dental": 3},
        sub_sector_distribution={"Dental": 3},
        deal_size_stats=DealSizeStats(
            min_mm=150.0, median_mm=200.0, max_mm=250.0,
            deals_in_target_band=3, target_band="$100M-$400M",
        ),
        closed_deal_multiples=ClosedDealMultiples(
            median_ev_ebitda=13.0, median_ev_revenue=2.0, num_closed_deals=3,
        ),
        deal_type_mix={"Bolt-on Acquisition": 3},
        geography_mix={"National": 3},
        top_strategic_rationale_tags=[{"tag": "Platform Build", "count": 3}],
        most_recent_deal_year=2023,
        close_rate=1.0,
    )
