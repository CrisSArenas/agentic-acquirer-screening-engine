"""
CSV loading and evidence packet assembly.

Loads the 500-row M&A transaction dataset, groups by acquirer, and produces
structured EvidencePacket objects consumed by the scoring and tool layers.

Separating this from scoring keeps each module independently testable.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from .schemas import (
    ClosedDealMultiples,
    DealSizeStats,
    EvidencePacket,
    RelevantTransaction,
    TargetProfile,
)
from .scoring import get_adjacency

REQUIRED_COLUMNS = [
    "transaction_id", "target_company", "acquirer", "sector", "sub_sector",
    "deal_year", "deal_type", "geography", "deal_size_mm", "ev_ebitda_multiple",
    "ev_revenue_multiple", "ebitda_margin_pct", "outcome",
    "strategic_rationale_tags", "acquirer_type",
]


def load_csv(path: Path | str) -> pd.DataFrame:
    """Load and validate the M&A CSV. Raises on missing columns."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    return df


def build_evidence_packets(
    df: pd.DataFrame,
    target: TargetProfile,
    min_deals: int = 2,
) -> list[EvidencePacket]:
    """Group by acquirer; produce one EvidencePacket per qualifying acquirer."""
    size_lo, size_hi = target.size_lo_mm, target.size_hi_mm
    target_band = f"${size_lo:.0f}M-${size_hi:.0f}M"

    packets: list[EvidencePacket] = []

    for acquirer_name, group in df.groupby("acquirer"):
        if len(group) < min_deals:
            continue

        # Sector distributions
        sector_dist = group["sector"].value_counts().to_dict()
        sub_sector_dist = group["sub_sector"].value_counts().to_dict()

        # Sizes
        sizes = group["deal_size_mm"].dropna().tolist()
        in_band = sum(1 for s in sizes if size_lo <= s <= size_hi)

        # Closed-deal multiples
        closed = group[group["outcome"] == "Closed"]
        ev_ebitda = closed["ev_ebitda_multiple"].dropna()
        ev_rev = closed["ev_revenue_multiple"].dropna()

        # Deal type + geography mix
        deal_type_mix = group["deal_type"].value_counts().to_dict()
        geo_mix = group["geography"].value_counts().to_dict()

        # Top strategic rationale tags
        all_tags: list[str] = []
        for tag_str in group["strategic_rationale_tags"].dropna():
            all_tags.extend(t.strip() for t in str(tag_str).split("|") if t.strip())
        top_tags = Counter(all_tags).most_common(5)

        # Acquirer type (mode)
        acq_type = group["acquirer_type"].mode().iloc[0]

        close_rate = len(closed) / len(group)

        packets.append(
            EvidencePacket(
                acquirer_name=str(acquirer_name),
                acquirer_type=acq_type,
                total_deals=len(group),
                sector_distribution={str(k): int(v) for k, v in sector_dist.items()},
                sub_sector_distribution={str(k): int(v) for k, v in sub_sector_dist.items()},
                deal_size_stats=DealSizeStats(
                    min_mm=float(min(sizes)) if sizes else 0.0,
                    median_mm=float(pd.Series(sizes).median()) if sizes else 0.0,
                    max_mm=float(max(sizes)) if sizes else 0.0,
                    deals_in_target_band=in_band,
                    target_band=target_band,
                ),
                closed_deal_multiples=ClosedDealMultiples(
                    median_ev_ebitda=float(ev_ebitda.median()) if len(ev_ebitda) else None,
                    median_ev_revenue=float(ev_rev.median()) if len(ev_rev) else None,
                    num_closed_deals=len(closed),
                ),
                deal_type_mix={str(k): int(v) for k, v in deal_type_mix.items()},
                geography_mix={str(k): int(v) for k, v in geo_mix.items()},
                top_strategic_rationale_tags=[
                    {"tag": tag, "count": count} for tag, count in top_tags
                ],
                most_recent_deal_year=int(group["deal_year"].max()),
                close_rate=round(close_rate, 3),
            )
        )

    return packets


def get_deal_sizes_by_acquirer(df: pd.DataFrame) -> dict[str, list[float]]:
    """Return {acquirer_name: [deal_sizes_mm]} for the scoring function."""
    return {
        str(name): group["deal_size_mm"].dropna().tolist()
        for name, group in df.groupby("acquirer")
    }


def select_relevant_transactions(
    df: pd.DataFrame,
    acquirer_name: str,
    target: TargetProfile,
    n: int = 3,
) -> list[RelevantTransaction]:
    """Pick top N most relevant deals for an acquirer given the target profile.

    Scoring: sector match (exact=3, adjacent=1.5, other=0.5) + size proximity
    (in-band=2, 0.25x-4x=1, else 0). Highest scoring go to the prompt."""
    adj = get_adjacency(target.sector)
    group = df[df["acquirer"] == acquirer_name].copy()

    if group.empty:
        return []

    def relevance_score(row: pd.Series) -> float:
        score = 0.0
        if row["sector"] == target.sector:
            score += 3.0
        elif row["sector"] in adj["adjacent"]:
            score += 1.5
        elif row["sector"] in adj["other"]:
            score += 0.5

        size = row.get("deal_size_mm")
        if pd.notna(size):
            ratio = size / target.size_mm
            if 0.5 <= ratio <= 2.0:
                score += 2.0
            elif 0.25 <= ratio <= 4.0:
                score += 1.0
        return score

    group["_rel_score"] = group.apply(relevance_score, axis=1)
    group = group.sort_values("_rel_score", ascending=False).head(n)

    results: list[RelevantTransaction] = []
    for _, row in group.iterrows():
        results.append(
            RelevantTransaction(
                transaction_id=str(row["transaction_id"]),
                target_company=str(row["target_company"]),
                sector=str(row["sector"]),
                sub_sector=str(row["sub_sector"]),
                deal_year=int(row["deal_year"]),
                deal_type=str(row["deal_type"]),
                geography=str(row["geography"]),
                deal_size_mm=float(row["deal_size_mm"]) if pd.notna(row["deal_size_mm"]) else None,
                ev_ebitda_multiple=float(row["ev_ebitda_multiple"]) if pd.notna(row["ev_ebitda_multiple"]) else None,
                ev_revenue_multiple=float(row["ev_revenue_multiple"]) if pd.notna(row["ev_revenue_multiple"]) else None,
                ebitda_margin_pct=float(row["ebitda_margin_pct"]) if pd.notna(row["ebitda_margin_pct"]) else None,
                outcome=str(row["outcome"]),
                strategic_rationale_tags=str(row["strategic_rationale_tags"]),
                acquirer_type=str(row["acquirer_type"]),
            )
        )
    return results
