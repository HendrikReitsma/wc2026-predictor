from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.train_team_strength_trend_model import TeamStrengthTrendBundle
from src.models.train_tournament_readiness_model import TournamentReadinessBundle
from src.utils.config import config_value


@dataclass(frozen=True)
class IndirectVariant:
    name: str
    trend_weight: float
    readiness_weight: float
    max_total_adjustment: float
    indirect_only: bool = False


def configured_variants() -> list[IndirectVariant]:
    return [
        IndirectVariant("baseline", 0.0, 0.0, 0.0),
        IndirectVariant("trend_small", 0.25, 0.0, 25.0),
        IndirectVariant("trend_medium", 0.50, 0.0, 50.0),
        IndirectVariant("trend_large", 1.00, 0.0, 75.0),
        IndirectVariant("readiness_small", 0.0, 0.25, 25.0),
        IndirectVariant("readiness_medium", 0.0, 0.50, 50.0),
        IndirectVariant("readiness_large", 0.0, 1.00, 75.0),
        IndirectVariant("both_small", 0.25, 0.25, 25.0),
        IndirectVariant("both_medium", 0.50, 0.50, 50.0),
        IndirectVariant("both_large", 1.00, 1.00, 75.0),
        IndirectVariant("indirect_only", 0.50, 0.50, 50.0, indirect_only=True),
    ]


def variant_by_name(name: str) -> IndirectVariant:
    variants = {variant.name: variant for variant in configured_variants()}
    if name not in variants:
        raise ValueError(f"Unknown indirect variant {name!r}.")
    return variants[name]


def build_team_adjustments(
    teams: list[str] | set[str],
    latest_snapshots: pd.DataFrame,
    trend_bundle: TeamStrengthTrendBundle,
    readiness_rows: pd.DataFrame,
    readiness_bundle: TournamentReadinessBundle,
    variant: IndirectVariant,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    teams_frame = pd.DataFrame({"team": sorted(set(teams))})
    trend = trend_bundle.predict(latest_snapshots) if not latest_snapshots.empty else pd.DataFrame(columns=["team"])
    readiness = readiness_bundle.predict(readiness_rows) if not readiness_rows.empty else pd.DataFrame(columns=["team"])
    trend_output = teams_frame.merge(trend, on="team", how="left")
    readiness_output = teams_frame.merge(readiness, on="team", how="left")
    base = teams_frame.merge(
        latest_snapshots[["team", "base_elo"]] if not latest_snapshots.empty else pd.DataFrame(columns=["team", "base_elo"]),
        on="team",
        how="left",
    )
    trend_output["base_elo"] = trend_output.get("base_elo", base["base_elo"])
    readiness_output["base_elo"] = readiness_output.get("base_elo", base["base_elo"])
    trend_output["trend_adjustment"] = (
        trend_output["expected_future_elo_delta"].fillna(0.0) * variant.trend_weight
    )
    readiness_scale = float(config_value("indirect_models", "readiness_elo_points_per_goal", default=35.0))
    readiness_output["readiness_adjustment"] = (
        readiness_output["tournament_readiness_score"].fillna(0.0)
        * readiness_scale
        * variant.readiness_weight
    )
    adjustment = base.merge(
        trend_output[["team", "trend_adjustment", "trend_data_quality_flag"]], on="team", how="left"
    ).merge(
        readiness_output[["team", "readiness_adjustment", "readiness_data_quality_flag"]], on="team", how="left"
    )
    adjustment["base_elo"] = adjustment["base_elo"].fillna(1500.0)
    adjustment[["trend_adjustment", "readiness_adjustment"]] = adjustment[
        ["trend_adjustment", "readiness_adjustment"]
    ].fillna(0.0)
    raw_total = adjustment["trend_adjustment"] + adjustment["readiness_adjustment"]
    cap = float(variant.max_total_adjustment)
    adjustment["total_indirect_adjustment"] = raw_total.clip(lower=-cap, upper=cap) if cap > 0 else 0.0
    factor = np.where(raw_total.abs().gt(1e-12), adjustment["total_indirect_adjustment"] / raw_total, 1.0)
    adjustment["trend_adjustment"] *= factor
    adjustment["readiness_adjustment"] *= factor
    adjustment["adjusted_team_strength"] = adjustment["base_elo"] + adjustment["total_indirect_adjustment"]
    adjustment["adjustment_cap_applied"] = raw_total.abs().gt(cap) if cap > 0 else False
    adjustment["data_quality_flag"] = np.where(
        adjustment["trend_data_quality_flag"].fillna("missing").eq("ok")
        & adjustment["readiness_data_quality_flag"].fillna("missing").eq("ok"),
        "ok",
        "limited_or_missing_history",
    )
    return trend_output, readiness_output, adjustment


def apply_adjustments_to_matches(frame: pd.DataFrame, adjustments: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    values = adjustments.set_index("team")["total_indirect_adjustment"].to_dict()
    home = output["home_team"].map(values).fillna(0.0).to_numpy(dtype=float)
    away = output["away_team"].map(values).fillna(0.0).to_numpy(dtype=float)
    output["home_elo_pre"] = output["home_elo_pre"].to_numpy(dtype=float) + home
    output["away_elo_pre"] = output["away_elo_pre"].to_numpy(dtype=float) + away
    output["elo_diff"] = output["home_elo_pre"] - output["away_elo_pre"]
    return output
