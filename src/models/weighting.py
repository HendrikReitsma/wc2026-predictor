from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.config import config_value


def time_decay_weights(
    dates: pd.Series,
    reference_date: str | pd.Timestamp,
    half_life_years: float,
) -> np.ndarray:
    if half_life_years <= 0:
        raise ValueError("time-decay half-life must be positive.")
    parsed_dates = pd.to_datetime(dates)
    age_years = np.maximum((pd.Timestamp(reference_date) - parsed_dates).dt.days.to_numpy() / 365.25, 0.0)
    return np.power(0.5, age_years / float(half_life_years))


def match_importance_weights(frame: pd.DataFrame, profile: str | None = None) -> np.ndarray:
    profiles = config_value("modeling", "match_weight_profiles", default={}) or {}
    configured = (
        profiles.get(profile, {})
        if profile and profile not in {"configured", "none"}
        else config_value("modeling", "match_type_weights", default={}) or {}
    )
    default = float(configured.get("default", 0.9))
    weights = np.full(len(frame), default, dtype=float)
    weights = np.where(frame["is_friendly"].fillna(0).astype(bool), float(configured.get("friendly", 0.5)), weights)
    minor_mask = frame["tournament"].astype(str).str.contains("nations league", case=False, na=False)
    weights = np.where(minor_mask, float(configured.get("minor_competitive", 1.2)), weights)
    qualifier_mask = (
        frame["tournament"].astype(str).str.contains("qualif|wcq", case=False, regex=True, na=False)
        & ~frame["is_world_cup"].fillna(0).astype(bool)
    )
    weights = np.where(qualifier_mask, float(configured.get("qualifier", 1.0)), weights)
    stage = frame["stage"].astype(str).str.lower() if "stage" in frame.columns else pd.Series("", index=frame.index)
    knockout = stage.str.contains("round|quarter|semi|final|knockout", regex=True, na=False)
    continental = frame["is_continental_competition"].fillna(0).astype(bool)
    world_cup = frame["is_world_cup"].fillna(0).astype(bool)
    weights = np.where(continental, float(configured.get("continental", 1.25)), weights)
    weights = np.where(continental & ~knockout, float(configured.get("continental_group", configured.get("continental", 1.25))), weights)
    weights = np.where(continental & knockout, float(configured.get("continental_knockout", configured.get("continental", 1.25))), weights)
    weights = np.where(world_cup, float(configured.get("world_cup", 1.5)), weights)
    weights = np.where(world_cup & ~knockout, float(configured.get("world_cup_group", configured.get("world_cup", 1.5))), weights)
    weights = np.where(world_cup & knockout, float(configured.get("world_cup_knockout", configured.get("world_cup", 1.5))), weights)
    return weights


def combined_sample_weights(
    frame: pd.DataFrame,
    reference_date: str | pd.Timestamp,
    half_life_years: float | None = None,
    use_match_importance: bool | None = None,
    importance_profile: str | None = None,
) -> np.ndarray:
    half_life = float(
        half_life_years
        if half_life_years is not None
        else config_value("modeling", "time_decay_half_life_years", default=8)
    )
    use_importance = bool(
        use_match_importance
        if use_match_importance is not None
        else config_value("modeling", "use_match_importance_weights", default=True)
    )
    weights = time_decay_weights(frame["date"], reference_date, half_life)
    if use_importance and importance_profile != "none":
        weights = weights * match_importance_weights(frame, importance_profile)
    return weights / max(float(np.mean(weights)), 1e-12)
