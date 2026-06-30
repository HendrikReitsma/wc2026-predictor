from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json

import pandas as pd

from src.data.build_dataset import build_training_features
from src.features.dynamic_rating import build_rating_features
from src.models.train_goal_model import GOAL_FEATURE_GROUPS, train_goal_model
from src.models.train_outcome_model import FEATURE_GROUPS, train_outcome_model
from src.models.train_target_model import train_selected_target_model
from src.models.train_indirect_models import train_indirect_models
from src.utils.paths import MODELS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the World Cup prediction models.")
    parser.add_argument("--cutoff-date", required=True, help="Prediction cutoff date in YYYY-MM-DD format.")
    parser.add_argument("--half-life-years", type=float, default=None, help="Override time-decay half-life.")
    parser.add_argument("--minimum-year", type=int, default=None, help="Override minimum training year.")
    args = parser.parse_args()
    cutoff_date = pd.Timestamp(args.cutoff_date)
    selection_path = MODELS_DIR / "model_selection.json"
    selected_half_life = args.half_life_years
    selected_feature_columns = None
    selected_use_importance = None
    selected_goal_features = None
    selected_goal_model_type = "poisson"
    selected_minimum_year = args.minimum_year
    selected_importance_profile = None
    selected_goal_cap = None
    selected_rating_model = "standard_elo"
    selected_k_scale = 1.0
    if selected_half_life is None and selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selected_half_life = float(selection.get("training_half_life_years", selection["half_life_years"]))
        selected_feature_columns = FEATURE_GROUPS.get(str(selection.get("feature_group", "")))
        selected_use_importance = bool(selection.get("use_match_importance_weights", True))
        selected_goal_features = GOAL_FEATURE_GROUPS.get(str(selection.get("goal_feature_group", "full_poisson")))
        selected_goal_model_type = str(selection.get("goal_model_type", "poisson"))
        strategy = selection.get("training_strategy_recommendation", {})
        selected_importance_profile = str(strategy.get("importance_profile", selection.get("weighting_scheme", "aggressive")))
        selected_goal_cap = int(strategy.get("goal_cap", selection.get("goal_cap", 8)))
        selected_rating_model = str(strategy.get("rating_model", selection.get("rating_model", "standard_elo")))
        selected_k_scale = float(strategy.get("elo_k_scale", selection.get("elo_k_scale", 1.0)))
        if selected_minimum_year is None:
            selected_minimum_year = int(selection.get("minimum_training_year", 1990))
    training_features, _ = build_training_features(cutoff_date)
    training_features, rating_state = build_rating_features(
        training_features,
        model_name=selected_rating_model,
        k_scale=selected_k_scale,
    )
    (MODELS_DIR / "rating_model_state.json").write_text(json.dumps(rating_state.to_dict(), indent=2), encoding="utf-8")
    train_outcome_model(
        training_features,
        cutoff_date,
        half_life_years=selected_half_life,
        feature_columns=selected_feature_columns,
        use_match_importance=selected_use_importance,
        minimum_year=selected_minimum_year,
    )
    train_goal_model(
        training_features,
        cutoff_date,
        half_life_years=selected_half_life,
        use_match_importance=selected_use_importance,
        minimum_year=selected_minimum_year,
        feature_columns=selected_goal_features,
        model_type=selected_goal_model_type,
        goal_cap=selected_goal_cap,
        importance_profile=selected_importance_profile,
    )
    train_selected_target_model(training_features, cutoff_date)
    if (MODELS_DIR / "indirect_model_selection.json").exists():
        train_indirect_models(cutoff_date)


if __name__ == "__main__":
    main()
