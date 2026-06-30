from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import pandas as pd

from src.data.build_dataset import build_training_features, load_fixtures_frame
from src.data.validate_data import load_team_mappings
from src.models.predict_match import predict_match
from src.simulation.simulate_tournament import simulate_tournament
from src.evaluation.reporting import create_prediction_summary
from src.utils.paths import MANUAL_DATA_DIR, MODELS_DIR, PREDICTIONS_DIR
from src.utils.config import config_value


def _predict_fixtures(cutoff_date: pd.Timestamp) -> pd.DataFrame:
    fixture_path = MANUAL_DATA_DIR / "worldcup_2026_fixtures.csv"
    if not fixture_path.exists():
        raise FileNotFoundError(
            f"Missing fixture file at {fixture_path}. Provide the World Cup 2026 manual fixture file first."
        )
    mappings = load_team_mappings(MANUAL_DATA_DIR / "team_name_mappings.csv")
    fixtures = load_fixtures_frame(mappings)
    predictions = []
    for _, fixture in fixtures.iterrows():
        if not str(fixture["home_team"]).strip() or not str(fixture["away_team"]).strip():
            continue
        result = dict(
            predict_match(
                home_team=str(fixture["home_team"]),
                away_team=str(fixture["away_team"]),
                match_date=pd.Timestamp(fixture["match_date"]),
                neutral=bool(fixture["neutral"]),
                venue_country=str(fixture["country"]) if pd.notna(fixture["country"]) else None,
                tournament="World Cup",
                stage=str(fixture["stage"]),
            )
        )
        result.update(
            {
                "match_id": fixture["match_id"],
                "stage": fixture["stage"],
                "group": fixture["group"],
                "match_date": fixture["match_date"],
            }
        )
        predictions.append(result)
    predictions_frame = pd.DataFrame(predictions)
    output_columns = [
        "match_id",
        "stage",
        "group",
        "match_date",
        "home_team",
        "away_team",
        "expected_goals_home",
        "expected_goals_away",
        "baseline_expected_goals_home",
        "indirect_expected_goals_home",
        "baseline_expected_goals_away",
        "indirect_expected_goals_away",
        "p_home_win",
        "p_draw",
        "p_away_win",
        "baseline_p_home_win",
        "indirect_p_home_win",
        "baseline_p_draw",
        "indirect_p_draw",
        "baseline_p_away_win",
        "indirect_p_away_win",
        "most_likely_score",
        "baseline_most_likely_score",
        "indirect_most_likely_score",
        "most_likely_score_probability",
        "top_5_scorelines",
        "p_over_2_5_goals",
        "p_under_2_5_goals",
        "p_both_teams_score",
        "p_clean_sheet_home",
        "p_clean_sheet_away",
        "model_used",
        "indirect_adjustment_used",
        "indirect_comparison_variant",
        "trend_adjustment_home",
        "trend_adjustment_away",
        "readiness_adjustment_home",
        "readiness_adjustment_away",
        "feature_set",
        "calibration_method",
        "draw_correction",
        "model_version",
        "prediction_cutoff_date",
    ]
    predictions_frame = predictions_frame.reindex(columns=output_columns)
    predictions_path = PREDICTIONS_DIR / "match_predictions_2026.csv"
    predictions_frame.to_csv(predictions_path, index=False)
    comparison = predictions_frame[
        [
            "match_id", "home_team", "away_team",
            "baseline_expected_goals_home", "indirect_expected_goals_home",
            "baseline_expected_goals_away", "indirect_expected_goals_away",
            "baseline_p_home_win", "indirect_p_home_win",
            "baseline_p_draw", "indirect_p_draw",
            "baseline_p_away_win", "indirect_p_away_win",
            "baseline_most_likely_score", "indirect_most_likely_score",
        ]
    ].copy()
    comparison["delta_p_home_win"] = comparison["indirect_p_home_win"] - comparison["baseline_p_home_win"]
    comparison["delta_p_away_win"] = comparison["indirect_p_away_win"] - comparison["baseline_p_away_win"]
    comparison.to_csv(PREDICTIONS_DIR / "team_only_vs_indirect_match_comparison.csv", index=False)
    return predictions_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict the full World Cup 2026 tournament.")
    parser.add_argument("--cutoff-date", required=True, help="Prediction cutoff date in YYYY-MM-DD format.")
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=int(config_value("simulation", "default_n_simulations", default=10000)),
        help="Monte Carlo simulation count.",
    )
    args = parser.parse_args()
    cutoff_date = pd.Timestamp(args.cutoff_date)
    max_simulations = int(config_value("simulation", "max_n_simulations", default=10000))
    if args.n_simulations > max_simulations:
        raise ValueError(f"n-simulations cannot exceed configured maximum of {max_simulations}.")
    required_models = [
        MODELS_DIR / "goal_model_home.joblib",
        MODELS_DIR / "goal_model_away.joblib",
        MODELS_DIR / "outcome_model_best.joblib",
    ]
    missing_models = [str(path) for path in required_models if not path.exists()]
    if missing_models:
        raise FileNotFoundError(
            "Trained models are missing. Run scripts/train_models.py first. Missing: " + ", ".join(missing_models)
        )
    build_training_features(cutoff_date)
    _predict_fixtures(cutoff_date)
    simulate_tournament(n_simulations=args.n_simulations, seed=42)
    create_prediction_summary(str(cutoff_date.date()), n_simulations=args.n_simulations)


if __name__ == "__main__":
    main()
