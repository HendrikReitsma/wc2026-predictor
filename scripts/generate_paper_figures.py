from __future__ import annotations

import _bootstrap  # noqa: F401

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.models.predict_match import predict_match
from src.models.weighting import match_importance_weights
from src.utils.paths import PREDICTIONS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR


CUTOFF_DATE = pd.Timestamp("2026-06-10")
FIGURES_DIR = REPORTS_DIR / "figures"


def _save(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close()


def _training_frame() -> pd.DataFrame:
    frame = pd.read_csv(PROCESSED_DATA_DIR / "match_features.csv", low_memory=False)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame[(frame["date"].dt.year >= 1990) & (frame["date"] <= CUTOFF_DATE)].copy()


def create_feature_distributions(frame: pd.DataFrame) -> None:
    plot_frame = frame.assign(
        total_goals=frame["home_score"] + frame["away_score"],
        rolling_goals_for_5=pd.concat(
            [frame["rolling_goals_for_home_5"], frame["rolling_goals_for_away_5"]],
            ignore_index=True,
        ),
        attack_rating=pd.concat(
            [frame["home_attack_rating_pre"], frame["away_attack_rating_pre"]],
            ignore_index=True,
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    sns.histplot(frame["elo_diff"].clip(-600, 600), bins=50, ax=axes[0, 0], color="#35618c")
    axes[0, 0].set_title("Pre-match Elo difference")
    axes[0, 0].set_xlabel("Home Elo minus away Elo (clipped for display)")

    sns.histplot(plot_frame["rolling_goals_for_5"].clip(0, 6), bins=30, ax=axes[0, 1], color="#438c6e")
    axes[0, 1].set_title("Five-match rolling goals scored")
    axes[0, 1].set_xlabel("Goals per match (both home and away team features)")

    sns.histplot(plot_frame["attack_rating"].clip(-2, 2.5), bins=40, ax=axes[1, 0], color="#9a6a3a")
    axes[1, 0].set_title("Smoothed attack residual rating")
    axes[1, 0].set_xlabel("Goals above Elo-derived expectation (clipped for display)")

    sns.histplot(plot_frame["total_goals"].clip(0, 10), discrete=True, ax=axes[1, 1], color="#8b4f78")
    axes[1, 1].set_title("Observed total goals")
    axes[1, 1].set_xlabel("Total match goals (10 includes higher totals)")
    fig.suptitle("Selected Training-Data Distributions, 1990 to 10 June 2026", fontsize=15)
    _save("paper_feature_distributions.png")


def create_feature_correlations(frame: pd.DataFrame) -> None:
    columns = {
        "elo_diff": "Elo difference",
        "rolling_goals_for_home_5": "Home GF, last 5",
        "rolling_goals_against_home_5": "Home GA, last 5",
        "rolling_goals_for_away_5": "Away GF, last 5",
        "rolling_goals_against_away_5": "Away GA, last 5",
        "home_attack_rating_pre": "Home attack rating",
        "home_defence_rating_pre": "Home defence rating",
        "away_attack_rating_pre": "Away attack rating",
        "away_defence_rating_pre": "Away defence rating",
    }
    correlation = frame[list(columns)].rename(columns=columns).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.4,
        cbar_kws={"label": "Pearson correlation"},
    )
    plt.title("Correlation Among Selected Continuous Model Features")
    _save("paper_feature_correlation.png")


def create_importance_weights(frame: pd.DataFrame) -> None:
    weights = match_importance_weights(frame, profile="aggressive")
    labels = np.select(
        [
            frame["is_friendly"].astype(bool),
            frame["is_world_cup"].astype(bool),
            frame["is_continental_competition"].astype(bool),
            frame["tournament"].astype(str).str.contains("qualif|wcq", case=False, regex=True, na=False),
        ],
        ["Friendly", "World Cup labeled", "Continental", "Qualifier"],
        default="Other competitive",
    )
    summary = (
        pd.DataFrame({"match_type": labels, "weight": weights})
        .groupby("match_type", as_index=False)
        .agg(matches=("weight", "size"), mean_weight=("weight", "mean"))
        .sort_values("mean_weight")
    )
    plt.figure(figsize=(9, 5))
    bars = plt.barh(summary["match_type"], summary["mean_weight"], color="#35618c")
    for bar, matches in zip(bars, summary["matches"]):
        plt.text(bar.get_width() + 0.04, bar.get_y() + bar.get_height() / 2, f"n={matches:,}", va="center")
    plt.xlabel("Mean pre-normalization sample weight")
    plt.title("Selected Aggressive Importance Weights and Available Match Counts")
    plt.xlim(0, max(summary["mean_weight"]) + 0.9)
    _save("paper_match_importance_weights.png")


def create_expected_vs_modal_goals() -> None:
    matches = pd.read_csv(PREDICTIONS_DIR / "match_predictions_2026.csv")
    score_parts = matches["most_likely_score"].str.split("-", expand=True).astype(int)
    matches["modal_total_goals"] = score_parts[0] + score_parts[1]
    matches["expected_total_goals"] = matches["expected_goals_home"] + matches["expected_goals_away"]
    plt.figure(figsize=(8, 6))
    sns.stripplot(
        data=matches,
        x="modal_total_goals",
        y="expected_total_goals",
        jitter=0.14,
        alpha=0.72,
        color="#35618c",
    )
    max_value = float(max(matches["expected_total_goals"].max(), matches["modal_total_goals"].max()) + 0.25)
    plt.plot([0, max_value], [0, max_value], linestyle="--", color="grey", label="Expected total = modal total")
    spain = matches[(matches["home_team"] == "Spain") & (matches["away_team"] == "Cape Verde")].iloc[0]
    plt.scatter(
        spain["modal_total_goals"],
        spain["expected_total_goals"],
        color="#b33b32",
        edgecolor="black",
        s=90,
        zorder=3,
        label="Spain vs Cape Verde",
    )
    plt.xlabel("Total goals in most likely exact score")
    plt.ylabel("Expected total goals")
    plt.title("Expected Goals Commonly Exceed the Modal Exact-Score Total")
    plt.legend()
    _save("paper_expected_vs_modal_goals.png")


def create_spain_cape_verde_surface() -> None:
    prediction = predict_match(
        home_team="Spain",
        away_team="Cape Verde",
        match_date="2026-06-15",
        neutral=True,
        venue_country="United States",
        tournament="World Cup",
        stage="Group Stage",
    )
    matrix = np.asarray(prediction["_scoreline_matrix_values"], dtype=float)
    display = matrix[:7, :7]
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        display,
        annot=True,
        fmt=".1%",
        cmap="Blues",
        cbar_kws={"label": "Scoreline probability"},
    )
    plt.xlabel("Cape Verde goals")
    plt.ylabel("Spain goals")
    plt.title(
        "Spain vs Cape Verde Scoreline Distribution\n"
        f"E[goals] = {prediction['expected_goals_home']:.2f}-{prediction['expected_goals_away']:.2f}; "
        f"mode = {prediction['most_likely_score']}"
    )
    _save("paper_spain_cape_verde_score_matrix.png")


def write_figure_metadata(frame: pd.DataFrame) -> None:
    metadata = {
        "cutoff_date": CUTOFF_DATE.date().isoformat(),
        "training_rows": int(len(frame)),
        "figures": [
            "paper_feature_distributions.png",
            "paper_feature_correlation.png",
            "paper_match_importance_weights.png",
            "paper_expected_vs_modal_goals.png",
            "paper_spain_cape_verde_score_matrix.png",
        ],
    }
    (FIGURES_DIR / "paper_figure_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    frame = _training_frame()
    create_feature_distributions(frame)
    create_feature_correlations(frame)
    create_importance_weights(frame)
    create_expected_vs_modal_goals()
    create_spain_cape_verde_surface()
    write_figure_metadata(frame)


if __name__ == "__main__":
    main()
