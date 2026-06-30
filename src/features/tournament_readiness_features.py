from __future__ import annotations

import pandas as pd

from src.features.team_strength_features import TREND_FEATURE_COLUMNS, latest_team_snapshots


READINESS_FEATURE_COLUMNS = [
    "base_elo",
    "pre_tournament_elo_rank",
    "elo_change_last_5_matches",
    "elo_change_last_10_matches",
    "elo_change_last_12_months",
    "recent_points_above_expectation_10",
    "recent_goal_difference_above_expectation_10",
    "recent_opponent_elo_10",
    "matches_last_12_months",
    "qualification_points_per_match",
    "qualification_goal_difference_per_match",
    "qualification_points_above_expectation_per_match",
    "qualification_opponent_average_elo",
    "host_flag",
]


def _qualification_summary(snapshots: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    qualifying = snapshots[
        (snapshots["date"] < start)
        & (snapshots["date"] >= start - pd.DateOffset(years=4))
        & snapshots["tournament"].astype(str).str.contains("qualif|wcq", case=False, regex=True, na=False)
    ]
    if qualifying.empty:
        return pd.DataFrame(columns=["team"])
    return (
        qualifying.groupby("team", as_index=False)
        .agg(
            qualification_matches=("date", "size"),
            qualification_points_per_match=("actual_points", "mean"),
            qualification_goal_difference_per_match=("goal_difference", "mean"),
            qualification_points_above_expectation_per_match=("points_above_expectation", "mean"),
            qualification_opponent_average_elo=("opponent_elo", "mean"),
        )
    )


def readiness_snapshot_for_teams(
    snapshots: pd.DataFrame,
    teams: list[str] | set[str],
    tournament_start: str | pd.Timestamp,
    tournament_year: int,
    host_teams: set[str] | None = None,
) -> pd.DataFrame:
    start = pd.Timestamp(tournament_start)
    latest = latest_team_snapshots(snapshots, teams, start)
    if latest.empty:
        return latest
    latest = latest.copy()
    latest["tournament_year"] = int(tournament_year)
    latest["tournament_start"] = start
    latest["pre_tournament_elo_rank"] = latest["base_elo"].rank(method="min", ascending=False)
    latest["host_flag"] = latest["team"].isin(host_teams or set()).astype(int)
    qualification = _qualification_summary(snapshots, start)
    latest = latest.merge(qualification, on="team", how="left")
    if "qualification_matches" not in latest:
        latest["qualification_matches"] = 0.0
    for column in READINESS_FEATURE_COLUMNS:
        if column not in latest:
            latest[column] = 0.0
    return latest


def build_historical_readiness_dataset(
    match_features: pd.DataFrame,
    snapshots: pd.DataFrame,
) -> pd.DataFrame:
    frame = match_features.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    fifa_world_cup = frame["tournament"].astype(str).str.strip().eq("FIFA World Cup")
    rows: list[pd.DataFrame] = []
    for year, tournament in frame[fifa_world_cup].groupby(frame.loc[fifa_world_cup, "date"].dt.year):
        start = tournament["date"].min()
        teams = set(tournament["home_team"]).union(tournament["away_team"])
        host_teams = set()
        for side in ["home", "away"]:
            host_teams.update(
                tournament.loc[
                    tournament.get(f"playing_in_home_country_{side}", pd.Series(0, index=tournament.index)).eq(1),
                    f"{side}_team",
                ]
            )
        pre = readiness_snapshot_for_teams(snapshots, teams, start, int(year), host_teams)
        actual = snapshots[
            snapshots["tournament"].astype(str).str.strip().eq("FIFA World Cup")
            & snapshots["date"].dt.year.eq(int(year))
            & snapshots["team"].isin(teams)
        ]
        targets = (
            actual.groupby("team", as_index=False)
            .agg(
                tournament_matches=("date", "size"),
                tournament_performance_above_elo_expectation=("goal_difference_above_expectation", "mean"),
                tournament_goal_difference_above_expectation=("goal_difference_above_expectation", "sum"),
                tournament_points_above_expectation=("points_above_expectation", "sum"),
                tournament_points=("actual_points", "sum"),
                tournament_goal_difference=("goal_difference", "sum"),
                readiness_target_end_date=("date", "max"),
            )
        )
        rows.append(pre.merge(targets, on="team", how="inner"))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
