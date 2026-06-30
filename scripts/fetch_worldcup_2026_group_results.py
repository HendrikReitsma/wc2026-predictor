from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import html
import re
from urllib.request import Request, urlopen

import pandas as pd

from src.data.validate_data import load_team_mappings, normalize_team_name
from src.utils.paths import MANUAL_DATA_DIR


DEFAULT_SOURCE_URL = "https://www.sbnation.com/soccer/1117513/world-cup-schedule-2026-how-to-watch-every-match-scores-and-more"
OUTPUT_COLUMNS = [
    "match_id",
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
    "group",
    "source_name",
    "source_url",
]


SOURCE_ALIASES = {
    "Cape Verde": ["Cape Verde", "Cabo Verde"],
    "Czech Republic": ["Czech Republic", "Czechia"],
    "Curaçao": ["Curaçao", "Curacao"],
    "Turkey": ["Turkey", "Türkiye", "Turkiye"],
}


def _fetch_source_text(source_url: str) -> str:
    request = Request(
        source_url,
        headers={
            "User-Agent": "wc2026-predictor/0.1 results audit",
        },
    )
    with urlopen(request, timeout=30) as response:
        raw = response.read().decode("utf-8", errors="replace")
    without_scripts = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", " ", raw, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", "\n", without_scripts)
    text = html.unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def _team_aliases(team: str, mappings: dict[str, str]) -> list[str]:
    aliases = {team, *SOURCE_ALIASES.get(team, [])}
    for source, target in mappings.items():
        if target == team and source:
            aliases.add(source)
    return sorted(aliases, key=len, reverse=True)


def _score_pattern(home_alias: str, away_alias: str) -> re.Pattern[str]:
    home = re.escape(home_alias).replace(r"\ ", r"\s+")
    away = re.escape(away_alias).replace(r"\ ", r"\s+")
    return re.compile(rf"\b{home}\s+(\d+)\s*,?\s+{away}\s+(\d+)\b", re.IGNORECASE)


def _extract_score(text: str, home_team: str, away_team: str, mappings: dict[str, str]) -> tuple[int, int]:
    for home_alias in _team_aliases(home_team, mappings):
        for away_alias in _team_aliases(away_team, mappings):
            match = _score_pattern(home_alias, away_alias).search(text)
            if match:
                return int(match.group(1)), int(match.group(2))
            reverse_match = _score_pattern(away_alias, home_alias).search(text)
            if reverse_match:
                return int(reverse_match.group(2)), int(reverse_match.group(1))
    raise ValueError(f"Could not find score for {home_team} vs {away_team} in source text.")


def build_group_results(source_url: str = DEFAULT_SOURCE_URL) -> pd.DataFrame:
    text = _fetch_source_text(source_url)
    mappings = load_team_mappings(MANUAL_DATA_DIR / "team_name_mappings.csv")
    fixtures = pd.read_csv(MANUAL_DATA_DIR / "worldcup_2026_fixtures.csv")
    group_fixtures = fixtures[fixtures["stage"].astype(str).str.contains("group", case=False, na=False)].copy()
    rows: list[dict[str, object]] = []
    for _, fixture in group_fixtures.sort_values("match_id").iterrows():
        home_team = normalize_team_name(str(fixture["home_team"]), mappings)
        away_team = normalize_team_name(str(fixture["away_team"]), mappings)
        home_score, away_score = _extract_score(text, home_team, away_team, mappings)
        rows.append(
            {
                "match_id": int(fixture["match_id"]),
                "date": pd.Timestamp(fixture["match_date"]).date().isoformat(),
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "tournament": "FIFA World Cup",
                "city": fixture["city"],
                "country": fixture["country"],
                "neutral": bool(fixture["neutral"]),
                "group": fixture["group"],
                "source_name": "SB Nation",
                "source_url": source_url,
            }
        )
    results = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    if len(results) != 72 or results["match_id"].nunique() != 72:
        raise ValueError("Expected exactly 72 unique group-stage results.")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and normalize completed World Cup 2026 group-stage results.")
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument(
        "--output",
        default=str(MANUAL_DATA_DIR / "worldcup_2026_group_results.csv"),
        help="CSV path for normalized group-stage results.",
    )
    args = parser.parse_args()
    results = build_group_results(args.source_url)
    results.to_csv(args.output, index=False)
    germany_curacao = results[
        results["home_team"].eq("Germany") & results["away_team"].eq("Curaçao")
    ].iloc[0]
    print(
        f"Saved {len(results)} group-stage results to {args.output}. "
        f"Example: Germany {germany_curacao.home_score}-{germany_curacao.away_score} Curaçao."
    )


if __name__ == "__main__":
    main()
