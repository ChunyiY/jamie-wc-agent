"""Elo rating system for international football teams."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import AppConfig, load_config
from src.team_mapping import canonical_team_name

TOURNAMENT_WEIGHTS = {
    "fifa world cup": 1.5,
    "world cup": 1.5,
    "euro": 1.35,
    "uefa nations league": 1.2,
    "copa america": 1.25,
    "afcon": 1.2,
    "friendly": 1.0,
    "qualifier": 1.15,
}


def tournament_importance(tournament: str) -> float:
    key = tournament.lower().strip()
    for name, weight in TOURNAMENT_WEIGHTS.items():
        if name in key:
            return weight
    return 1.0


def goal_difference_multiplier(goal_diff: int) -> float:
    if goal_diff <= 1:
        return 1.0
    if goal_diff == 2:
        return 1.5
    return (11 + goal_diff) / 8.0


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def actual_score(home_score: int, away_score: int, perspective: str) -> float:
    if home_score == away_score:
        return 0.5
    if perspective == "home":
        return 1.0 if home_score > away_score else 0.0
    return 1.0 if away_score > home_score else 0.0


class EloSystem:
    def __init__(self, config: AppConfig | None = None):
        self.config = config or load_config()
        self.ratings: dict[str, float] = {}

    def get_rating(self, team: str) -> float:
        team = canonical_team_name(team)
        return self.ratings.get(team, self.config.elo_initial_rating)

    def set_rating(self, team: str, rating: float) -> None:
        self.ratings[canonical_team_name(team)] = float(rating)

    def update_match(
        self,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        *,
        neutral: bool = False,
        tournament: str = "Friendly",
    ) -> tuple[float, float]:
        home_team = canonical_team_name(home_team)
        away_team = canonical_team_name(away_team)

        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)

        home_effective = home_rating + (0.0 if neutral else self.config.elo_home_advantage)
        away_effective = away_rating

        expected_home = expected_score(home_effective, away_effective)
        expected_away = 1.0 - expected_home

        actual_home = actual_score(home_score, away_score, "home")
        actual_away = actual_score(home_score, away_score, "away")

        goal_diff = abs(home_score - away_score)
        gd_mult = goal_difference_multiplier(goal_diff)
        weight = tournament_importance(tournament)
        k = self.config.elo_k_factor * gd_mult * weight

        new_home = home_rating + k * (actual_home - expected_home)
        new_away = away_rating + k * (actual_away - expected_away)

        self.ratings[home_team] = new_home
        self.ratings[away_team] = new_away
        return new_home, new_away

    def fit_history(self, results: pd.DataFrame) -> pd.DataFrame:
        history_rows: list[dict[str, float | str]] = []
        for _, row in results.sort_values("date").iterrows():
            home_before = self.get_rating(row["home_team"])
            away_before = self.get_rating(row["away_team"])
            home_after, away_after = self.update_match(
                row["home_team"],
                row["away_team"],
                int(row["home_score"]),
                int(row["away_score"]),
                neutral=bool(row.get("neutral", False)),
                tournament=str(row.get("tournament", "Friendly")),
            )
            history_rows.append(
                {
                    "date": row["date"],
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "home_elo_before": home_before,
                    "away_elo_before": away_before,
                    "home_elo_after": home_after,
                    "away_elo_after": away_after,
                }
            )
        return pd.DataFrame(history_rows)

    def to_dataframe(self) -> pd.DataFrame:
        rows = [{"team": team, "elo": rating} for team, rating in self.ratings.items()]
        frame = pd.DataFrame(rows)
        if frame.empty:
            return pd.DataFrame(columns=["team", "elo"])
        return frame.sort_values("elo", ascending=False).reset_index(drop=True)

    def save(self, path: Path) -> None:
        self.to_dataframe().to_csv(path, index=False)

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        frame = pd.read_csv(path)
        self.ratings = {
            canonical_team_name(row["team"]): float(row["elo"])
            for _, row in frame.iterrows()
        }
