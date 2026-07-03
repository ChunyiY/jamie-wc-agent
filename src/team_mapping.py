"""National team name normalization and aliases for market parsing."""

from __future__ import annotations

import re
from functools import lru_cache

from src.utils import normalize_team_name

TEAM_ALIASES: dict[str, str] = {
    "usa": "United States",
    "us": "United States",
    "united states": "United States",
    "u.s.": "United States",
    "u.s.a.": "United States",
    "u.s.a.": "United States",
    "england": "England",
    "uk": "England",
    "great britain": "England",
    "south korea": "South Korea",
    "korea republic": "South Korea",
    "korea rep": "South Korea",
    "republic of korea": "South Korea",
    "north korea": "North Korea",
    "dpr korea": "North Korea",
    "korea dpr": "North Korea",
    "ivory coast": "Ivory Coast",
    "cote d'ivoire": "Ivory Coast",
    "côte d'ivoire": "Ivory Coast",
    "czech republic": "Czech Republic",
    "czechia": "Czech Republic",
    "bosnia": "Bosnia and Herzegovina",
    "bosnia & herzegovina": "Bosnia and Herzegovina",
    "bosnia and herzegovina": "Bosnia and Herzegovina",
    "turkiye": "Turkey",
    "türkiye": "Turkey",
    "curacao": "Curaçao",
    "curaçao": "Curaçao",
    "dr congo": "DR Congo",
    "congo dr": "DR Congo",
    "democratic republic of the congo": "DR Congo",
    "republic of ireland": "Ireland",
    "northern ireland": "Northern Ireland",
    "uae": "United Arab Emirates",
    "united arab emirates": "United Arab Emirates",
    "holland": "Netherlands",
    "the netherlands": "Netherlands",
    "brasil": "Brazil",
    "deutschland": "Germany",
    "espana": "Spain",
    "españa": "Spain",
    "cape verde islands": "Cape Verde",
}


def canonical_team_name(name: str | float | None) -> str:
    cleaned = normalize_team_name(name)
    if not cleaned:
        return ""
    key = cleaned.lower()
    return TEAM_ALIASES.get(key, cleaned)


@lru_cache(maxsize=1)
def known_teams_from_results(results_path: str) -> tuple[str, ...]:
    import pandas as pd

    try:
        frame = pd.read_csv(results_path)
    except FileNotFoundError:
        return tuple()

    teams: set[str] = set()
    for column in ("home_team", "away_team", "team_a", "team_b"):
        if column in frame.columns:
            teams.update(frame[column].dropna().astype(str).map(canonical_team_name))
    return tuple(sorted(teams))


def fuzzy_match_team(candidate: str, known_teams: list[str] | tuple[str, ...]) -> tuple[str | None, float]:
    """Return best team match and confidence score in [0, 1]."""
    canonical = canonical_team_name(candidate)
    known = [canonical_team_name(team) for team in known_teams]
    if canonical in known:
        return canonical, 1.0

    lowered = canonical.lower()
    for team in known:
        if team.lower() == lowered:
            return team, 0.98

    for team in known:
        if lowered in team.lower() or team.lower() in lowered:
            return team, 0.88

    tokens = set(re.findall(r"[a-zA-Z]+", lowered))
    best_team: str | None = None
    best_score = 0.0
    for team in known:
        team_tokens = set(re.findall(r"[a-zA-Z]+", team.lower()))
        if not tokens or not team_tokens:
            continue
        overlap = len(tokens & team_tokens) / max(len(tokens), len(team_tokens))
        if overlap > best_score:
            best_score = overlap
            best_team = team

    if best_score >= 0.6:
        return best_team, min(0.84, best_score)
    return None, 0.0
