"""Squad roster team name alignment with fixtures and results."""

from pathlib import Path

import pandas as pd

from src.squad_data import load_squads_csv, team_squad_profile
from src.team_mapping import canonical_team_name


def test_cape_verde_squad_matches_fixture_name():
    squads = load_squads_csv(Path("data/wc_squads.csv"))
    assert "Cape Verde" in squads["country"].values
    profile = team_squad_profile(squads, "Cape Verde")
    assert profile is not None
    assert profile.squad_size >= 20
    names = {p.name for p in profile.players}
    assert "Dailon Livramento" in names


def test_fixture_teams_resolve_to_squad_countries():
    squads = load_squads_csv(Path("data/wc_squads.csv"))
    squad_teams = set(squads["country"].unique())
    fixtures = pd.read_csv("data/fixtures.csv")
    f_teams = set(fixtures["team_a"]) | set(fixtures["team_b"])
    missing = []
    for team in f_teams:
        canon = canonical_team_name(team)
        if canon not in squad_teams:
            missing.append(team)
    assert missing == [], f"fixtures without squad: {missing}"


def test_argentina_attack_per90_uses_forwards():
    from src.squad_data import load_per90_csv, merge_squads_per90

    squads = load_squads_csv(Path("data/wc_squads.csv"))
    per90 = load_per90_csv(Path("data/wc_per90_stats.csv"))
    merged = merge_squads_per90(squads, per90)
    profile = team_squad_profile(merged, "Argentina")
    assert profile is not None
    assert profile.top11_goals_per90 >= 0.35
