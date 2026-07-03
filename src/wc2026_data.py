"""FIFA World Cup 2026 enriched data (xG, shots on target) from mominullptr open dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.team_mapping import canonical_team_name

WC2026_BASE = "https://raw.githubusercontent.com/mominullptr/FIFA-World-Cup-2026-Dataset/main"
WC2026_FILES = {
    "wc2026_teams.csv": f"{WC2026_BASE}/teams.csv",
    "wc2026_matches_raw.csv": f"{WC2026_BASE}/matches.csv",
    "wc2026_match_team_stats_raw.csv": f"{WC2026_BASE}/match_team_stats.csv",
    "wc2026_squads_players.csv": f"{WC2026_BASE}/squads_and_players.csv",
}

# Dataset naming → results.csv canonical names
WC2026_TEAM_ALIASES = {
    "Cabo Verde": "Cape Verde",
    "Czechia": "Czech Republic",
    "USA": "United States",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Türkiye": "Turkey",
    "Congo DR": "DR Congo",
}


def _canon_team(name: str) -> str:
    raw = str(name).strip()
    return canonical_team_name(WC2026_TEAM_ALIASES.get(raw, raw))


def load_wc2026_teams(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["team"] = frame["team_name"].map(_canon_team)
    return frame


def build_wc2026_match_stats(data_dir: Path) -> pd.DataFrame:
    """Merge matches + per-team stats into one row per team per match."""
    teams_path = data_dir / "wc2026_teams.csv"
    matches_path = data_dir / "wc2026_matches_raw.csv"
    stats_path = data_dir / "wc2026_match_team_stats_raw.csv"
    if not all(p.exists() for p in (teams_path, matches_path, stats_path)):
        return pd.DataFrame()

    teams = load_wc2026_teams(teams_path)
    team_map = dict(zip(teams["team_id"], teams["team"]))
    matches = pd.read_csv(matches_path)
    stats = pd.read_csv(stats_path)

    rows: list[dict] = []
    for _, m in matches.iterrows():
        if str(m.get("status", "")).lower() != "completed":
            continue
        mid = int(m["match_id"])
        home_id, away_id = int(m["home_team_id"]), int(m["away_team_id"])
        home = team_map.get(home_id, "")
        away = team_map.get(away_id, "")
        if not home or not away:
            continue
        date = pd.to_datetime(m["date"], errors="coerce", utc=True)
        mstats = stats[stats["match_id"] == mid]
        for _, s in mstats.iterrows():
            tid = int(s["team_id"])
            team = team_map.get(tid, "")
            if not team:
                continue
            opponent = away if team == home else home
            is_home = team == home
            hs = m.get("home_score")
            aws = m.get("away_score")
            gf = int(hs) if is_home else int(aws)
            ga = int(aws) if is_home else int(hs)
            shots = int(s.get("total_shots", 0) or 0)
            sot = int(s.get("shots_on_target", 0) or 0)
            xg = float(m["home_xg"]) if is_home else float(m["away_xg"])
            if pd.isna(xg):
                xg = 0.0
            rows.append(
                {
                    "date": date,
                    "team": team,
                    "opponent": opponent,
                    "is_home": is_home,
                    "goals_for": gf,
                    "goals_against": ga,
                    "xg": xg,
                    "shots": shots,
                    "shots_on_target": sot,
                    "possession_pct": float(s.get("possession_pct", 0) or 0),
                    "corners": int(s.get("corners", 0) or 0),
                    "tournament": "FIFA World Cup",
                    "season": "2026",
                    "source": str(s.get("data_source", "wc2026")),
                }
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out["sot_pct"] = (out["shots_on_target"] / out["shots"].replace(0, pd.NA)).fillna(0.0)
    return out


def build_wc2026_fifa_rankings(data_dir: Path) -> pd.DataFrame:
    teams = load_wc2026_teams(data_dir / "wc2026_teams.csv")
    if teams.empty:
        return pd.DataFrame()
    return teams[
        ["team", "fifa_ranking_pre_tournament", "elo_rating", "confederation", "group_letter"]
    ].rename(
        columns={
            "fifa_ranking_pre_tournament": "fifa_rank",
            "elo_rating": "fifa_dataset_elo",
            "group_letter": "wc_group",
        }
    )
