"""Aggregate StatsBomb Open Data for international tournaments (xG, shots, SOT)."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pandas as pd

from src.team_mapping import canonical_team_name

SB_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
COMPETITIONS_URL = f"{SB_BASE}/competitions.json"

# Recent men's international tournaments with full event coverage
TARGET_COMPETITIONS = {
    "FIFA World Cup",
    "UEFA Euro",
    "Copa America",
    "African Cup of Nations",
}
RECENT_SEASONS = {"2018", "2022", "2020", "2024", "2023", "2019", "2021"}


def _fetch_json(url: str, timeout: int = 60):
    with urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


def _canon_team(name: str) -> str:
    aliases = {
        "Korea Republic": "South Korea",
        "USA": "United States",
        "Côte d'Ivoire": "Ivory Coast",
        "Cote d'Ivoire": "Ivory Coast",
        "IR Iran": "Iran",
    }
    return canonical_team_name(aliases.get(str(name), str(name)))


def _aggregate_events(events: list[dict], home_team: str, away_team: str) -> tuple[dict, dict]:
    """Return per-team shot/xG aggregates from StatsBomb events."""
    home = _canon_team(home_team)
    away = _canon_team(away_team)
    buckets = {
        home: {"shots": 0, "sot": 0, "xg": 0.0},
        away: {"shots": 0, "sot": 0, "xg": 0.0},
    }
    for ev in events:
        if ev.get("type", {}).get("name") != "Shot":
            continue
        team = _canon_team(ev.get("team", {}).get("name", ""))
        if team not in buckets:
            continue
        shot = ev.get("shot", {}) or {}
        buckets[team]["shots"] += 1
        outcome = shot.get("outcome", {}).get("name", "")
        if outcome in {"Goal", "Saved", "Saved to Post"}:
            buckets[team]["sot"] += 1
        xg = shot.get("statsbomb_xg")
        if xg is not None:
            buckets[team]["xg"] += float(xg)
    return buckets[home], buckets[away]


def download_statsbomb_international(
    data_dir: Path,
    *,
    max_matches: int = 400,
) -> pd.DataFrame:
    """
    Build intl_match_stats.csv from StatsBomb open JSON (WC, Euro, Copa, AFCON).

    Each row = one team in one match with xG, shots, shots on target.
    """
    try:
        competitions = _fetch_json(COMPETITIONS_URL)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not fetch StatsBomb competitions: {exc}") from exc

    targets = [
        c
        for c in competitions
        if c.get("competition_international")
        and c.get("competition_name") in TARGET_COMPETITIONS
        and c.get("competition_gender") == "male"
        and str(c.get("season_name", "")) in RECENT_SEASONS
        and c.get("match_available")
    ]

    rows: list[dict] = []
    processed = 0
    for comp in targets:
        cid, sid = comp["competition_id"], comp["season_id"]
        comp_name = comp["competition_name"]
        season = comp["season_name"]
        try:
            matches = _fetch_json(f"{SB_BASE}/matches/{cid}/{sid}.json")
        except (HTTPError, URLError, TimeoutError):
            continue
        for m in matches:
            if processed >= max_matches:
                break
            mid = m.get("match_id")
            if not mid or m.get("match_status") not in {"available", "complete", "finished"}:
                # StatsBomb uses match_status field inconsistently; try events anyway
                pass
            home = m.get("home_team", {}).get("home_team_name", "")
            away = m.get("away_team", {}).get("away_team_name", "")
            if not home or not away:
                continue
            try:
                events = _fetch_json(f"{SB_BASE}/events/{mid}.json")
            except (HTTPError, URLError, TimeoutError):
                continue
            if not events:
                continue
            h_stats, a_stats = _aggregate_events(events, home, away)
            date = pd.to_datetime(m.get("match_date"), errors="coerce", utc=True)
            hs = m.get("home_score")
            aws = m.get("away_score")
            for team, opp, is_home, gf, ga, st in (
                (_canon_team(home), _canon_team(away), True, hs, aws, h_stats),
                (_canon_team(away), _canon_team(home), False, aws, hs, a_stats),
            ):
                shots = st["shots"]
                sot = st["sot"]
                rows.append(
                    {
                        "date": date,
                        "team": team,
                        "opponent": opp,
                        "is_home": is_home,
                        "goals_for": int(gf) if gf is not None else 0,
                        "goals_against": int(ga) if ga is not None else 0,
                        "xg": round(st["xg"], 3),
                        "shots": shots,
                        "shots_on_target": sot,
                        "possession_pct": float("nan"),
                        "corners": 0,
                        "tournament": comp_name,
                        "season": str(season),
                        "source": "statsbomb",
                        "match_id": mid,
                    }
                )
            processed += 1
        if processed >= max_matches:
            break

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out["sot_pct"] = (out["shots_on_target"] / out["shots"].replace(0, pd.NA)).fillna(0.0)
    return out
