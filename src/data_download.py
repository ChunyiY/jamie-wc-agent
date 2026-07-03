"""Download international football datasets from Kaggle or official GitHub upstream."""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests

from src.config import DATA_DIR, load_config
from src.squad_data import PER90_URL, SQUADS_URL
from src.statsbomb_intl import download_statsbomb_international
from src.wc2026_data import (
    WC2026_FILES,
    build_wc2026_fifa_rankings,
    build_wc2026_match_stats,
)
from src.match_stats import MATCH_STATS_FILENAME, FIFA_RANKINGS_FILENAME

KAGGLE_DATASET = "martj42/international-football-results-from-1872-to-2017"
GITHUB_RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
GITHUB_GOALSCORERS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv"
GITHUB_SHOOTOUTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv"
WORLDCUP_FIXTURES_URL = "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"


@dataclass
class DownloadResult:
    source: str
    files: list[str]
    message: str


def _download_file(url: str, destination: Path, timeout: int = 120) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=timeout, stream=True)
    response.raise_for_status()
    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)


def download_from_github(data_dir: Path | None = None) -> DownloadResult:
    data_dir = data_dir or DATA_DIR
    files = {
        "results.csv": GITHUB_RESULTS_URL,
        "goalscorers.csv": GITHUB_GOALSCORERS_URL,
        "shootouts.csv": GITHUB_SHOOTOUTS_URL,
    }
    saved: list[str] = []
    for name, url in files.items():
        path = data_dir / name
        _download_file(url, path)
        saved.append(str(path))
    return DownloadResult(
        source="github",
        files=saved,
        message="Downloaded official upstream dataset from martj42/international_results (same source as Kaggle).",
    )


def download_squad_data(data_dir: Path | None = None) -> DownloadResult:
    """Download WC 2026 squads and per-90 club stats (risingtransfers open data)."""
    data_dir = data_dir or DATA_DIR
    files = {
        "wc_squads.csv": SQUADS_URL,
        "wc_per90_stats.csv": PER90_URL,
    }
    saved: list[str] = []
    for name, url in files.items():
        path = data_dir / name
        _download_file(url, path)
        saved.append(str(path))
    return DownloadResult(
        source="risingtransfers",
        files=saved,
        message="Downloaded World Cup 2026 squads (1,363 players) and per-90 club stats.",
    )


def download_from_kaggle(data_dir: Path | None = None) -> DownloadResult:
    data_dir = data_dir or DATA_DIR
    if shutil.which("kaggle") is None:
        raise RuntimeError("Kaggle CLI not installed. Run: pip install kaggle")

    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(data_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    for archive in data_dir.glob("*.zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(data_dir)
        archive.unlink(missing_ok=True)

    saved = [str(p) for p in data_dir.glob("*.csv")]
    return DownloadResult(
        source="kaggle",
        files=saved,
        message=f"Downloaded dataset from Kaggle: {KAGGLE_DATASET}",
    )


def download_results(prefer_kaggle: bool = True, data_dir: Path | None = None) -> DownloadResult:
    data_dir = data_dir or DATA_DIR
    if prefer_kaggle and shutil.which("kaggle") is not None:
        try:
            return download_from_kaggle(data_dir)
        except (RuntimeError, subprocess.CalledProcessError, FileNotFoundError):
            pass
    return download_from_github(data_dir)


def download_worldcup_fixtures(data_dir: Path | None = None) -> DownloadResult:
    import csv
    import json

    data_dir = data_dir or DATA_DIR
    temp_json = data_dir / "worldcup2026.json"
    _download_file(WORLDCUP_FIXTURES_URL, temp_json)

    with temp_json.open() as handle:
        payload = json.load(handle)

    rows = []
    for match in payload.get("matches", []):
        team1 = str(match.get("team1", "")).strip()
        team2 = str(match.get("team2", "")).strip()
        if not team1 or not team2:
            continue
        if team1[0] in {"W", "L"} or team2[0] in {"W", "L"}:
            continue
        round_name = str(match.get("round", ""))
        stage = "group" if "matchday" in round_name.lower() or "group" in round_name.lower() else "knockout"
        rows.append(
            {
                "date": match.get("date"),
                "team_a": team1,
                "team_b": team2,
                "neutral": True,
                "tournament": "FIFA World Cup",
                "stage": stage,
                "group": match.get("group", ""),
                "round": round_name,
                "ground": match.get("ground", ""),
                "time": match.get("time", ""),
            }
        )

    fixtures_path = data_dir / "fixtures.csv"
    fieldnames = ["date", "team_a", "team_b", "neutral", "tournament", "stage", "group", "round", "ground", "time"]
    with fixtures_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp_json.unlink(missing_ok=True)
    return DownloadResult(
        source="openfootball",
        files=[str(fixtures_path)],
        message=f"Downloaded {len(rows)} World Cup 2026 fixtures from openfootball/worldcup.json.",
    )


def download_wc2026_enriched(data_dir: Path | None = None) -> DownloadResult:
    """WC 2026 xG, shots on target, FIFA rankings (mominullptr open dataset)."""
    import pandas as pd

    data_dir = data_dir or DATA_DIR
    saved: list[str] = []
    for name, url in WC2026_FILES.items():
        path = data_dir / name
        _download_file(url, path)
        saved.append(str(path))

    rankings = build_wc2026_fifa_rankings(data_dir)
    rankings_path = data_dir / FIFA_RANKINGS_FILENAME
    if not rankings.empty:
        rankings.to_csv(rankings_path, index=False)
        saved.append(str(rankings_path))

    wc_stats = build_wc2026_match_stats(data_dir)
    intl_path = data_dir / MATCH_STATS_FILENAME
    if not wc_stats.empty:
        if intl_path.exists():
            existing = pd.read_csv(intl_path)
            combined = pd.concat([existing, wc_stats], ignore_index=True)
            combined = combined.drop_duplicates(subset=["date", "team", "opponent", "source"], keep="last")
            combined.to_csv(intl_path, index=False)
        else:
            wc_stats.to_csv(intl_path, index=False)
        saved.append(str(intl_path))

    return DownloadResult(
        source="mominullptr/wc2026",
        files=saved,
        message=f"Downloaded WC 2026 enriched data ({len(wc_stats)} team-match rows with xG/SOT).",
    )


def download_statsbomb_intl_data(data_dir: Path | None = None, *, max_matches: int = 250) -> DownloadResult:
    """StatsBomb open data: international tournament xG + shots (2018–2024)."""
    import pandas as pd

    data_dir = data_dir or DATA_DIR
    frame = download_statsbomb_international(data_dir, max_matches=max_matches)
    if frame.empty:
        return DownloadResult(source="statsbomb", files=[], message="No StatsBomb international rows downloaded.")

    path = data_dir / MATCH_STATS_FILENAME
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, frame], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "team", "opponent", "source"], keep="last")
        combined.to_csv(path, index=False)
    else:
        frame.to_csv(path, index=False)

    return DownloadResult(
        source="statsbomb",
        files=[str(path)],
        message=f"Aggregated {len(frame)} team-match rows from StatsBomb international tournaments.",
    )


def download_all(prefer_kaggle: bool = True, *, include_statsbomb: bool = True) -> list[DownloadResult]:
    config = load_config()
    data_dir = config.results_csv.parent
    results = [
        download_results(prefer_kaggle=prefer_kaggle, data_dir=data_dir),
        download_worldcup_fixtures(data_dir),
        download_squad_data(data_dir),
        download_wc2026_enriched(data_dir),
    ]
    if include_statsbomb:
        try:
            results.append(download_statsbomb_intl_data(data_dir, max_matches=250))
        except RuntimeError as exc:
            results.append(DownloadResult(source="statsbomb", files=[], message=str(exc)))
    return results
