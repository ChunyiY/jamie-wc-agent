#!/usr/bin/env python3
"""CLI helper to download football datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import APP_FULL_TITLE

from src.data_download import download_all, download_results, download_worldcup_fixtures


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Download datasets for {APP_FULL_TITLE}")
    parser.add_argument("--results", action="store_true", help="Download international results CSV")
    parser.add_argument("--fixtures", action="store_true", help="Download World Cup 2026 fixtures CSV")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--github-only", action="store_true", help="Skip Kaggle and use GitHub upstream")
    args = parser.parse_args()

    if not any([args.results, args.fixtures, args.all]):
        args.all = True

    prefer_kaggle = not args.github_only
    outputs = []
    if args.all:
        outputs = download_all(prefer_kaggle=prefer_kaggle)
    else:
        if args.results:
            outputs.append(download_results(prefer_kaggle=prefer_kaggle))
        if args.fixtures:
            outputs.append(download_worldcup_fixtures())

    for item in outputs:
        print(f"[{item.source}] {item.message}")
        for path in item.files:
            print(f"  - {path}")


if __name__ == "__main__":
    main()
