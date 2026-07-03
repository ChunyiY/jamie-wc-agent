# Data files

Jamie downloads most datasets automatically. Use the app sidebar **Download latest data** or:

```bash
python scripts/download_data.py --all
```

| File | Source | Purpose |
|------|--------|---------|
| `results.csv` | [martj42/international_results](https://github.com/martj42/international_results) | Historical international matches |
| `goalscorers.csv` | martj42 | Scorer timelines |
| `shootouts.csv` | martj42 | Penalty shootout history |
| `fixtures.csv` | [openfootball/worldcup.json](https://github.com/openfootball/worldcup.json) | WC 2026 schedule |
| `wc_squads.csv`, `wc_per90_stats.csv` | [risingtransfers/world-cup-2026-data](https://github.com/risingtransfers/world-cup-2026-data) | Squads & club per-90 |
| `intl_match_stats.csv` | StatsBomb Open Data + WC2026 enriched | xG, shots, SOT |
| `fifa_rankings_wc2026.csv` | mominullptr WC2026 dataset | FIFA rankings |

Generated at runtime: `elo_ratings.csv`, `models/*.joblib`.
