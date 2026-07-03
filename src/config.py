"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"

load_dotenv(PROJECT_ROOT / ".env")


class AppConfig(BaseModel):
    """Central configuration for the analytics dashboard."""

    app_mode: str = Field(default="analytics")
    enable_live_trading: bool = Field(default=False)

    user_country_code: str | None = Field(default=None)

    polymarket_platform: str | None = Field(default=None)
    polymarket_us_key_id: str | None = Field(default=None)
    polymarket_us_secret_key: str | None = Field(default=None)

    polymarket_gamma_url: str = Field(default="https://gamma-api.polymarket.com")
    polymarket_clob_url: str = Field(default="https://clob.polymarket.com")
    polymarket_geoblock_url: str = Field(default="https://polymarket.com/api/geoblock")

    default_bankroll: float = Field(default=70.0)
    default_min_edge: float = Field(default=0.07)
    default_max_spread: float = Field(default=0.05)
    default_min_liquidity: float = Field(default=100.0)
    default_kelly_multiplier: float = Field(default=0.08)
    default_max_risk_per_trade: float = Field(default=0.005)
    default_max_exposure_per_match: float = Field(default=0.015)
    default_max_total_exposure: float = Field(default=0.06)
    default_min_model_confidence: float = Field(default=0.58)
    min_ticket_size: float = Field(default=1.0)
    liquidity_participation_cap: float = Field(default=0.05)
    max_price_move_bps: float = Field(default=200.0)

    results_csv: Path = Field(default=DATA_DIR / "results.csv")
    fixtures_csv: Path = Field(default=DATA_DIR / "fixtures.csv")
    goalscorers_csv: Path = Field(default=DATA_DIR / "goalscorers.csv")
    wc_squads_csv: Path = Field(default=DATA_DIR / "wc_squads.csv")
    wc_per90_csv: Path = Field(default=DATA_DIR / "wc_per90_stats.csv")
    paper_trades_csv: Path = Field(default=DATA_DIR / "paper_trades.csv")
    elo_table_path: Path = Field(default=DATA_DIR / "elo_ratings.csv")
    compliance_log_path: Path = Field(default=LOGS_DIR / "compliance.log")

    elo_initial_rating: float = Field(default=1500.0)
    elo_k_factor: float = Field(default=32.0)
    elo_home_advantage: float = Field(default=100.0)

    parser_confidence_threshold: float = Field(default=0.85)
    slippage_buffer: float = Field(default=0.02)
    model_uncertainty_buffer: float = Field(default=0.03)

    request_timeout_seconds: int = Field(default=15)
    elo_match_limit: int = Field(default=20_000)
    modeling_match_limit: int = Field(default=6_000)
    fast_train_skip_calibration: bool = Field(default=False)
    walk_forward_retrain_every: int = Field(default=500)


def load_config() -> AppConfig:
    """Load configuration from environment with safe defaults."""
    return AppConfig(
        app_mode=os.getenv("APP_MODE", "analytics"),
        enable_live_trading=os.getenv("ENABLE_LIVE_TRADING", "false").lower() == "true",
        user_country_code=os.getenv("USER_COUNTRY_CODE") or None,
        polymarket_platform=os.getenv("POLYMARKET_PLATFORM") or None,
        polymarket_us_key_id=os.getenv("POLYMARKET_KEY_ID") or None,
        polymarket_us_secret_key=os.getenv("POLYMARKET_SECRET_KEY") or None,
        polymarket_gamma_url=os.getenv("POLYMARKET_GAMMA_URL", "https://gamma-api.polymarket.com"),
        polymarket_clob_url=os.getenv("POLYMARKET_CLOB_URL", "https://clob.polymarket.com"),
        polymarket_geoblock_url=os.getenv("POLYMARKET_GEOBLOCK_URL", "https://polymarket.com/api/geoblock"),
        default_bankroll=float(os.getenv("DEFAULT_BANKROLL", "70")),
        default_min_edge=float(os.getenv("DEFAULT_MIN_EDGE", "0.07")),
        default_max_spread=float(os.getenv("DEFAULT_MAX_SPREAD", "0.05")),
        default_min_liquidity=float(os.getenv("DEFAULT_MIN_LIQUIDITY", "100")),
        default_kelly_multiplier=float(os.getenv("DEFAULT_KELLY_MULTIPLIER", "0.08")),
        default_max_risk_per_trade=float(os.getenv("DEFAULT_MAX_RISK_PER_TRADE", "0.005")),
        default_max_exposure_per_match=float(os.getenv("DEFAULT_MAX_EXPOSURE_PER_MATCH", "0.015")),
        default_max_total_exposure=float(os.getenv("DEFAULT_MAX_TOTAL_EXPOSURE", "0.06")),
        min_ticket_size=float(os.getenv("MIN_TICKET_SIZE", "1.0")),
        max_price_move_bps=float(os.getenv("MAX_PRICE_MOVE_BPS", "200")),
        elo_match_limit=int(os.getenv("ELO_MATCH_LIMIT", "20000")),
        modeling_match_limit=int(os.getenv("MODELING_MATCH_LIMIT", "6000")),
        fast_train_skip_calibration=os.getenv("FAST_TRAIN_SKIP_CALIBRATION", "false").lower() == "true",
    )


def ensure_directories() -> None:
    """Create required runtime directories."""
    for path in (DATA_DIR, LOGS_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)
