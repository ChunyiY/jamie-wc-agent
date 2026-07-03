"""Compliance checks, jurisdiction validation, and live-trading guards."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import requests

from src.config import AppConfig, load_config
from src.utils import setup_logger, utc_now_iso

COMPLIANCE_BANNER = (
    "This tool is for football analytics and paper-trading research. "
    "It does not guarantee profit. Live trading is disabled by default and must "
    "comply with all applicable laws, platform rules, and jurisdiction restrictions."
)

# Known restricted jurisdictions per Polymarket public documentation.
# This list is informational; the geoblock endpoint is authoritative at runtime.
KNOWN_BLOCKED_COUNTRIES = {
    "AU", "BE", "BY", "BI", "BR", "CF", "CD", "CU", "DE", "ET", "FR", "GB",
    "IR", "IQ", "IT", "KP", "LB", "LY", "MM", "NI", "NL", "RU", "SK", "SO",
    "SS", "SD", "SY", "UM", "US", "VE", "YE", "ZW",
}

KNOWN_CLOSE_ONLY_COUNTRIES = {"PL", "SG", "TH", "TW"}
KNOWN_FRONTEND_RESTRICTED = {"JP", "MT"}


class TradingMode(str, Enum):
    READ_ONLY_ANALYTICS = "read_only_analytics"
    PAPER_TRADING = "paper_trading"
    LIVE_TRADING_DISABLED = "live_trading_disabled"


class MarketPlatform(str, Enum):
    INTERNATIONAL = "international"
    US = "us"


@dataclass
class ComplianceStatus:
    mode: TradingMode
    market_platform: MarketPlatform
    jurisdiction_allowed: bool
    jurisdiction_reason: str
    geoblock_blocked: bool | None
    geoblock_country: str | None
    geoblock_region: str | None
    geoblock_error: str | None
    live_trading_enabled: bool
    warnings: list[str] = field(default_factory=list)
    can_trade_live: bool = False
    can_paper_trade: bool = True
    can_view_analytics: bool = True

    @property
    def is_restricted(self) -> bool:
        if self.market_platform == MarketPlatform.US:
            return not self.jurisdiction_allowed
        return not self.jurisdiction_allowed or self.geoblock_blocked is True


def _log_compliance_event(config: AppConfig, event: str, payload: dict[str, Any]) -> None:
    logger = setup_logger("compliance", config.compliance_log_path)
    logger.info("%s | %s", event, json.dumps(payload, default=str))


def resolve_market_platform(
    config: AppConfig | None = None,
    *,
    user_country: str | None = None,
    geoblock_country: str | None = None,
) -> MarketPlatform:
    """Choose Polymarket US vs international based on config and detected location."""
    config = config or load_config()
    explicit = (config.polymarket_platform or "").strip().lower()
    if explicit == MarketPlatform.US.value:
        return MarketPlatform.US
    if explicit == MarketPlatform.INTERNATIONAL.value:
        return MarketPlatform.INTERNATIONAL

    candidates = {
        (user_country or "").upper().strip(),
        (geoblock_country or "").upper().strip(),
        (config.user_country_code or "").upper().strip(),
    }
    if "US" in candidates:
        return MarketPlatform.US
    return MarketPlatform.INTERNATIONAL


def check_jurisdiction_allowed(
    country_code: str | None = None,
    config: AppConfig | None = None,
    *,
    market_platform: MarketPlatform | None = None,
) -> tuple[bool, str]:
    """
    Validate whether the declared or detected jurisdiction may use trading features.

    Returns (allowed, reason). Unknown jurisdiction defaults to analytics-only.
    """
    config = config or load_config()
    platform = market_platform or resolve_market_platform(config, user_country=country_code)
    code = (country_code or config.user_country_code or "").upper().strip()

    if platform == MarketPlatform.US:
        if code == "US" or not code:
            reason = (
                "United States user on Polymarket US (regulated). "
                "International polymarket.com trading is blocked in the US by design. "
                "Confirm your state eligibility on polymarket.us before placing real orders."
            )
            allowed = True
            _log_compliance_event(
                config,
                "jurisdiction_check",
                {"country": code or "US", "platform": platform.value, "allowed": allowed, "reason": reason},
            )
            return allowed, reason
        reason = (
            f"Polymarket US platform selected, but declared jurisdiction is {code or 'unknown'}. "
            "Analytics remain available; verify platform eligibility before manual trading."
        )
        _log_compliance_event(
            config,
            "jurisdiction_check",
            {"country": code, "platform": platform.value, "allowed": True, "reason": reason},
        )
        return True, reason

    if not code:
        reason = (
            "Jurisdiction unknown. Only read-only analytics and paper trading are enabled. "
            "Confirm your local laws before any real trading."
        )
        _log_compliance_event(config, "jurisdiction_check", {"country": None, "allowed": False, "reason": reason})
        return False, reason

    if code in KNOWN_BLOCKED_COUNTRIES:
        reason = (
            f"Jurisdiction {code} is listed as restricted on international Polymarket. "
            "US residents should use Polymarket US (set POLYMARKET_PLATFORM=us). "
            "This app will not enable live trading or suggest bypassing restrictions."
        )
        _log_compliance_event(config, "jurisdiction_check", {"country": code, "allowed": False, "reason": reason})
        return False, reason

    if code in KNOWN_CLOSE_ONLY_COUNTRIES:
        reason = (
            f"Jurisdiction {code} may be close-only on Polymarket. "
            "Live order placement should be treated as restricted."
        )
        _log_compliance_event(config, "jurisdiction_check", {"country": code, "allowed": False, "reason": reason})
        return False, reason

    if code in KNOWN_FRONTEND_RESTRICTED:
        reason = (
            f"Jurisdiction {code} has frontend restrictions on Polymarket. "
            "Use read-only analytics unless you independently confirm eligibility."
        )
        _log_compliance_event(config, "jurisdiction_check", {"country": code, "allowed": False, "reason": reason})
        return False, reason

    reason = f"Jurisdiction {code} is not in the known blocked list. Analytics and paper trading are available."
    _log_compliance_event(config, "jurisdiction_check", {"country": code, "allowed": True, "reason": reason})
    return True, reason


def check_polymarket_geoblock_status(config: AppConfig | None = None) -> dict[str, Any]:
    """
    Query the official Polymarket geoblock endpoint.

    This uses the user's real network path. No proxies, VPNs, or spoofing are used.
    """
    config = config or load_config()
    try:
        response = requests.get(config.polymarket_geoblock_url, timeout=config.request_timeout_seconds)
        response.raise_for_status()
        data = response.json()
        _log_compliance_event(config, "geoblock_check", {"success": True, "data": data})
        return {
            "success": True,
            "blocked": bool(data.get("blocked", False)),
            "ip": data.get("ip"),
            "country": data.get("country"),
            "region": data.get("region"),
            "error": None,
        }
    except requests.RequestException as exc:
        payload = {"success": False, "error": str(exc)}
        _log_compliance_event(config, "geoblock_check", payload)
        return {
            "success": False,
            "blocked": None,
            "ip": None,
            "country": None,
            "region": None,
            "error": str(exc),
        }


def evaluate_compliance(
    user_confirmed_jurisdiction: str | None = None,
    config: AppConfig | None = None,
) -> ComplianceStatus:
    """Aggregate compliance checks for UI and strategy layers."""
    config = config or load_config()
    geoblock = check_polymarket_geoblock_status(config)
    country = user_confirmed_jurisdiction or config.user_country_code or geoblock.get("country")
    market_platform = resolve_market_platform(
        config,
        user_country=country,
        geoblock_country=geoblock.get("country"),
    )
    jurisdiction_allowed, jurisdiction_reason = check_jurisdiction_allowed(
        country,
        config,
        market_platform=market_platform,
    )

    warnings = [
        "This application provides analytics and educational research support only.",
        "Past performance does not guarantee future results.",
        "Never trade with money you cannot afford to lose.",
    ]

    geoblock_blocked = geoblock.get("blocked")
    if market_platform == MarketPlatform.US:
        warnings.append(
            "Using Polymarket US (polymarket.us), the regulated US platform. "
            "Manual limit orders only — this app never auto-places trades."
        )
        if geoblock_blocked is True:
            warnings.append(
                "International polymarket.com geoblock shows blocked — expected for US users. "
                "Use polymarket.us for legal US access."
            )
        if geoblock.get("region"):
            warnings.append(
                f"Detected region: {geoblock.get('region')}. "
                "Confirm Polymarket US is available in your state before trading."
            )
    else:
        if geoblock_blocked is True:
            warnings.append(
                "International Polymarket geoblock reports your IP as restricted for order placement. "
                "Live trading remains disabled."
            )
        if country and str(country).upper() == "US":
            warnings.append(
                "You appear to be in the United States. "
                "Set POLYMARKET_PLATFORM=us to use the legal Polymarket US data source."
            )
    if geoblock.get("error"):
        warnings.append(
            "Could not verify international Polymarket geoblock status. "
            "Defaulting to conservative analytics settings."
        )
    if not jurisdiction_allowed:
        warnings.append(jurisdiction_reason)

    live_enabled = config.enable_live_trading
    international_block_applies = market_platform != MarketPlatform.US and geoblock_blocked is True
    can_trade_live = (
        live_enabled
        and jurisdiction_allowed
        and not international_block_applies
        and config.enable_live_trading
    )

    mode = TradingMode.PAPER_TRADING
    if not jurisdiction_allowed or international_block_applies:
        mode = TradingMode.READ_ONLY_ANALYTICS
    if not live_enabled and jurisdiction_allowed and not international_block_applies:
        mode = TradingMode.LIVE_TRADING_DISABLED

    status = ComplianceStatus(
        mode=mode,
        market_platform=market_platform,
        jurisdiction_allowed=jurisdiction_allowed,
        jurisdiction_reason=jurisdiction_reason,
        geoblock_blocked=geoblock_blocked,
        geoblock_country=geoblock.get("country"),
        geoblock_region=geoblock.get("region"),
        geoblock_error=geoblock.get("error"),
        live_trading_enabled=live_enabled,
        warnings=warnings,
        can_trade_live=can_trade_live,
        can_paper_trade=True,
        can_view_analytics=True,
    )
    _log_compliance_event(config, "compliance_summary", status.__dict__)
    return status


def place_live_limit_order_stub(
    *,
    market_id: str,
    side: str,
    price: float,
    size: float,
    user_confirmed: bool = False,
    config: AppConfig | None = None,
) -> None:
    """
    Phase 2 stub for compliant live trading architecture.

    Live trading is disabled unless every guard passes. Limit orders only.
    """
    config = config or load_config()
    status = evaluate_compliance(config=config)

    if not config.enable_live_trading:
        raise NotImplementedError("Live trading is disabled. Set ENABLE_LIVE_TRADING=true to enable the stub.")
    if not status.jurisdiction_allowed:
        raise NotImplementedError("Live trading blocked: jurisdiction not confirmed as allowed.")
    if status.market_platform != MarketPlatform.US and status.geoblock_blocked is not False:
        raise NotImplementedError("Live trading blocked: international Polymarket geoblock check did not pass.")
    if not user_confirmed:
        raise NotImplementedError("Live trading blocked: manual user confirmation required.")
    if side.lower() not in {"yes", "no", "buy", "sell"}:
        raise ValueError("Only limit orders on YES/NO outcomes are supported in the stub interface.")
    raise NotImplementedError(
        "Live trading architecture stub only. No authenticated order placement is implemented in Phase 1/2."
    )
