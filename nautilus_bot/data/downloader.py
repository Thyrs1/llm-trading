from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import requests
from requests import Response

from nautilus_trader.adapters.binance import BINANCE, BinanceFuturesInstrumentProvider
from nautilus_trader.adapters.binance.common.enums import BinanceAccountType
from nautilus_trader.adapters.binance.http.client import BinanceHttpClient
from nautilus_trader.common.component import LiveClock
from nautilus_trader.common.config import InstrumentProviderConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import PriceType
from nautilus_trader.model.identifiers import InstrumentId, Venue
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.persistence.catalog.parquet import ParquetDataCatalog

from nautilus_bot.config import BotSettings, load_settings

logger = logging.getLogger(__name__)


class BinanceDownloaderError(RuntimeError):
    """Raised when Binance kline download fails."""


class BinanceKlineDownloader:
    """Lightweight Binance Futures kline downloader."""

    BASE_URL = "https://fapi.binance.com/fapi/v1/klines"
    LIMIT = 1500  # Binance maximum rows per request

    def __init__(self, symbol: str, interval: str, start: datetime, end: datetime) -> None:
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        self.symbol = symbol
        self.interval = interval
        self.start = start
        self.end = end

    def fetch(self) -> List[List]:
        rows: List[List] = []
        start_ms = int(self.start.timestamp() * 1000)
        end_ms = int(self.end.timestamp() * 1000)
        current = start_ms

        while current < end_ms:
            params = {
                "symbol": self.symbol.upper(),
                "interval": self.interval,
                "startTime": current,
                "endTime": end_ms,
                "limit": self.LIMIT,
            }
            response: Response = requests.get(self.BASE_URL, params=params, timeout=10)
            if response.status_code != 200:
                raise BinanceDownloaderError(
                    f"Binance API error {response.status_code}: {response.text}",
                )
            batch = response.json()
            if not batch:
                break
            rows.extend(batch)
            last_open_time = batch[-1][0]
            current = last_open_time + 1
            if len(batch) < self.LIMIT:
                break
        return rows


def _infer_price_precision(value: str) -> int:
    if "." not in value:
        return 0
    return len(value.split(".", 1)[1])


def klines_to_bars(rows: Iterable[List], bar_type_str: str) -> List[Bar]:
    bar_type = BarType.from_str(bar_type_str)
    bars: List[Bar] = []
    for row in rows:
        open_time = int(row[0]) * 1_000_000
        close_time = int(row[6]) * 1_000_000
        open_price = Price(float(row[1]), _infer_price_precision(row[1]))
        high_price = Price(float(row[2]), _infer_price_precision(row[2]))
        low_price = Price(float(row[3]), _infer_price_precision(row[3]))
        close_price = Price(float(row[4]), _infer_price_precision(row[4]))
        volume = Quantity(float(row[5]), _infer_price_precision(row[5]))

        bars.append(
            Bar(
                bar_type=bar_type,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
                ts_event=close_time,
                ts_init=open_time,
                is_revision=False,
            ),
        )
    return bars


def _has_existing_data(catalog_path: Path, bar_type: str) -> bool:
    """检查指定 bar_type 是否已存在 parquet 数据。"""

    if not catalog_path.exists():
        return False
    bar_dir = catalog_path / "data" / "bar" / bar_type.replace(":", "-")
    return any(bar_dir.glob("*.parquet"))


def _resolve_account_type(value: Optional[str]) -> BinanceAccountType:
    mapping = {
        "SPOT": BinanceAccountType.SPOT,
        "MARGIN": BinanceAccountType.MARGIN,
        "ISOLATED_MARGIN": BinanceAccountType.ISOLATED_MARGIN,
        "USDT_FUTURE": BinanceAccountType.USDT_FUTURES,
        "USDT_FUTURES": BinanceAccountType.USDT_FUTURES,
        "COIN_FUTURE": BinanceAccountType.COIN_FUTURES,
        "COIN_FUTURES": BinanceAccountType.COIN_FUTURES,
    }
    if not value:
        return BinanceAccountType.USDT_FUTURES
    key = value.strip().upper()
    if key in mapping:
        return mapping[key]
    try:
        return BinanceAccountType[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported Binance account type: {value}") from exc


def _fetch_instrument(
    settings: BotSettings,
    instrument_id: str,
    account_type: BinanceAccountType,
) -> Optional["Instrument"]:
    if not settings.binance.api_key or not settings.binance.api_secret:
        logger.warning(
            "Binance API key/secret not configured, skip instrument metadata download for %s",
            instrument_id,
        )
        return None
    try:
        clock = LiveClock()
        client = BinanceHttpClient(
            clock=clock,
            api_key=settings.binance.api_key,
            api_secret=settings.binance.api_secret,
            base_url=settings.binance.base_http_url or "https://fapi.binance.com",
        )
        provider = BinanceFuturesInstrumentProvider(
            client=client,
            clock=clock,
            account_type=account_type,
            config=InstrumentProviderConfig(load_all=False),
            venue=Venue(str(BINANCE)),
        )
        inst_id = InstrumentId.from_str(instrument_id)
        provider.load_ids([inst_id])
        instrument = provider.find(inst_id)
        if instrument is None:
            logger.warning("Instrument %s not returned by provider.", instrument_id)
        return instrument
    except Exception as exc:  # pragma: no cover - 网络或凭证错误
        logger.warning("Failed to load instrument %s from provider: %s", instrument_id, exc)
        return None


def ensure_catalog_data(
    catalog_path: Path,
    instrument_id: str,
    bar_type: str,
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
) -> Path:
    catalog_path = catalog_path.resolve()
    catalog_path.mkdir(parents=True, exist_ok=True)
    catalog = ParquetDataCatalog(path=str(catalog_path))

    settings = load_settings()
    account_type = _resolve_account_type(settings.binance.account_type)
    instrument = _fetch_instrument(settings, instrument_id, account_type)
    if instrument is not None:
        try:
            catalog.write_data([instrument])
            logger.info("Stored instrument metadata for %s", instrument_id)
        except Exception as exc:  # pragma: no cover - 已存在等情况
            logger.warning("Instrument metadata write skipped for %s: %s", instrument_id, exc)

    if _has_existing_data(catalog_path, bar_type):
        logger.info("Catalog %s already contains data for %s", catalog_path, bar_type)
        return catalog_path

    logger.info(
        "Downloading Binance data for %s [%s - %s]",
        symbol,
        start.isoformat(),
        end.isoformat(),
    )
    downloader = BinanceKlineDownloader(symbol=symbol, interval=interval, start=start, end=end)
    rows = downloader.fetch()
    if not rows:
        raise BinanceDownloaderError("No data returned for requested range")

    bars = klines_to_bars(rows, bar_type_str=bar_type)
    catalog.write_data(bars)
    logger.info(
        "Stored %d bars in catalog %s for %s",
        len(bars),
        catalog_path,
        bar_type,
    )
    return catalog_path


def _parse_cli_args() -> tuple[str, str, datetime, datetime, Path]:
    import argparse

    parser = argparse.ArgumentParser(description="Download Binance kline data into Parquet catalog")
    parser.add_argument("--symbol", required=True, help="Binance symbol, e.g. BTCUSDT")
    parser.add_argument("--interval", default="5m", help="Kline interval, e.g. 1m/5m/1h")
    parser.add_argument("--start", required=True, help="Start time (YYYY-MM-DD or ISO8601)")
    parser.add_argument("--end", required=True, help="End time (YYYY-MM-DD or ISO8601)")
    parser.add_argument("--catalog", default="./data/catalog", help="Catalog root directory")
    args = parser.parse_args()

    def _parse_dt(value: str) -> datetime:
        if len(value) == 10:
            return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    return (
        args.symbol,
        args.interval,
        _parse_dt(args.start),
        _parse_dt(args.end),
        Path(args.catalog),
    )


def _interval_to_bar_suffix(interval: str) -> str:
    raw = interval.strip()
    if not raw:
        raise ValueError("Interval cannot be empty")

    suffix = raw[-1]
    value = raw[:-1]
    suffix_lower = suffix.lower()

    if suffix_lower == "m":
        unit = "MONTH" if suffix.isupper() else "MINUTE"
    elif suffix_lower == "h":
        unit = "HOUR"
    elif suffix_lower == "d":
        unit = "DAY"
    elif suffix_lower == "w":
        unit = "WEEK"
    else:
        raise ValueError(f"Unsupported interval format: {interval}")
    if not value.isdigit():
        raise ValueError(f"Invalid interval value: {interval}")
    return f"{int(value)}-{unit}"


def main() -> None:  # pragma: no cover - CLI
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    symbol, interval, start, end, catalog_path = _parse_cli_args()

    instrument_id = f"{symbol}-PERP.BINANCE"
    bar_suffix = _interval_to_bar_suffix(interval)
    bar_type = f"{instrument_id}-{bar_suffix}-LAST-INTERNAL"
    ensure_catalog_data(
        catalog_path=catalog_path,
        instrument_id=instrument_id,
        bar_type=bar_type,
        symbol=symbol,
        interval=interval,
        start=start,
        end=end,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
