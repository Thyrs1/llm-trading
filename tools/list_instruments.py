"""列出 Binance 可用合约的辅助脚本。

优先尝试通过 Nautilus 的 BinanceInstrumentProvider 拉取；若未配置 API Key 或
凭证无效，则回退到公开 REST 接口（`fapi/v1/exchangeInfo` 等）。

运行示例：
    python tools/list_instruments.py --account-type usdt_futures --limit 20
"""

from __future__ import annotations

import argparse
from typing import Iterable

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests

from nautilus_trader.adapters.binance import BINANCE, BinanceFuturesInstrumentProvider
from nautilus_trader.adapters.binance.common.enums import BinanceAccountType
from nautilus_trader.adapters.binance.http.client import BinanceHttpClient
from nautilus_trader.common.component import LiveClock
from nautilus_trader.common.config import InstrumentProviderConfig
from nautilus_trader.model.identifiers import Venue

from nautilus_bot.config import BotSettings, load_settings

# 默认 Binance 永续 HTTP 端点
DEFAULT_HTTP_URL = "https://fapi.binance.com"
SPOT_EXCHANGE_INFO = "https://api.binance.com/api/v3/exchangeInfo"
USDT_FUTURES_EXCHANGE_INFO = "https://fapi.binance.com/fapi/v1/exchangeInfo"
COIN_FUTURES_EXCHANGE_INFO = "https://dapi.binance.com/dapi/v1/exchangeInfo"


def _resolve_account_type(value: str | None) -> BinanceAccountType:
    if not value:
        return BinanceAccountType.USDT_FUTURES
    normalized = value.strip().upper()
    mapping = {
        "USDT_FUTURE": "USDT_FUTURES",
        "USDT_FUTURES": "USDT_FUTURES",
        "COIN_FUTURE": "COIN_FUTURES",
        "COIN_FUTURES": "COIN_FUTURES",
        "SPOT": "SPOT",
        "MARGIN": "MARGIN",
        "ISOLATED_MARGIN": "ISOLATED_MARGIN",
    }
    key = mapping.get(normalized, normalized)
    try:
        return BinanceAccountType[key]
    except KeyError as exc:  # pragma: no cover - 输入非法
        raise ValueError(f"Unsupported account type: {value}") from exc


def _iter_instruments(settings: BotSettings, account_type: BinanceAccountType) -> Iterable[str]:
    """使用优先级：API Provider -> 公共 REST，迭代可用合约 ID。"""

    provider_ids: list[str] | None = None
    if settings.binance.api_key and settings.binance.api_secret:
        try:
            provider_ids = list(_list_by_provider(settings, account_type))
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️ Provider 加载失败（{exc}），转为公共 REST 模式")

    if provider_ids is not None and provider_ids:
        yield from provider_ids
        return

    yield from _list_by_rest(account_type)


def _list_by_provider(settings: BotSettings, account_type: BinanceAccountType) -> Iterable[str]:
    clock = LiveClock()
    http_client = BinanceHttpClient(
        clock=clock,
        api_key=settings.binance.api_key or "",
        api_secret=settings.binance.api_secret or "",
        base_url=settings.binance.base_http_url or DEFAULT_HTTP_URL,
    )
    provider = BinanceFuturesInstrumentProvider(
        client=http_client,
        clock=clock,
        account_type=account_type,
        config=InstrumentProviderConfig(load_all=True),
        venue=Venue(str(BINANCE)),
    )

    provider.load_all()
    cache = getattr(provider, "cache", None)
    if cache is None:
        raise RuntimeError("InstrumentProvider cache 未提供缓存")
    for instrument in cache.instruments():
        yield str(instrument.id)


def _list_by_rest(account_type: BinanceAccountType) -> Iterable[str]:
    if account_type in (BinanceAccountType.USDT_FUTURES, BinanceAccountType.COIN_FUTURES):
        url = USDT_FUTURES_EXCHANGE_INFO if account_type is BinanceAccountType.USDT_FUTURES else COIN_FUTURES_EXCHANGE_INFO
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        for entry in data.get("symbols", []):
            if entry.get("contractType") != "PERPETUAL":
                continue
            symbol = entry.get("symbol")
            if not symbol:
                continue
            yield f"{symbol}-PERP.BINANCE"
    else:
        resp = requests.get(SPOT_EXCHANGE_INFO, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        for entry in data.get("symbols", []):
            symbol = entry.get("symbol")
            if not symbol:
                continue
            yield f"{symbol}.BINANCE"


def main() -> None:
    parser = argparse.ArgumentParser(description="列出 Binance 合约 Instrument IDs")
    parser.add_argument(
        "--account-type",
        default=None,
        help="账户类型（默认读取 config.toml，支持 usdt_futures、coin_futures、spot 等）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制输出数量，默认全部打印",
    )
    args = parser.parse_args()

    settings = load_settings()
    account_type = _resolve_account_type(args.account_type or settings.binance.account_type)

    print(f"Listing instruments for account type: {account_type.name}\n")
    for index, instrument_id in enumerate(_iter_instruments(settings, account_type), start=1):
        print(f"{index:4d}: {instrument_id}")
        if args.limit is not None and index >= args.limit:
            break


if __name__ == "__main__":  # pragma: no cover
    main()
