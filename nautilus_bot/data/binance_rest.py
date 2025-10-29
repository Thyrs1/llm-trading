"""轻量级 Binance REST 客户端，仅用于启动阶段历史 K 线预热。"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Sequence

import logging

import requests

LOGGER = logging.getLogger(__name__)

_DEFAULT_FUTURES_HTTP = "https://fapi.binance.com"


class BinanceRESTError(RuntimeError):
    """封装 REST 请求异常。"""


def fetch_recent_klines(
    symbol: str,
    interval: str,
    limit: int,
    base_http_url: str | None = None,
) -> List[dict]:
    """拉取最新历史 K 线，用于策略启动前补齐上下文。

    返回列表中每条记录包含 `date/open/high/low/close/volume`，时间戳取 Binance 的收盘毫秒。
    """

    if limit <= 0:
        return []

    url = f"{(base_http_url or _DEFAULT_FUTURES_HTTP).rstrip('/')}/fapi/v1/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": min(max(limit, 1), 1500),
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        payload: Sequence[Sequence[str]] = resp.json()
    except Exception as exc:  # noqa: BLE001
        raise BinanceRESTError(f"获取 {symbol} K 线失败: {exc}") from exc

    rows: List[dict] = []
    for entry in payload:
        if len(entry) < 7:
            continue
        try:
            close_time_ms = int(entry[6])
            close_dt = datetime.fromtimestamp(close_time_ms / 1000.0, tz=timezone.utc)
            rows.append(
                {
                    "date": close_dt,
                    "open": float(entry[1]),
                    "high": float(entry[2]),
                    "low": float(entry[3]),
                    "close": float(entry[4]),
                    "volume": float(entry[5]),
                }
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("跳过无法解析的 K 线: %s (%s)", entry, exc)
    return rows
