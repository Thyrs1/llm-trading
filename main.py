from __future__ import annotations

import argparse

import asyncio

from nautilus_bot.runtime import load_bot_settings, run_backtest, run_live


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nautilus Trader 事件驱动 LLM 交易机器人")
    parser.add_argument(
        "--mode",
        choices=["live", "backtest"],
        default="live",
        help="运行模式：live 启动实时节点，backtest 运行官方回测框架。",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径（可选，支持 TOML / JSON）。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_bot_settings(args.config)
    if args.mode == "live":
        asyncio.run(run_live(settings))
    else:
        run_backtest(settings)


if __name__ == "__main__":
    main()
