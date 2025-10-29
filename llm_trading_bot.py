"""
兼容入口：从 v2 起推荐使用 `python main.py --mode live` 启动。
该脚本保留是为了兼容历史调用方式。
"""

import asyncio

from nautilus_bot.runtime import load_bot_settings, run_live


def main() -> None:
    settings = load_bot_settings(None)
    asyncio.run(run_live(settings))


if __name__ == "__main__":
    main()
