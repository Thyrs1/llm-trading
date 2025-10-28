"""
兼容入口：从 v2 起推荐使用 `python main.py --mode live` 启动。
该脚本保留是为了兼容历史调用方式。
"""

from nautilus_bot.orchestrator import build_orchestrator


def main() -> None:
    orchestrator = build_orchestrator()
    orchestrator.run_live()


if __name__ == "__main__":
    main()
