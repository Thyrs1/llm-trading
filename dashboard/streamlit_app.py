from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent.parent
SNAPSHOT_PATH = BASE_DIR / "dashboard_snapshot.json"


def load_snapshot(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except json.JSONDecodeError:
        return None


def render_risk_section(risk: Dict[str, Any]) -> None:
    st.subheader("风险监控")
    cols = st.columns(3)
    cols[0].metric("日初权益", f"{risk.get('daily_start_equity', 0):,.2f}")
    cols[1].metric("日内回撤(%)", f"{risk.get('daily_drawdown_pct', 0):.2f}")
    halted = "是" if risk.get("is_trading_halted") else "否"
    cols[2].metric("是否暂停交易", halted)

    detail_cols = st.columns(3)
    detail_cols[0].metric("连续亏损次数", risk.get("consecutive_losses", 0))
    trading_halted_until = risk.get("trading_halted_until", 0.0)
    if trading_halted_until:
        halted_until = datetime.fromtimestamp(trading_halted_until)
        halted_text = halted_until.strftime("%Y-%m-%d %H:%M")
    else:
        halted_text = "-"
    detail_cols[1].metric("暂停截止", halted_text)
    detail_cols[2].metric("风控处理日序号", risk.get("last_check_day", "-"))


def render_states_section(states: List[Dict[str, Any]]) -> None:
    st.subheader("策略状态")
    if not states:
        st.info("暂无策略状态记录")
        return

    summary_rows = []
    for state in states:
        summary_rows.append(
            {
                "合约": state.get("symbol", "-"),
                "持仓方向": state.get("side", "-"),
                "持仓数量": state.get("quantity", 0.0),
                "持仓均价": state.get("entry_price", 0.0),
                "最新价格": state.get("last_known_price", 0.0),
                "未实现盈亏": state.get("unrealized_pnl", 0.0),
                "是否持仓": bool(state.get("is_in_position")),
                "最后分析时间": state.get("last_analysis_time", "-"),
                "最新情绪分数": state.get("last_sentiment_score"),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.set_index("合约")
    st.dataframe(summary_df, width="stretch")

    for state in states:
        symbol = state.get("symbol", "未知合约")
        with st.expander(f"{symbol} 详情", expanded=False):
            st.write("**市场上下文**")
            market_context = state.get("market_context") or {}
            if market_context:
                context_rows = []
                for key, value in market_context.items():
                    if isinstance(value, (dict, list)):
                        display_value = json.dumps(value, ensure_ascii=False)
                    else:
                        display_value = str(value)
                    context_rows.append({"字段": key, "内容": display_value})
                context_df = pd.DataFrame(context_rows)
                st.dataframe(context_df, width="stretch")
            else:
                st.write("暂无市场上下文")

            st.write("**激活触发器**")
            active_triggers = state.get("active_triggers") or []
            if active_triggers:
                st.json(active_triggers)
            else:
                st.write("暂无触发器")

            st.write("**最近 AI 响应**")
            last_ai_response = state.get("last_ai_response") or "暂无记录"
            st.code(last_ai_response, language="markdown")


def main() -> None:
    st.set_page_config(page_title="LLM Trading Dashboard", layout="wide")
    st.title("LLM Trading 仪表盘")

    auto_refresh_enabled = st.sidebar.checkbox("启用自动刷新", value=True)
    refresh_interval = st.sidebar.slider("刷新间隔(秒)", min_value=5, max_value=120, value=15, step=5)

    snapshot = load_snapshot(SNAPSHOT_PATH)
    if snapshot is None:
        st.warning("未找到有效的 dashboard_snapshot.json，请先运行策略或确认快照路径。")
        st.stop()

    generated_at = snapshot.get("generated_at")
    st.caption(f"最近更新时间：{generated_at or '未知'}")

    risk = snapshot.get("risk") or {}
    render_risk_section(risk)

    states = snapshot.get("states") or []
    render_states_section(states)

    st.sidebar.markdown("#### 使用提示")
    st.sidebar.markdown(
        """
        - 启动策略或回测后将自动更新快照\n
        - 可通过左侧开关控制自动刷新行为\n
        - 若数据异常，可点击右上角“Rerun”或关闭自动刷新再手动刷新
        """
    )

    if auto_refresh_enabled and refresh_interval > 0:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
