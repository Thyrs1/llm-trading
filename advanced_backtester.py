# advanced_backtester.py

import backtrader as bt
import pandas as pd
from datetime import datetime
import json

# Import the REAL analysis and learning methods from your live bot
from ai_processor import analyze_freqtrade_data, summarize_and_learn_sync, init_ai_client

# Import the SIMULATED AI brain
from simulation_ai_processor import get_simulated_ai_decision

# --- Custom Data Feed for Binance CSV Format ---
class BinanceData(bt.feeds.PandasData):
    lines = ('open_time', 'close_time', 'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore',)
    params = (
        ('datetime', None), ('open', 'open'), ('high', 'high'), ('low', 'low'),
        ('close', 'close'), ('volume', 'volume'), ('openinterest', None),
        ('open_time', 'open_time'), ('close_time', 'close_time'), ('quote_volume', 'quote_volume'),
        ('count', 'count'), ('taker_buy_volume', 'taker_buy_volume'),
        ('taker_buy_quote_volume', 'taker_buy_quote_volume'), ('ignore', 'ignore'),
    )

# --- The Advanced Strategy ---
class AdvancedLLMStrategy(bt.Strategy):
    """
    A more precise translation of the AI's logic for backtesting.
    - Implements both Long and Short entry logic.
    - Simulates the summarize_and_learn function.
    """
    params = (
        ('ema_fast_period', 20),
        ('ema_slow_period', 50),
        ('rsi_period', 14),
        ('rsi_overbought', 70),
        ('rsi_oversold', 30),
        ('adx_period', 14),
        ('adx_threshold', 20),
        ('rr_ratio', 1.5),
        ('sl_pct', 0.02),
    )

    def __init__(self):
        self.ema_fast_5m = bt.indicators.EMA(self.data.close, period=self.p.ema_fast_period)
        self.ema_slow_5m = bt.indicators.EMA(self.data.close, period=self.p.ema_slow_period)
        self.rsi_5m = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.adx_15m = bt.indicators.ADX(self.datas[1], period=self.p.adx_period)
        self.ema_slow_1h = bt.indicators.EMA(self.datas[2].close, period=self.p.ema_slow_period)
        self.order = None
        self.entry_reason = ""

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.datetime(0)
        print(f'{dt.isoformat()} | {txt}')

    def summarize_and_learn(self, trade):
        if trade.pnlcomm == 0:
            return
        outcome = "WIN" if trade.pnlcomm > 0 else "LOSS"
        side = "LONG" if trade.history[0].event.size > 0 else "SHORT"
        lesson = ""
        if outcome == "WIN":
            lesson = f"Lesson: Successful {side} trade confirmed that entering during high volatility (ADX > {self.p.adx_threshold}) with HTF confirmation is profitable."
        else:
            if abs(trade.history[0].event.price - trade.history[1].event.price) / trade.history[0].event.price > self.p.sl_pct * 1.5:
                 lesson = f"Lesson: Loss due to high slippage or gap suggests avoiding entries during extreme volatility spikes."
            else:
                 lesson = f"Lesson: Failed {side} trade suggests momentum signals can reverse quickly. Consider adding a counter-trend filter like RSI divergence."
        log_message = f"- [{self.data._name}] {lesson}\n"
        with open("backtest_lessons.txt", "a") as f:
            f.write(log_message)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status == order.Completed:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.2f}, Size: {order.executed.size:.4f}')
            elif order.issell():
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.2f}, Size: {order.executed.size:.4f}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f'TRADE CLOSED, PNL: {trade.pnlcomm:.2f}, Entry Reason: {self.entry_reason}')
            self.summarize_and_learn(trade)
            self.entry_reason = ""

    def next(self):
        if self.order:
            return
        if not self.position:
            is_momentum_bullish = self.ema_fast_5m[0] > self.ema_slow_5m[0]
            is_volatile = self.adx_15m.ADX[0] > self.p.adx_threshold
            is_not_overbought = self.rsi_5m[0] < self.p.rsi_overbought
            is_htf_bullish = self.datas[2].close[0] > self.ema_slow_1h[0]
            if is_momentum_bullish and is_volatile and is_not_overbought and is_htf_bullish:
                price = self.data.close[0]
                stop_loss = price * (1.0 - self.p.sl_pct)
                size = (self.broker.get_cash() * 0.02) / (price - stop_loss)
                self.entry_reason = "Bullish momentum (5m EMA cross) with HTF (1h) trend confirmation and ADX > 20."
                self.log(f'LONG ENTRY CREATE, Price: {price:.2f}, Size: {size:.4f}')
                self.order = self.buy(size=size, exectype=bt.Order.Market)
                return
            is_momentum_bearish = self.ema_fast_5m[0] < self.ema_slow_5m[0]
            is_not_oversold = self.rsi_5m[0] > self.p.rsi_oversold
            is_htf_bearish = self.datas[2].close[0] < self.ema_slow_1h[0]
            if is_momentum_bearish and is_volatile and is_not_oversold and is_htf_bearish:
                price = self.data.close[0]
                stop_loss = price * (1.0 + self.p.sl_pct)
                size = (self.broker.get_cash() * 0.02) / (stop_loss - price)
                self.entry_reason = "Bearish momentum (5m EMA cross) with HTF (1h) trend confirmation and ADX > 20."
                self.log(f'SHORT ENTRY CREATE, Price: {price:.2f}, Size: {size:.4f}')
                self.order = self.sell(size=size, exectype=bt.Order.Market)
                return
        else:
            if self.position.size > 0:
                if self.data.close[0] >= (self.position.price * (1.0 + self.p.sl_pct * self.p.rr_ratio)):
                    self.log(f'TAKE PROFIT (LONG), Price: {self.data.close[0]:.2f}')
                    self.close()
                elif self.data.close[0] <= (self.position.price * (1.0 - self.p.sl_pct)):
                    self.log(f'STOP LOSS (LONG), Price: {self.data.close[0]:.2f}')
                    self.close()
            elif self.position.size < 0:
                if self.data.close[0] <= (self.position.price * (1.0 - self.p.sl_pct * self.p.rr_ratio)):
                    self.log(f'TAKE PROFIT (SHORT), Price: {self.data.close[0]:.2f}')
                    self.close()
                elif self.data.close[0] >= (self.position.price * (1.0 + self.p.sl_pct)):
                    self.log(f'STOP LOSS (SHORT), Price: {self.data.close[0]:.2f}')
                    self.close()

if __name__ == '__main__':
    cerebro = bt.Cerebro()
    init_ai_client()
    data_path = 'SOLUSDT-5m-data.csv'
    try:
        # ########################################################################### #
        # ################## START OF MODIFIED SECTION ############################## #
        # ########################################################################### #
        dataframe = pd.read_csv(
            data_path,
            names=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'],
            header=None # CRITICAL FIX: Tell pandas there is no header row
        )
        # ########################################################################### #
        # ################### END OF MODIFIED SECTION ############################### #
        # ########################################################################### #
        dataframe['datetime'] = pd.to_datetime(dataframe['open_time'], unit='ms')
        dataframe.set_index('datetime', inplace=True)
    except FileNotFoundError:
        print(f"ERROR: Data file '{data_path}' not found.")
        exit()
    data_5m = BinanceData(dataname=dataframe, name="SOLUSDT")
    cerebro.adddata(data_5m)
    cerebro.resampledata(data_5m, timeframe=bt.TimeFrame.Minutes, compression=15)
    cerebro.resampledata(data_5m, timeframe=bt.TimeFrame.Minutes, compression=60)
    cerebro.addstrategy(AdvancedLLMStrategy)
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.0004)
    trade_log_file = 'advanced_backtest_trade_log.csv'
    cerebro.addwriter(bt.WriterFile, csv=True, out=trade_log_file)
    print(f"✍️  Trade log will be saved to: {trade_log_file}")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio', timeframe=bt.TimeFrame.Days)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    results = cerebro.run()
    strat = results[0]
    final_value = cerebro.broker.getvalue()
    print('Final Portfolio Value: %.2f' % final_value)
    trade_analysis = strat.analyzers.trade_analyzer.get_analysis()
    sharpe_ratio = strat.analyzers.sharpe_ratio.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    returns = strat.analyzers.returns.get_analysis()
    performance_report = {
        "start_portfolio": 10000.0, "final_portfolio": final_value,
        "net_pnl": final_value - 10000.0, "sharpe_ratio": sharpe_ratio.get('sharperatio', 'N/A'),
        "max_drawdown_pct": drawdown.max.drawdown, "total_return_pct": returns.get('rtot', 0) * 100,
        "trade_analysis": {}
    }
    if trade_analysis.total.total > 0:
        performance_report["trade_analysis"] = {
            "total_trades": trade_analysis.total.total, "winning_trades": trade_analysis.won.total,
            "losing_trades": trade_analysis.lost.total,
            "win_rate_pct": (trade_analysis.won.total / trade_analysis.total.total) * 100,
            "total_net_pnl": trade_analysis.pnl.net.total,
            "average_win": trade_analysis.won.pnl.average, "average_loss": trade_analysis.lost.pnl.average,
        }
    report_file = 'advanced_backtest_report.txt'
    with open(report_file, 'w') as f:
        json.dump(performance_report, f, indent=4)
    print(f"💾 Final performance report saved to: {report_file}")
    print("\n--- Strategy Performance ---")
    print(json.dumps(performance_report, indent=4))
    figure = cerebro.plot(style='candlestick', barup='green', bardown='red', volume=False)[0][0]
    plot_file = 'advanced_backtest_plot.png'
    figure.savefig(plot_file, dpi=300)
    print(f"📈 Backtest plot saved to: {plot_file}")