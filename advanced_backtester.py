# live_ai_backtester.py

import backtrader as bt
import pandas as pd
from datetime import datetime
import json
import time
import traceback

# Import all the REAL methods from your project files
from ai_processor import (
    init_ai_client,
    analyze_freqtrade_data,
    get_ai_decision_sync,
    summarize_and_learn_sync
)
from database_manager import setup_database, log_trade, log_system_message
import config

# --- Custom Data Feed for Binance CSV Format ---
class BinanceData(bt.feeds.PandasData):
    """
    Custom data feed to parse the detailed Binance CSV format.
    """
    lines = ('open_time', 'close_time', 'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore',)
    
    params = (
        ('datetime', None), ('open', 'open'), ('high', 'high'), ('low', 'low'),
        ('close', 'close'), ('volume', 'volume'), ('openinterest', None),
        ('open_time', 'open_time'), ('close_time', 'close_time'), ('quote_volume', 'quote_volume'),
        ('count', 'count'), ('taker_buy_volume', 'taker_buy_volume'),
        ('taker_buy_quote_volume', 'taker_buy_quote_volume'), ('ignore', 'ignore'),
    )

# --- The Live AI Backtesting Strategy ---
class LiveAIStrategy(bt.Strategy):
    """
    This strategy makes a LIVE API call to the AI on each candle and correctly
    manages the trade lifecycle based on the AI's response.
    """
    def __init__(self):
        self.d5m = self.datas[0]
        self.order = None
        # State variables that mimic the live bot
        self.trade_entry_price = 0
        self.trade_entry_reason = ""
        self.trade_side = "" # "LONG" or "SHORT"
        self.active_sl_tp = {} 

    def log(self, txt, dt=None):
        dt = dt or self.d5m.datetime.datetime(0)
        log_system_message(txt, f"BACKTEST-{self.d5m._name}")

    def notify_trade(self, trade):
        if trade.isclosed:
            self.log(f'TRADE CLOSED, PNL: {trade.pnlcomm:.2f}')
            
            # CRITICAL FIX: Determine side from the trade object itself, which is robust.
            side = "LONG" if trade.islong else "SHORT"
            
            pnl_pct = (trade.pnlcomm / (self.trade_entry_price * abs(trade.size))) if self.trade_entry_price * abs(trade.size) != 0 else 0.0
            
            log_trade(
                symbol=self.d5m._name, side=side, entry_price=self.trade_entry_price,
                exit_price=trade.price, quantity=abs(trade.size), pnl=trade.pnlcomm,
                pnl_pct=pnl_pct, reasoning=self.trade_entry_reason
            )

            outcome = "WIN" if trade.pnlcomm > 0 else "LOSS"
            trade_summary = f"Outcome: {outcome}, PNL: {trade.pnlcomm:.2f} USDT. Entry Reason: {self.trade_entry_reason}"
            summarize_and_learn_sync(trade_summary, self.d5m._name)
            
            # Reset all trade-specific state
            self.trade_entry_reason = ""
            self.trade_entry_price = 0
            self.active_sl_tp = {}
            self.trade_side = ""

    def notify_order(self, order):
        if order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')
        self.order = None

    def next(self):
        if self.order:
            return

        bars_to_process = min(len(self.d5m), 200)
        if bars_to_process < 60: return
            
        df_5m = pd.DataFrame({
            'date': [bt.num2date(x) for x in self.d5m.datetime.get(ago=-1, size=bars_to_process)],
            'open': self.d5m.open.get(ago=-1, size=bars_to_process),
            'high': self.d5m.high.get(ago=-1, size=bars_to_process),
            'low': self.d5m.low.get(ago=-1, size=bars_to_process),
            'close': self.d5m.close.get(ago=-1, size=bars_to_process),
            'volume': self.d5m.volume.get(ago=-1, size=bars_to_process),
        })
        df_5m.set_index('date', inplace=True)
        
        current_price = self.d5m.close[0]
        analysis_bundle = analyze_freqtrade_data(df_5m, current_price)
        pos_report = "Side: FLAT" if not self.position else f"Side: {self.trade_side}, Entry: {self.trade_entry_price:.4f}"
        
        self.log("Requesting live AI decision...")
        decision, _, _ = get_ai_decision_sync(
            analysis_data=analysis_bundle, position_data=pos_report, context_summary="{}",
            live_equity=self.broker.getvalue(), sentiment_score=0.0
        )
        
        if not decision:
            self.log("AI returned no decision.")
            return

        action = decision.get('action')
        
        # --- LOGIC FOR WHEN NOT IN A POSITION ---
        if not self.position:
            if action == 'OPEN_POSITION' and decision.get('confidence') == 'high':
                side = decision.get('decision')
                sl = decision.get('stop_loss')
                tp = decision.get('take_profit')
                risk_percent_from_ai = decision.get('risk_percent', 1.0)

                if not all([side, sl, tp]):
                    self.log("AI decision to OPEN is missing critical parameters (side, sl, or tp).")
                    return

                cash = self.broker.getvalue()
                ai_risk_fraction = risk_percent_from_ai / 100.0
                risk_to_use = min(ai_risk_fraction, config.MAX_RISK_PER_TRADE)
                risk_amount = cash * risk_to_use
                
                size = risk_amount / abs(current_price - sl) if (current_price - sl) != 0 else 0
                if size <= 0: return
                
                # Set state for the new trade
                self.trade_entry_reason = decision.get('reasoning', 'N/A')
                self.trade_entry_price = current_price
                self.active_sl_tp = {'sl': sl, 'tp': tp}
                self.trade_side = side

                if side == 'LONG':
                    self.log(f"AI DECISION: GO LONG, Size: {size:.4f}, Risk: {risk_to_use*100:.2f}%")
                    self.order = self.buy(size=size)
                elif side == 'SHORT':
                    self.log(f"AI DECISION: GO SHORT, Size: {size:.4f}, Risk: {risk_to_use*100:.2f}%")
                    self.order = self.sell(size=size)
            else:
                self.log(f"AI DECISION: {action}. Reason: {decision.get('reasoning', 'N/A')}")
        
        # --- LOGIC FOR WHEN IN A POSITION ---
        elif self.position:
            # CRITICAL FIX: Handle in-trade decisions from the AI
            if action == 'CLOSE_POSITION':
                self.log(f"AI DECISION: CLOSE POSITION at {current_price:.2f}. Reason: {decision.get('reasoning')}")
                self.close() # Close the position
                return

            # Check for manual SL/TP exit
            sl = self.active_sl_tp.get('sl')
            tp = self.active_sl_tp.get('tp')

            if self.position.size > 0: # In a Long position
                if tp and current_price >= tp:
                    self.log(f"TAKE PROFIT (LONG) HIT at {current_price:.2f}")
                    self.close()
                elif sl and current_price <= sl:
                    self.log(f"STOP LOSS (LONG) HIT at {current_price:.2f}")
                    self.close()
            
            elif self.position.size < 0: # In a Short position
                if tp and current_price <= tp:
                    self.log(f"TAKE PROFIT (SHORT) HIT at {current_price:.2f}")
                    self.close()
                elif sl and current_price >= sl:
                    self.log(f"STOP LOSS (SHORT) HIT at {current_price:.2f}")
                    self.close()
            else:
                 self.log(f"AI DECISION (IN TRADE): HOLD. Reason: {decision.get('reasoning', 'N/A')}")


if __name__ == '__main__':
    cerebro = bt.Cerebro()
    
    print("Initializing AI Client and Database for backtest...")
    init_ai_client()
    setup_database()

    data_path = 'SOLUSDT-5m-data.csv'
    
    try:
        dataframe = pd.read_csv(
            data_path,
            header=0,
            names=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore']
        )
        dataframe['datetime'] = pd.to_datetime(dataframe['open_time'], unit='ms')
        dataframe.set_index('datetime', inplace=True)
    except FileNotFoundError:
        print(f"ERROR: Data file '{data_path}' not found.")
        exit()
    except Exception as e:
        print(f"Error loading data: {e}")
        print(traceback.format_exc())
        exit()

    data_5m = BinanceData(dataname=dataframe, name="SOLUSDT")
    cerebro.adddata(data_5m)

    cerebro.addstrategy(LiveAIStrategy)
    cerebro.broker.setcash(10000.0)
    cerebro.broker.setcommission(commission=0.0004)

    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trade_analyzer')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    start_time = time.time()
    results = []
    try:
        results = cerebro.run()
    except Exception as e:
        log_system_message(f"CRITICAL BACKTEST FAILURE: {e}", "BACKTEST-SYSTEM")
        print(f"CRITICAL BACKTEST FAILURE: {traceback.format_exc()}")
    finally:
        end_time = time.time()
        print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
        print(f'Backtest Duration: {(end_time - start_time):.2f} seconds')

        if results:
            strat = results[0]
            trade_analysis = strat.analyzers.trade_analyzer.get_analysis()
            sharpe = strat.analyzers.sharpe_ratio.get_analysis().get('sharperatio')
            drawdown = strat.analyzers.drawdown.get_analysis().max.drawdown
            
            report = {
                "sharpe_ratio": sharpe,
                "max_drawdown_pct": drawdown,
                "total_trades": trade_analysis.total.total if trade_analysis else 0,
                "win_rate_pct": (trade_analysis.won.total / trade_analysis.total.total) * 100 if trade_analysis and trade_analysis.total.total > 0 else 0,
                "total_net_pnl": trade_analysis.pnl.net.total if trade_analysis else 0
            }

            print("\n--- Live AI Backtest Performance ---")
            print(json.dumps(report, indent=4))
            
            report_file = 'live_ai_backtest_report.txt'
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=4)
            print(f"💾 Final performance report saved to: {report_file}")

            plot_file = 'live_ai_backtest_plot.png'
            try:
                figure = cerebro.plot(style='candlestick', barup='green', bardown='red')[0][0]
                figure.savefig(plot_file, dpi=300)
                print(f"📈 Backtest plot saved to: {plot_file}")
            except Exception as e:
                print(f"Could not generate plot: {e}")