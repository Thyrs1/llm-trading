# ai_processor.py

import time
import json
import traceback
from typing import Tuple, Dict, List, Any, Optional
from openai import OpenAI
import pandas as pd
import pandas_ta as ta
import feedparser
import re
import os
from datetime import datetime, timezone

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

import config

AI_CLIENT: Optional[OpenAI] = None
sentiment_analyzer = None

def init_ai_client() -> Optional[OpenAI]:
    """Initializes the SYNCHRONOUS DeepSeek/OpenAI client."""
    global AI_CLIENT
    try:
        AI_CLIENT = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)
        AI_CLIENT.models.list()
        print(f"✅ Sync AI client initialized (Model: {config.AI_MODEL_NAME}).")
        return AI_CLIENT
    except Exception as e:
        print(f"❌ CRITICAL: AI Client setup failed: {e}")
        return None

def init_finbert_analyzer():
    """Initializes a FinBERT instance in the main thread."""
    global sentiment_analyzer
    if sentiment_analyzer is not None: return
    print("🤖 Initializing FinBERT instance in main thread...")
    try:
        # Temporarily disable this function to prevent segfault for debugging
        # print("⚠️ FinBERT sentiment analysis is DISABLED for debugging.")
        # pass
        device = 0 if torch.cuda.is_available() else -1
        if device == 0: print(f"✅ CUDA available. Loading FinBERT on GPU.")
        else: print("⚠️ CUDA not found. Loading FinBERT on CPU.")
        
        local_model_path = os.path.abspath("./local_finbert")
        
        if os.path.isdir(local_model_path):
            print(f"✅ Found local FinBERT model at: {local_model_path}")
            tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
        else:
            print(f"⚠️ Local FinBERT model not found. Downloading from Hub (ProsusAI/finbert)...")
            tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

        sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)
        print("✅ FinBERT instance ready.")
    except Exception as e:
        print(f"❌ Failed to initialize FinBERT: {e}")
        sentiment_analyzer = None

def get_sentiment_score_sync(text: str) -> float:
    """Synchronously performs sentiment analysis."""
    global sentiment_analyzer
    if sentiment_analyzer is None:
        # init_finbert_analyzer() # Let the main loop handle initialization
        return 0.0 # Return neutral if not initialized
    
    if not text or len(text) < 10:
        return 0.0
    try:
        results = sentiment_analyzer(text, max_length=512, truncation=True)
        score = 0.0
        for res in results:
            label = res['label'].lower()
            if label == 'positive': score += res['score']
            elif label == 'negative': score -= res['score']
        return round(score / len(results), 2) if results else 0.0
    except Exception as e:
        print(f"❌ Error during sentiment analysis: {e}")
        return 0.0

def get_news_from_rss(symbol: str, limit=10) -> str:
    rss_feeds = ["https://cointelegraph.com/rss", "https://www.coindesk.com/arc/outboundfeeds/rss/"]
    relevant_headlines = []
    base_symbol = symbol.split('/')[0]
    coin_names = {'SOL': ['solana'], 'BTC': ['bitcoin'], 'ETH': ['ethereum'], 'BNB': ['binance coin'], 'XRP': ['ripple']}
    search_terms = [base_symbol.lower()] + coin_names.get(base_symbol.upper(), [])
    pattern = re.compile(r'\b(' + '|'.join(search_terms) + r')\b', re.IGNORECASE)
    for url in rss_feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                if pattern.search(entry.title) or (hasattr(entry, 'summary') and pattern.search(entry.summary)):
                    relevant_headlines.append(entry.title)
        except Exception: pass
    if not relevant_headlines: return f"No specific news found for {symbol}."
    return ". ".join(list(dict.fromkeys(relevant_headlines))[:limit])

def parse_decision_block(raw_text: str) -> Dict:
    decision = {}
    try:
        decision_str = raw_text.split('[DECISION_BLOCK]')[1].split('[END_BLOCK]')[0].strip()
    except IndexError:
        print("⚠️ [DECISION_BLOCK] not found in AI response.")
        return {}
    lines = decision_str.splitlines()
    json_buffer = ""
    in_json_block = False
    type_map = {"ENTRY_PRICE": float, "STOP_LOSS": float, "TAKE_PROFIT": float, "LEVERAGE": int, "RISK_PERCENT": float, "TRAILING_DISTANCE_PCT": float, "NEW_STOP_LOSS": float, "NEW_TAKE_PROFIT": float, "TRIGGER_TIMEOUT": int}
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.upper().startswith("TRIGGERS:"):
            in_json_block = True
            try: json_buffer += stripped_line.split(':', 1)[1].strip()
            except IndexError: pass
            continue
        if in_json_block:
            json_buffer += stripped_line
            if stripped_line.endswith("]"):
                try:
                    clean_json = json_buffer.strip().strip(',')
                    decision['triggers'] = json.loads(clean_json)
                except json.JSONDecodeError as e:
                    print(f"❌ Failed to parse TRIGGERS JSON block: {e}\nBlock content was: {json_buffer}")
                in_json_block = False
                json_buffer = ""
        else:
            if ':' in line:
                key, value = line.split(':', 1)
                key, value = key.strip().upper(), value.strip()
                key_lower = key.lower()
                if key_lower == 'triggers': continue
                if key in type_map:
                    try: decision[key_lower] = type_map[key](value)
                    except (ValueError, TypeError): decision[key_lower] = None
                else:
                    decision[key_lower] = value
    return decision

def parse_chain_of_thought_block(raw_text: str) -> str:
    """Extracts the content of the [CHAIN_OF_THOUGHT_BLOCK]."""
    try:
        thought_process = raw_text.split('[CHAIN_OF_THOUGHT_BLOCK]')[1].split('[END_CHAIN_OF_THOUGHT_BLOCK]')[0].strip()
        return thought_process
    except IndexError:
        return "No Chain of Thought block found in AI response."

def parse_context_block(raw_text: str) -> Dict:
    context = {}
    context_str = ""
    try:
        context_str = raw_text.split('[MARKET_CONTEXT_BLOCK]')[1].split('[END_CONTEXT_BLOCK]')[0].strip()
    except IndexError:
        return {}
    list_keys = ["KEY_SUPPORT_LEVELS", "KEY_RESISTANCE_LEVELS"]
    for line in context_str.splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            key, value = key.strip().upper(), value.strip()
            key_lower = key.lower()
            if key in list_keys:
                try:
                    levels = [float(level.strip()) for level in value.split(',') if level.strip()]
                    context[key_lower] = levels
                except (ValueError, TypeError):
                    context[key_lower] = []
            else:
                context[key_lower] = value
    if context: context['last_full_analysis_timestamp'] = datetime.now(timezone.utc).isoformat()
    return context

def process_klines(klines: List[List]) -> pd.DataFrame:
    df = pd.DataFrame(klines, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    df = df.set_index('date')
    df = df[~df.index.duplicated(keep='last')] 
    return df[['open', 'high', 'low', 'close', 'volume']].astype(float)

def analyze_freqtrade_data(df_5m: pd.DataFrame, current_price: float) -> str:
    """
    Performs a hybrid technical analysis, providing both the latest indicator value,
    its change from the previous candle (Delta), and a short-term history list.
    """
    if len(df_5m) < 60: return "Insufficient data for meaningful analysis."
    df_5m_clean = df_5m[~df_5m.index.duplicated(keep='last')]
    if len(df_5m_clean) < 60: return "Insufficient unique data for meaningful analysis."
    analysis_report = f"### 0. Current Market Price (Anchor)\n- **Current Price:** {current_price:.4f} USDT\n\n"
    analysis_report += "### 1. Multi-Timeframe Hybrid Technical Analysis\n"
    timeframe_settings = {'1d': '1d', '4h': '4h', '1h': '1h', '15m': '15Min'}
    for tf_name, rule in timeframe_settings.items():
        try:
            df_tf = df_5m_clean.resample(rule).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
            if len(df_tf) < 50:
                analysis_report += f"--- Report ({tf_name}) ---\nInsufficient data for {tf_name} indicators.\n"
                continue
            df_tf.ta.ema(length=20, append=True)
            df_tf.ta.ema(length=50, append=True)
            df_tf.ta.rsi(length=14, append=True)
            df_tf.ta.adx(length=14, append=True)
            df_tf.ta.macd(append=True)
            df_tf.ta.bbands(length=20, append=True)
            df_tf.ta.obv(append=True)
            last_five = df_tf.iloc[-5:]
            if len(last_five) < 2: continue
            latest = last_five.iloc[-1]
            previous = last_five.iloc[-2]
            trend = "BULLISH" if latest.get('EMA_20', 0) > latest.get('EMA_50', 0) else "BEARISH" if latest.get('EMA_20', 0) < latest.get('EMA_50', 0) else "RANGING"
            rsi_delta = latest.get('RSI_14', 0) - previous.get('RSI_14', 0)
            rsi_history = last_five['RSI_14'].round(2).tolist()
            macd_hist_delta = latest.get('MACDh_12_26_9', 0) - previous.get('MACDh_12_26_9', 0)
            macd_hist_history = last_five['MACDh_12_26_9'].round(4).tolist()
            obv_ema = ta.ema(df_tf['OBV'], length=10)
            obv_ema_history = obv_ema.iloc[-5:].round(0).tolist() if not obv_ema.empty else []
            bb_pos = "Above Upper Band" if latest['close'] > latest.get('BBU_20_2.0', 0) else "Below Lower Band" if latest['close'] < latest.get('BBL_20_2.0', 0) else "Between Bands"
            analysis_report += f"--- Report ({tf_name}) ---\n"
            analysis_report += f"Close: {latest['close']:.4f}, Trend: {trend}\n"
            analysis_report += f"RSI: {latest.get('RSI_14', 50):.2f} (Δ: {rsi_delta:+.2f}) | History: {rsi_history}\n"
            analysis_report += f"MACD Hist: {latest.get('MACDh_12_26_9', 0):.4f} (Δ: {macd_hist_delta:+.4f}) | History: {macd_hist_history}\n"
            analysis_report += f"ADX: {latest.get('ADX_14', 0):.2f}, BBands: {bb_pos}\n"
            analysis_report += f"OBV EMA(10) History: {obv_ema_history}\n"
        except Exception as e:
            analysis_report += f"--- Report ({tf_name}) ---\nError during analysis: {e}\n"
    return analysis_report

# ########################################################################### #
# ################## START OF MODIFIED SECTION ############################## #
# ########################################################################### #
def get_market_regime(df_1d: pd.DataFrame) -> str:
    """Determines the overall market regime based on long-term indicators."""
    if len(df_1d) < 200:
        return "UNCLEAR (Insufficient 1D data)"
    try:
        ema_200 = ta.ema(df_1d['close'], length=200).iloc[-1]
        latest_close = df_1d['close'].iloc[-1]
        bbands = ta.bbands(df_1d['close'], length=20)
        bb_width_pct = ((bbands['BBU_20_2.0'] - bbands['BBL_20_2.0']) / bbands['BBM_20_2.0'] * 100).iloc[-1]
        regime = ""
        if latest_close > ema_200:
            regime = "BULLISH TREND"
        else:
            regime = "BEARISH TREND"
        if bb_width_pct < 15:
            regime = "RANGING / CONSOLIDATION"
        return regime
    except Exception as e:
        print(f"⚠️ Error calculating market regime: {e}")
        return "UNCLEAR (Calculation Error)"
# ########################################################################### #
# ################### END OF MODIFIED SECTION ############################### #
# ########################################################################### #

def get_ai_decision_sync(analysis_data: str, position_data: str, context_summary: str, live_equity: float, sentiment_score: float, market_regime: str) -> Tuple[Dict, Dict, str, str]:
    """Synchronously retrieves decision, context, raw response, and thoughts from the AI."""
    global AI_CLIENT
    if not AI_CLIENT: return {}, {}, "", ""
    try:
        with open("trade_memory.txt", "r") as f: lessons = "".join(f.readlines()[-15:])
    except FileNotFoundError: lessons = "No past trade lessons available yet."
    
    prompt_body = f"""**--- OVERALL MARKET REGIME ---**
The current macro market condition is determined to be: **{market_regime}**

**--- CRITICAL ACCOUNT CONSTRAINTS ---**
My current account equity is ${live_equity:.2f} USDT.

**--- NEWS SENTIMENT ---**
Current News Sentiment Score: {sentiment_score:.2f} (-1 Negative, +1 Positive).

**--- STRATEGIC MEMORY: LESSONS FROM PAST TRADES ---**
{lessons}

**--- HISTORY OF YOUR RECENT MARKET ANALYSES ---**
{context_summary}

**--- CURRENT DATA FOR ANALYSIS ---**
1. Current Position Status: {position_data}
2. Multi-Timeframe Hybrid Technical Analysis:\n{analysis_data}"""
    
    try:
        response = AI_CLIENT.chat.completions.create(
            model=config.AI_MODEL_NAME,
            messages=[
                {"role": "system", "content": config.AI_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_body}
            ],
            temperature=0.0,
            timeout=45
        )
        raw_response_text = response.choices[0].message.content
        decision_dict = parse_decision_block(raw_response_text)
        context_dict = parse_context_block(raw_response_text)
        chain_of_thought = parse_chain_of_thought_block(raw_response_text)
        return decision_dict, context_dict, raw_response_text, chain_of_thought
    except Exception:
        print(f"❌ Unexpected Error in AI call: {traceback.format_exc()}")
        return {}, {}, "", ""

def summarize_and_learn_sync(trade_summary: str, symbol: str):
    global AI_CLIENT
    if not AI_CLIENT: return
    memory_prompt = f"You are a master trading analyst. Analyze this trade summary for {symbol}: {trade_summary}. Condense your analysis into a single, powerful, one-sentence \"Lesson:\" learned. Provide only the single \"Lesson:\" line."
    try:
        response = AI_CLIENT.chat.completions.create(model=config.AI_MODEL_NAME, messages=[{"role": "user", "content": memory_prompt}], temperature=0.0)
        lesson = response.choices[0].message.content.strip()
        if "Lesson:" in lesson:
            print(f"💡 New lesson learned: {lesson}", symbol)
            with open("trade_memory.txt", "a") as f: f.write(f"- [{symbol}] {lesson}\n")
    except Exception as e:
        print(f"❌ Error during trade summarization: {e}", symbol)