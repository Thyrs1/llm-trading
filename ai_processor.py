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
        init_finbert_analyzer()
    
    if not sentiment_analyzer or not text or len(text) < 10:
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
    coin_names = {'SOL': ['solana'], 'BTC': ['bitcoin'], 'ETH': ['ethereum']}
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
    """
    Parses [DECISION_BLOCK] using a hybrid approach. It handles simple KEY: VALUE lines
    and also detects and parses a multi-line JSON value for the 'TRIGGERS' key.
    """
    decision = {}
    try:
        decision_str = raw_text.split('[DECISION_BLOCK]')[1].split('[END_BLOCK]')[0].strip()
    except IndexError:
        print("⚠️ [DECISION_BLOCK] not found in AI response.")
        return {}

    lines = decision_str.splitlines()
    json_buffer = ""
    in_json_block = False

    type_map = {
        "ENTRY_PRICE": float, "STOP_LOSS": float, "TAKE_PROFIT": float, "LEVERAGE": int,
        "RISK_PERCENT": float, "TRAILING_ACTIVATION_PRICE": float, "TRAILING_DISTANCE_PCT": float,
        "NEW_STOP_LOSS": float, "NEW_TAKE_PROFIT": float, "TRIGGER_PRICE": float, 
        "TRIGGER_LEVEL": float, "TRIGGER_TIMEOUT": int
    }

    for line in lines:
        stripped_line = line.strip()
        
        # State machine to handle the TRIGGERS JSON block
        if stripped_line.upper().startswith("TRIGGERS:"):
            in_json_block = True
            # Start capturing the JSON from the opening bracket
            try:
                json_buffer += stripped_line.split(':', 1)[1].strip()
            except IndexError:
                pass # Handles case where '[' is on the next line
            continue

        if in_json_block:
            json_buffer += stripped_line
            # If we find the closing bracket, the block is complete
            if stripped_line.endswith("]"):
                try:
                    # Clean up the buffer and parse as JSON
                    clean_json = json_buffer.strip().strip(',')
                    decision['triggers'] = json.loads(clean_json)
                except json.JSONDecodeError as e:
                    print(f"❌ Failed to parse TRIGGERS JSON block: {e}\nBlock content was: {json_buffer}")
                # Reset state machine
                in_json_block = False
                json_buffer = ""
        
        # If not in a JSON block, parse as a simple KEY: VALUE pair
        else:
            if ':' in line:
                key, value = line.split(':', 1)
                key, value = key.strip().upper(), value.strip()
                key_lower = key.lower()

                # Skip TRIGGERS key if it's handled separately
                if key_lower == 'triggers':
                    continue

                if key in type_map:
                    try:
                        decision[key_lower] = type_map[key](value)
                    except (ValueError, TypeError):
                        decision[key_lower] = None
                else:
                    decision[key_lower] = value
                    
    return decision

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
    if len(df_5m) < 60: return "Insufficient data for meaningful analysis."
    df_5m_clean = df_5m[~df_5m.index.duplicated(keep='last')]
    if len(df_5m_clean) < 60: return "Insufficient unique data for meaningful analysis."
    analysis_report = f"### 0. Current Market Price (Anchor)\n- **Current Price:** {current_price:.4f} USDT\n\n"
    analysis_report += "### 1. Freqtrade Multi-Timeframe Analysis\n"
    timeframe_settings = {'1d': '1d', '4h': '4h', '1h': '1h', '15m': '15Min', '5m': '5Min'}
    for tf_name, rule in timeframe_settings.items():
        if len(df_5m_clean) < 20 and tf_name != '5m':
            analysis_report += f"--- Report ({tf_name}) ---\nInsufficient data for {tf_name} resampling.\n"
            continue
        df_tf = df_5m_clean.copy() if tf_name == '5m' else df_5m_clean.resample(rule).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}).dropna()
        if len(df_tf) < 50:
            analysis_report += f"--- Report ({tf_name}) ---\nInsufficient data for {tf_name} indicators.\n"
            continue
        df_tf.ta.ema(length=20, append=True)
        df_tf.ta.ema(length=50, append=True)
        df_tf.ta.rsi(length=14, append=True)
        df_tf.ta.adx(length=14, append=True)
        df_tf.ta.macd(close=df_tf['close'], append=True)
        latest = df_tf.iloc[-1]
        ema_20 = latest.get('EMA_20', 0)
        ema_50 = latest.get('EMA_50', 0)
        trend = "BULLISH" if ema_20 > ema_50 else "BEARISH" if ema_20 < ema_50 else "RANGING"
        analysis_report += f"--- Report ({tf_name}) ---\n"
        analysis_report += f"Close: {latest['close']:.4f}, Trend: {trend}\n"
        analysis_report += f"RSI: {latest.get('RSI_14', 50):.2f}, ADX: {latest.get('ADX_14', 0):.2f}, MACD_Hist: {latest.get('MACDh_12_26_9', 0):.4f}\n"
    return analysis_report

def get_ai_decision_sync(analysis_data: str, position_data: str, context_summary: str, live_equity: float, sentiment_score: float) -> Tuple[Dict, Dict, str]:
    """Synchronously retrieves decision, context, and raw response from the AI."""
    global AI_CLIENT
    if not AI_CLIENT: return {}, {}, ""
    try:
        with open("trade_memory.txt", "r") as f: lessons = "".join(f.readlines()[-15:])
    except FileNotFoundError: lessons = "No past trade lessons available yet."
    
    prompt_body = f"""**--- CRITICAL ACCOUNT CONSTRAINTS ---**
My current account equity is ${live_equity:.2f} USDT. Leverage limit is 50x.
**--- NEWS SENTIMENT ---**
Current News Sentiment Score: {sentiment_score:.2f} (-1 Negative, +1 Positive).
**--- STRATEGIC MEMORY: LESSONS FROM PAST TRADES ---**
{lessons}
**--- YOUR LAST MARKET CONTEXT ---**
{context_summary}
**--- CURRENT DATA FOR ANALYSIS ---**
1. Current Position Status: {position_data}
2. Freqtrade Market Analysis:\n{analysis_data}"""
    
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
        return decision_dict, context_dict, raw_response_text
    except Exception:
        print(f"❌ Unexpected Error in AI call: {traceback.format_exc()}")
        return {}, {}, ""

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