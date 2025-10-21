# ai_processor.py

import asyncio
from concurrent.futures import ProcessPoolExecutor
import time
import json
import traceback
from typing import Tuple, Dict, List, Any, Optional
from openai import AsyncOpenAI, APIStatusError, APITimeoutError
import pandas as pd
import pandas_ta as ta
import feedparser
import re
import os
from datetime import datetime, timezone

# --- NEW: Import torch and transformers for local model loading ---
# These imports are here to support the FinBERT sentiment analysis
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# --- Local Imports ---
import config

# --- AI & Sentiment Globals ---
# AI_CLIENT is set per-process in the new structure's init.
AI_CLIENT: Optional[AsyncOpenAI] = None
sentiment_analyzer = None 

# --- Initialization Functions ---
async def init_ai_client() -> Optional[AsyncOpenAI]:
    """Initializes the ASYNCHRONOUS DeepSeek/OpenAI client."""
    # This must be called inside the event loop of each process.
    global AI_CLIENT
    try:
        AI_CLIENT = AsyncOpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)
        # Simple health check
        await AI_CLIENT.models.list()
        print(f"✅ Async AI client initialized (Model: {config.AI_MODEL_NAME}).")
        return AI_CLIENT
    except Exception as e:
        print(f"❌ CRITICAL: AI Client setup failed: {e}")
        return None

def init_finbert_process():
    """Initializes a FinBERT instance, prioritizing a local model."""
    global sentiment_analyzer
    # Check if analyzer is already initialized (might be called multiple times by executor)
    if sentiment_analyzer is not None: return
    print("🤖 Initializing FinBERT instance in new process...")
    try:
        # Check for CUDA availability
        device = 0 if torch.cuda.is_available() else -1
        if device == 0: print(f"✅ CUDA available. Loading FinBERT on GPU.")
        else: print("⚠️ CUDA not found. Loading FinBERT on CPU.")
        
        # Look for a local model directory first
        local_model_path = os.path.abspath("./local_finbert")
        
        if os.path.isdir(local_model_path):
            print(f"✅ Found local FinBERT model at: {local_model_path}")
            tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
        else:
            print(f"⚠️ Local FinBERT model not found. Downloading from Hub (ProsusAI/finbert)...")
            tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

        # Create the sentiment pipeline
        sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)
        print("✅ FinBERT instance ready in process.")
    except Exception as e:
        print(f"❌ Failed to initialize FinBERT in process: {e}")
        # Setting to None will disable sentiment for this process/pool
        sentiment_analyzer = None

# --- CPU-Bound Task (to be run in the executor) ---
def analyze_sentiment_blocking(text: str) -> float:
    """
    Synchronous function for sentiment analysis, executed in a separate process 
    via ProcessPoolExecutor.
    """
    global sentiment_analyzer
    if not sentiment_analyzer or not text or len(text) < 10:
        return 0.0
    try:
        results = sentiment_analyzer(text, max_length=512, truncation=True)
        score = 0.0
        # Aggregate scores (positive +score, negative -score)
        for res in results:
            label = res['label'].lower()
            if label == 'positive': score += res['score']
            elif label == 'negative': score -= res['score']
        return round(score / len(results), 2) if results else 0.0
    except Exception as e:
        # Note: Printing here is less reliable from a process pool, but useful for debugging
        print(f"❌ Error during sentiment analysis in subprocess: {e}")
        return 0.0

# --- Async Wrapper for Sentiment Analysis ---
async def get_sentiment_score_async(executor: ProcessPoolExecutor, text: str) -> float:
    """
    The asynchronous wrapper that the main bot will call.
    It runs the blocking analysis in the process pool without freezing the event loop.
    """
    if executor is None:
        # This will happen if init_finbert_process failed in the calling process
        return 0.0
    try:
        loop = asyncio.get_running_loop()
        score = await loop.run_in_executor(executor, analyze_sentiment_blocking, text)
        return score
    except Exception:
        # This handles cases where the executor itself fails or is shut down
        return 0.0

# --- News Fetching ---
def get_news_from_rss(symbol: str, limit=10) -> str:
    """Fetches and filters headlines from multiple free RSS feeds."""
    rss_feeds = [
        "https://cointelegraph.com/rss",
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        # Add more feeds if needed
    ]
    relevant_headlines = []
    # Extract the base symbol (e.g., SOL from SOL/USDT)
    base_symbol = symbol.split('/')[0]
    coin_names = {'SOL': ['solana'], 'BTC': ['bitcoin'], 'ETH': ['ethereum']}
    # Create search terms (e.g., ['sol', 'solana'])
    search_terms = [base_symbol.lower()] + coin_names.get(base_symbol.upper(), [])
    # Regex pattern to match whole words
    pattern = re.compile(r'\b(' + '|'.join(search_terms) + r')\b', re.IGNORECASE)

    for url in rss_feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                # Check both title and summary for relevance
                if pattern.search(entry.title) or (hasattr(entry, 'summary') and pattern.search(entry.summary)):
                    relevant_headlines.append(entry.title)
        except Exception: 
            # Silently fail on RSS feed errors
            pass 
            
    if not relevant_headlines: return f"No specific news found for {symbol}."
    # Use dict.fromkeys for fast unique list generation, then limit
    return ". ".join(list(dict.fromkeys(relevant_headlines))[:limit])

# --- Core Parsing Functions ---
def parse_decision_block(raw_text: str) -> Dict:
    """Parses [DECISION_BLOCK] into a dictionary, handling JSON for triggers."""
    decision = {}
    decision_str = ""
    try:
        # Extract content between [DECISION_BLOCK] and [END_BLOCK]
        decision_str = raw_text.split('[DECISION_BLOCK]')[1].split('[END_BLOCK]')[0].strip()
        
        # Check for the multi-trigger JSON format
        if '"triggers":' in decision_str.lower():
            start_brace = decision_str.find('{')
            end_brace = decision_str.rfind('}') + 1
            if start_brace != -1 and end_brace != -1:
                json_part = decision_str[start_brace:end_brace]
                decision = json.loads(json_part)
                # Convert keys to lowercase for consistency
                return {k.lower(): v for k, v in decision.items()}
    except (IndexError, json.JSONDecodeError) as e:
        # Fallback for plain-text blocks or failed JSON parse
        print(f"⚠️ JSON parsing failed for decision block: {e}. Falling back to line-by-line.")
    
    # Line-by-line fallback parser
    type_map = {
        "ENTRY_PRICE": float, "STOP_LOSS": float, "TAKE_PROFIT": float, "LEVERAGE": int,
        "RISK_PERCENT": float, "TRAILING_ACTIVATION_PRICE": float, "TRAILING_DISTANCE_PCT": float,
        "NEW_STOP_LOSS": float, "NEW_TAKE_PROFIT": float, "TRIGGER_PRICE": float, 
        "TRIGGER_LEVEL": float, "TRIGGER_TIMEOUT": int
    }
    for line in decision_str.splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            key, value = key.strip().upper(), value.strip()
            key_lower = key.lower()
            if key in type_map:
                try: decision[key_lower] = type_map[key](value)
                except (ValueError, TypeError): decision[key_lower] = None # Set to None if type conversion fails
            else:
                decision[key_lower] = value
    return decision

def parse_context_block(raw_text: str) -> Dict:
    """Parses [MARKET_CONTEXT_BLOCK] into a dictionary."""
    context = {}
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
                    # Parse comma-separated list of float levels
                    levels = [float(level.strip()) for level in value.split(',') if level.strip()]
                    context[key_lower] = levels
                except (ValueError, TypeError):
                    context[key_lower] = []
            else:
                context[key_lower] = value
    
    # Add a timestamp for the context freshness
    if context: context['last_full_analysis_timestamp'] = datetime.now(timezone.utc).isoformat()
    return context

# --- Freqtrade-Style Data Analysis ---
def process_klines(klines: List[List]) -> pd.DataFrame:
    """Converts CCXT klines into a standard Pandas DataFrame."""
    df = pd.DataFrame(klines, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    df = df.set_index('date')
    return df[['open', 'high', 'low', 'close', 'volume']].astype(float)

def analyze_freqtrade_data(df_5m: pd.DataFrame, current_price: float) -> str:
    """Generates a rich multi-timeframe analysis bundle with advanced indicators."""
    if len(df_5m) < 60: return "Insufficient data for meaningful analysis."
    
    analysis_report = f"### 0. Current Market Price (Anchor)\n- **Current Price:** {current_price:.4f} USDT\n\n"
    analysis_report += "### 1. Freqtrade Multi-Timeframe Analysis\n"
    
    # Define timeframes for resampling
    timeframe_settings = {'1d': '1d', '4h': '4h', '1h': '1h', '15m': '15Min', '5m': '5Min'}
    
    for tf_name, rule in timeframe_settings.items():
        if len(df_5m) < 20 and tf_name != '5m':
            analysis_report += f"--- Report ({tf_name}) ---\nInsufficient data for {tf_name} resampling.\n"
            continue
            
        # Resample data for higher timeframes
        df_tf = df_5m.copy() if tf_name == '5m' else df_5m.resample(rule).agg(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
        ).dropna()
        
        if len(df_tf) < 50:
            analysis_report += f"--- Report ({tf_name}) ---\nInsufficient data for {tf_name} indicators.\n"
            continue
            
        # Calculate Technical Indicators
        df_tf.ta.ema(length=20, append=True)
        df_tf.ta.ema(length=50, append=True)
        df_tf.ta.rsi(length=14, append=True)
        df_tf.ta.adx(length=14, append=True)
        df_tf.ta.macd(close=df_tf['close'], append=True)
        
        # Extract latest values
        latest = df_tf.iloc[-1]
        
        # Determine simple trend based on EMAs
        ema_20 = latest.get('EMA_20', 0)
        ema_50 = latest.get('EMA_50', 0)
        trend = "BULLISH" if ema_20 > ema_50 else "BEARISH" if ema_20 < ema_50 else "RANGING"
        
        analysis_report += f"--- Report ({tf_name}) ---\n"
        analysis_report += f"Close: {latest['close']:.4f}, Trend: {trend}\n"
        analysis_report += f"RSI: {latest.get('RSI_14', 50):.2f}, ADX: {latest.get('ADX_14', 0):.2f}, MACD_Hist: {latest.get('MACDh_12_26_9', 0):.4f}\n"
        
    return analysis_report

# --- Main Decision Function ---
async def get_ai_decision(analysis_data: str, position_data: str, context_summary: str, live_equity: float, sentiment_score: float) -> Tuple[Dict, Dict]:
    """Asynchronously retrieves decision and context from the AI."""
    global AI_CLIENT
    if not AI_CLIENT: return {}, {}
    
    # Load strategic memory (trade lessons)
    try:
        # Read the last 15 lessons from the file
        with open("trade_memory.txt", "r") as f: lessons = "".join(f.readlines()[-15:])
    except FileNotFoundError: lessons = "No past trade lessons available yet."
    
    system_message = config.AI_SYSTEM_PROMPT
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
        # Call the DeepSeek/OpenAI API
        response = await AI_CLIENT.chat.completions.create(
            model=config.AI_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt_body}
            ],
            temperature=0.0, # Prefer deterministic, reliable output
            timeout=45
        )
        raw_response_text = response.choices[0].message.content
        decision_dict = parse_decision_block(raw_response_text)
        context_dict = parse_context_block(raw_response_text)
        return decision_dict, context_dict
    except Exception as e:
        print(f"❌ Unexpected Error in AI call: {traceback.format_exc()}")
        return {}, {}

# --- Trade Summarization & Learning ---
async def summarize_and_learn(trade_summary: str, symbol: str):
    """Asynchronously sends trade summary to AI for learning and appends the lesson to memory."""
    global AI_CLIENT
    if not AI_CLIENT: return
    
    memory_prompt = f"You are a master trading analyst. Analyze this trade summary for {symbol}: {trade_summary}. Condense your analysis into a single, powerful, one-sentence \"Lesson:\" learned. Provide only the single \"Lesson:\" line."
    
    try:
        response = await AI_CLIENT.chat.completions.create(
            model=config.AI_MODEL_NAME, 
            messages=[{"role": "user", "content": memory_prompt}], 
            temperature=0.0
        )
        lesson = response.choices[0].message.content.strip()
        # Ensure the output is in the expected format before saving
        if "Lesson:" in lesson:
            print(f"💡 New lesson learned: {lesson}", symbol)
            with open("trade_memory.txt", "a") as f: f.write(f"- [{symbol}] {lesson}\n")
    except Exception as e:
        print(f"❌ Error during trade summarization: {e}", symbol)