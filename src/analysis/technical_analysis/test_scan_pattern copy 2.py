"""
5-minute Market Scanner & Pattern-based Signal Summarizer

What it does:
- Scans multiple tickers on a 5-minute timeframe (uses yfinance by default).
- Computes indicators: EMA9, EMA21, MACD, RSI, ATR.
- Detects simple patterns & triggers (EMA crossovers, MACD cross, RSI extremes, engulfing candles).
- Suggests LONG / SHORT with stop-loss and take-profit levels (ATR-based).
- Runs continuously in a loop and prints (and optionally writes) a readable text summary.

Notes / Caveats:
- yfinance is used for equities and ETFs. For faster/professional/symbol-limited data, swap the `fetch_ohlc` function
  to use your broker/API (Alpaca, Interactive Brokers, CCXT for crypto, Binance, etc.).
- This script is educational and *not* financial advice. Always paper-test and adjust parameters to your strategy.

Requirements:
pip install yfinance pandas numpy ta

Usage:
- Edit the `CONFIG` block below (tickers, risk multipliers, polling interval).
- Run: python market_5min_scanner.py
"""

import time
import datetime as dt
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import yfinance as yf

# Optional: use `ta` library for indicators if available. We'll implement a few manually.
try:
    import ta
    TA_AVAILABLE = True
except Exception:
    TA_AVAILABLE = False


# ---------------------------
# Config
# ---------------------------
CONFIG = {
    "tickers": ["AAPL", "MSFT", "TSLA", "NVDA", "SPY"],  # edit list
    "interval": "5m",
    "lookback_period_days": 2,  # how many days of history to fetch (yfinance limit matters)
    "poll_interval_seconds": 60,  # how often to refresh (in seconds)
    # risk management multipliers
    "atr_period": 14,
    "sl_atr_mult": 1.5,
    "tp_atr_mult": 3.0,
    # signal thresholds
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    # print/write options
    "output_file": None,  # e.g. 'signals.txt' or None to disable file write
}


# ---------------------------
# Utilities / Indicators
# ---------------------------

def fetch_ohlc(tickers: List[str], interval: str, period_days: int) -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV data for multiple tickers using yfinance. Returns dict ticker -> dataframe.

    yfinance supports batch download; it returns a multi-indexed DataFrame which we split.
    """
    # Build period string (yfinance accepts days like '2d', '5d', '60d')
    period = f"{max(period_days, 1)}d"
    # yfinance.download will return a multi-index columns when multiple tickers provided
    df = yf.download(tickers, period=period, interval=interval, group_by='ticker', threads=True, auto_adjust=False, prepost=False, progress=False)

    out = {}
    if len(tickers) == 1:
        out[tickers[0]] = df.dropna()
        return out

    # When multiple tickers, yfinance returns columns like ('AAPL', 'Open')
    for t in tickers:
        try:
            sub = df[t].dropna()
            # Ensure columns: Open, High, Low, Close, Volume
            if set(["Open","High","Low","Close","Volume"]).issubset(sub.columns):
                out[t] = sub
        except Exception:
            # fallback: try to extract by column name
            cols = [c for c in df.columns if c[0] == t]
            if cols:
                sub = df[cols]
                sub.columns = [c[1] for c in cols]
                out[t] = sub.dropna()
    return out


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    return atr


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=period, min_periods=1).mean()
    ma_down = down.rolling(window=period, min_periods=1).mean()
    rs = ma_up / (ma_down.replace(0, 1e-9))
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(close: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series]:
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


# Candlestick pattern: bullish/bearish engulfing
def is_bullish_engulfing(prev_open, prev_close, open_, close_) -> bool:
    return (prev_close < prev_open) and (close_ > open_) and (close_ > prev_open) and (open_ < prev_close)


def is_bearish_engulfing(prev_open, prev_close, open_, close_) -> bool:
    return (prev_close > prev_open) and (close_ < open_) and (open_ > prev_close) and (close_ < prev_open)


# ---------------------------
# Signal logic
# ---------------------------

def analyze_dataframe(df: pd.DataFrame, config: dict) -> Dict:
    """Compute indicators and return the latest signal summary for this ticker."""
    res = {}
    if df.shape[0] < max(30, config['atr_period'] + 5):
        res['reason'] = 'insufficient data'
        return res

    close = df['Close']
    high = df['High']
    low = df['Low']
    open_ = df['Open']

    # indicators
    ema9 = ema(close, 9)
    ema21 = ema(close, 21)
    macd_line, macd_signal = compute_macd(close)
    rsi = compute_rsi(close, period=14)
    atr = compute_atr(df, period=config['atr_period'])

    latest_idx = df.index[-1]
    prev_idx = df.index[-2]

    latest = {
        'time': latest_idx,
        'close': float(close.iloc[-1]),
        'open': float(open_.iloc[-1]),
        'high': float(high.iloc[-1]),
        'low': float(low.iloc[-1]),
    }

    # signal rules (simple heuristic combining patterns)
    signals = []

    # EMA crossover
    if ema9.iloc[-2] < ema21.iloc[-2] and ema9.iloc[-1] > ema21.iloc[-1]:
        signals.append(('ema_cross', 'bull'))
    if ema9.iloc[-2] > ema21.iloc[-2] and ema9.iloc[-1] < ema21.iloc[-1]:
        signals.append(('ema_cross', 'bear'))

    # MACD crossover
    if macd_line.iloc[-2] < macd_signal.iloc[-2] and macd_line.iloc[-1] > macd_signal.iloc[-1]:
        signals.append(('macd_cross', 'bull'))
    if macd_line.iloc[-2] > macd_signal.iloc[-2] and macd_line.iloc[-1] < macd_signal.iloc[-1]:
        signals.append(('macd_cross', 'bear'))

    # RSI extremes
    if rsi.iloc[-1] < config['rsi_oversold']:
        signals.append(('rsi', 'oversold'))
    if rsi.iloc[-1] > config['rsi_overbought']:
        signals.append(('rsi', 'overbought'))

    # Engulfing patterns on last two candles
    p_open, p_close = open_.iloc[-2], close.iloc[-2]
    o, c = open_.iloc[-1], close.iloc[-1]

    if is_bullish_engulfing(p_open, p_close, o, c):
        signals.append(('engulfing', 'bull'))
    if is_bearish_engulfing(p_open, p_close, o, c):
        signals.append(('engulfing', 'bear'))

    # Decide final side: simple voting
    score = 0
    for s in signals:
        if s[1] in ('bull', 'oversold'):
            score += 1
        elif s[1] in ('bear', 'overbought'):
            score -= 1

    # Build suggested side
    suggested = 'neutral'
    if score >= 1:
        suggested = 'long'
    elif score <= -1:
        suggested = 'short'

    # Entry is current close
    entry = latest['close']
    latest_atr = float(atr.iloc[-1]) if not np.isnan(atr.iloc[-1]) else 0.0

    if suggested == 'long':
        stop_loss = entry - config['sl_atr_mult'] * latest_atr
        take_profit = entry + config['tp_atr_mult'] * latest_atr
    elif suggested == 'short':
        stop_loss = entry + config['sl_atr_mult'] * latest_atr
        take_profit = entry - config['tp_atr_mult'] * latest_atr
    else:
        stop_loss = None
        take_profit = None

    res.update({
        'latest_time': latest_idx,
        'entry': entry,
        'atr': latest_atr,
        'signals': signals,
        'score': score,
        'suggested': suggested,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'rsi': float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else None,
        'ema9': float(ema9.iloc[-1]),
        'ema21': float(ema21.iloc[-1]),
    })

    return res


# ---------------------------
# Summary / Reporting
# ---------------------------

def make_text_summary(ticker: str, analysis: Dict) -> str:
    if 'reason' in analysis:
        return f"{ticker}: skipped ({analysis['reason']})"

    t = analysis['latest_time']
    time_str = t.strftime('%Y-%m-%d %H:%M') if hasattr(t, 'strftime') else str(t)
    s = f"{ticker} | {time_str} | Close={analysis['entry']:.4f} | Side={analysis['suggested'].upper()} | Score={analysis['score']}\n"

    if analysis['suggested'] != 'neutral':
        s += f"  ATR={analysis['atr']:.4f} | SL={analysis['stop_loss']:.4f} | TP={analysis['take_profit']:.4f}\n"
    else:
        s += f"  ATR={analysis['atr']:.4f} | No trade suggestion\n"

    if analysis['signals']:
        sigs = ', '.join([f"{p[0]}:{p[1]}" for p in analysis['signals']])
        s += f"  Signals: {sigs}\n"
    s += f"  RSI={analysis.get('rsi'):.1f} | EMA9={analysis.get('ema9'):.4f} EMA21={analysis.get('ema21'):.4f}\n"
    return s


# ---------------------------
# Main loop
# ---------------------------

def run_loop(config: dict):
    tickers = config['tickers']
    interval = config['interval']
    lookback = config['lookback_period_days']

    print(f"Starting 5-min scanner for {len(tickers)} tickers at {interval} timeframe. Poll every {config['poll_interval_seconds']}s")

    while True:
        try:
            start_ts = dt.datetime.utcnow()
            data = fetch_ohlc(tickers, interval, lookback)
            analyses = {}
            summaries = []

            for t in tickers:
                df = data.get(t)
                if df is None or df.empty:
                    analyses[t] = {'reason': 'no data'}
                else:
                    analyses[t] = analyze_dataframe(df, config)
                summaries.append(make_text_summary(t, analyses[t]))

            out_text = "\n".join(summaries)
            header = f"=== Market summary @ {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (local) ==="
            final = header + "\n" + out_text

            print(final)

            if config.get('output_file'):
                with open(config['output_file'], 'a') as f:
                    f.write(final + "\n\n")

            elapsed = (dt.datetime.utcnow() - start_ts).total_seconds()
            sleep_time = max(1, config['poll_interval_seconds'] - elapsed)
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("Interrupted by user — exiting.")
            break
        except Exception as e:
            print("Error in main loop:", str(e))
            # wait a bit before retry
            time.sleep(min(60, config['poll_interval_seconds']))


if __name__ == '__main__':
    run_loop(CONFIG)
