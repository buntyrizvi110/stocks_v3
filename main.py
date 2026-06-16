import os, json, re, asyncio, threading, math, time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from openai import OpenAI

# ============================================================
# ENV - set these in Azure App Service Configuration / local shell. Do not hardcode keys in this file.
# ============================================================
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LOCAL_TIMEZONE = os.getenv("LOCAL_TIMEZONE", "Asia/Dubai")

CAPITAL_BASE_URL = os.getenv("CAPITAL_BASE_URL", "https://demo-api-capital.backend-capital.com/api/v1")
CAPITAL_API_KEY = os.getenv("CAPITAL_API_KEY", "")
CAPITAL_IDENTIFIER = os.getenv("CAPITAL_IDENTIFIER", "")
CAPITAL_PASSWORD = os.getenv("CAPITAL_PASSWORD", "")

CAPITAL_SESSION = {"cst": "", "security": "", "ts": 0}
CAPITAL_LOCK = threading.Lock()

app = FastAPI(title="AI Market Navigator - Capital.com - By Syed Abbas")
executor = ThreadPoolExecutor(max_workers=int(os.getenv('APP_WORKERS', '3')))

# ============================================================
# AZURE GLOBAL HIT COUNTER + SIGNAL MEMORY
# ============================================================
HIT_COUNTER_FILE = Path(os.getenv("HIT_COUNTER_FILE", "/home/site/hit_counter.json"))
SIGNAL_TRACKER_FILE = Path(os.getenv("SIGNAL_TRACKER_FILE", "/home/site/signal_tracker.json"))
if not HIT_COUNTER_FILE.parent.exists():
    HIT_COUNTER_FILE = Path("hit_counter.json")
if not SIGNAL_TRACKER_FILE.parent.exists():
    SIGNAL_TRACKER_FILE = Path("signal_tracker.json")

HIT_LOCK = threading.Lock()
TRACKER_LOCK = threading.Lock()

# ============================================================
# AZURE PERFORMANCE CACHE
# ============================================================
# Keeps UI unchanged while avoiding repeated Yahoo/Finnhub/OpenAI calls on every refresh.
# Azure App Service can be slow with repeated external calls; these caches keep responses fast.
CACHE_LOCK = threading.RLock()
CANDLE_CACHE = {}
NEWS_CACHE = {}
AI_CACHE = {}
RESPONSE_CACHE = {}
CHART_CACHE = {}

CANDLE_TTL_SECONDS = int(os.getenv("CANDLE_TTL_SECONDS", "180"))     # Capital.com candle data
NEWS_TTL_SECONDS = int(os.getenv("NEWS_TTL_SECONDS", "300"))        # Finnhub news
AI_TTL_SECONDS = int(os.getenv("AI_TTL_SECONDS", "600"))            # OpenAI sentiment
RESPONSE_TTL_SECONDS = int(os.getenv("RESPONSE_TTL_SECONDS", "25")) # smooth repeat refresh
CHART_TTL_SECONDS = int(os.getenv("CHART_TTL_SECONDS", "120"))      # chart JSON cache

def _safe_copy(value):
    try:
        if isinstance(value, pd.DataFrame):
            return value.copy(deep=False)
        if isinstance(value, dict):
            return json.loads(json.dumps(value))
        if isinstance(value, list):
            return json.loads(json.dumps(value))
        return value.copy() if hasattr(value, "copy") else value
    except Exception:
        return value

def _cache_get(cache, key, ttl):
    now = time.time()
    with CACHE_LOCK:
        item = cache.get(key)
        if not item:
            return None
        ts, value = item
        if now - ts <= ttl:
            return _safe_copy(value)
        cache.pop(key, None)
    return None

def _cache_set(cache, key, value):
    with CACHE_LOCK:
        cache[key] = (time.time(), _safe_copy(value))
        # prevent unbounded growth on long-running Azure instances
        if len(cache) > 200:
            oldest = sorted(cache.items(), key=lambda kv: kv[1][0])[:50]
            for k, _ in oldest:
                cache.pop(k, None)

def _news_fingerprint(news):
    try:
        return "|".join([(n.get("headline","")[:80] + str(n.get("source",""))) for n in (news or [])[:8]])
    except Exception:
        return ""



def read_hit_count() -> int:
    try:
        if HIT_COUNTER_FILE.exists():
            data = json.loads(HIT_COUNTER_FILE.read_text())
            return int(data.get("hits", 0))
    except Exception:
        pass
    return 0


def write_hit_count(count: int) -> None:
    try:
        HIT_COUNTER_FILE.parent.mkdir(parents=True, exist_ok=True)
        HIT_COUNTER_FILE.write_text(json.dumps({"hits": int(count)}))
    except Exception:
        pass


ASSETS = {
    "GOLD": {"name": "Gold Spot", "icon": "🟡", "epic": "GOLD", "keywords": ["gold", "xau", "bullion", "safe haven", "inflation", "fed", "rates", "dollar"]},
    "SILVER": {"name": "Silver", "icon": "⚪", "epic": "SILVER", "keywords": ["silver", "xag", "precious metal", "industrial metal", "solar", "dollar", "rates"]},
    "WTI": {"name": "Crude Oil WTI", "icon": "🛢️", "epic": "OIL_CRUDE", "keywords": ["wti", "crude", "oil", "opec", "iran", "gulf", "hormuz", "sanctions", "us strikes", "middle east"]},
    "BRENT": {"name": "Brent Crude", "icon": "🛢️", "epic": "OIL_BRENT", "keywords": ["brent", "crude", "oil", "opec", "iran", "gulf", "hormuz", "sanctions", "shipping"]},
    "BTC": {"name": "Bitcoin", "icon": "₿", "epic": "BTCUSD", "keywords": ["bitcoin", "btc", "crypto", "etf", "risk assets", "liquidity", "fed", "rates"]},
    "USTEC100": {"name": "USTEC 100", "icon": "📈", "epic": "US100", "keywords": ["nasdaq", "nasdaq 100", "tech stocks", "ai stocks", "fed", "rates", "yields"]},
}

INTERVALS = {
    "1M": {"cap": "MINUTE", "max": 420},
    "15M": {"cap": "MINUTE_15", "max": 420},
    "30M": {"cap": "MINUTE_30", "max": 420},
    "1H": {"cap": "HOUR", "max": 420},
    "1D": {"cap": "DAY", "max": 220},
}


def clamp(x, lo=-100, hi=100):
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return 0.0


def safe_float(x, default=0.0):
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def clean(x):
    return re.sub(r"\s+", " ", str(x or "")).strip()


# ============================================================
# MARKET DATA
# ============================================================
def _mid_price(price_obj):
    if not isinstance(price_obj, dict):
        return np.nan
    bid = safe_float(price_obj.get("bid"), np.nan)
    ask = safe_float(price_obj.get("ask"), np.nan)
    if not pd.isna(bid) and not pd.isna(ask) and bid > 0 and ask > 0:
        return (bid + ask) / 2
    last = safe_float(price_obj.get("lastTraded"), np.nan)
    return last


def capital_login():

    with CAPITAL_LOCK:

        if (
            CAPITAL_SESSION["cst"]
            and CAPITAL_SESSION["security"]
            and time.time() - CAPITAL_SESSION["ts"] < 540
        ):
            return (
                CAPITAL_SESSION["cst"],
                CAPITAL_SESSION["security"]
            )

        url = f"{CAPITAL_BASE_URL}/session"

        headers = {
            "X-CAP-API-KEY": CAPITAL_API_KEY,
            "Content-Type": "application/json"
        }

        if not CAPITAL_API_KEY or not CAPITAL_IDENTIFIER or not CAPITAL_PASSWORD:
            raise RuntimeError("Capital.com credentials missing. Set CAPITAL_API_KEY, CAPITAL_IDENTIFIER, CAPITAL_PASSWORD.")

        payload = {
            "identifier": CAPITAL_IDENTIFIER,
            "password": CAPITAL_PASSWORD,
            "encryptedPassword": False
        }

        with httpx.Client(timeout=15) as client:
            r = client.post(
                url,
                headers=headers,
                json=payload
            )

        r.raise_for_status()

        cst = r.headers.get("CST")
        security = r.headers.get("X-SECURITY-TOKEN")

        if not cst or not security:
            raise RuntimeError(
                "Capital login succeeded but tokens not returned"
            )

        CAPITAL_SESSION["cst"] = cst
        CAPITAL_SESSION["security"] = security
        CAPITAL_SESSION["ts"] = time.time()

        return cst, security
        
def get_capital_live_price(epic):
    cst, security = capital_login()

    url = f"{CAPITAL_BASE_URL.rstrip('/')}/markets/{epic}"
    headers = {
        "X-CAP-API-KEY": CAPITAL_API_KEY,
        "CST": cst,
        "X-SECURITY-TOKEN": security,
    }

    with httpx.Client(timeout=10) as client:
        r = client.get(url, headers=headers)

        if r.status_code in (401, 403):
            CAPITAL_SESSION.update({"cst": "", "security": "", "ts": 0})
            cst, security = capital_login()
            headers["CST"] = cst
            headers["X-SECURITY-TOKEN"] = security
            r = client.get(url, headers=headers)

        r.raise_for_status()
        data = r.json()

    snapshot = data.get("snapshot", {})

    bid = safe_float(snapshot.get("bid"), 0)
    offer = safe_float(snapshot.get("offer"), 0)

    if bid > 0 and offer > 0:
        return round(offer, 3)

    if bid > 0:
        return round(bid, 3)

    if offer > 0:
        return round(offer, 3)

    return 0
    

def clean_market_candles(df, tf="30M"):
    """
    Clean broker candle impurities before indicators/chart:
    - remove weekend rows
    - repair invalid OHLC ordering
    - remove extreme bad ticks / broken candles using rolling median range
    - keep gaps as gaps; frontend rangebreaks compress non-trading weekends
    """
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()
    d["time"] = pd.to_datetime(d["time"], errors="coerce", utc=True)
    d = d.dropna(subset=["time", "open", "high", "low", "close"])
    for c in ["open", "high", "low", "close", "volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["open", "high", "low", "close"])
    d = d[(d["open"] > 0) & (d["high"] > 0) & (d["low"] > 0) & (d["close"] > 0)]
    d = d.sort_values("time").drop_duplicates("time")

    # Weekend quotes from CFD feeds can create visible gaps/flat impurities on intraday charts.
    # Keep daily candles intact, but remove weekend intraday timestamps.
    if str(tf).upper() in {"1M", "15M", "30M", "1H"}:
        d = d[d["time"].dt.weekday < 5]

    if d.empty:
        return pd.DataFrame()

    # Repair any broker row where high/low are inconsistent with open/close.
    ohlc_max = d[["open", "high", "low", "close"]].max(axis=1)
    ohlc_min = d[["open", "high", "low", "close"]].min(axis=1)
    d["high"] = ohlc_max
    d["low"] = ohlc_min

    if len(d) >= 40:
        rng = (d["high"] - d["low"]).abs()
        body = (d["close"] - d["open"]).abs()
        ref_range = rng.rolling(30, min_periods=10).median().replace(0, np.nan)
        ref_price = d["close"].rolling(30, min_periods=10).median().replace(0, np.nan)
        jump = d["close"].pct_change().abs()

        bad_range = (rng > ref_range * 8) & (rng > d["close"] * 0.006)
        bad_body = (body > ref_range * 10) & (body > d["close"] * 0.008)
        bad_jump = (jump > 0.08) & ((d["close"] - ref_price).abs() > ref_price * 0.06)
        bad = (bad_range | bad_body | bad_jump).fillna(False)
        if bad.any() and bad.mean() < 0.08:
            d = d.loc[~bad].copy()

    d = d.sort_values("time").drop_duplicates("time")
    return d.reset_index(drop=True)


def compute_daily_change(df):
    """Return realistic 24h/previous-session change from cleaned candles."""
    try:
        d = df.copy().dropna(subset=["time", "close"])
        if d.empty or len(d) < 2:
            return {"value": 0.0, "percent": 0.0}
        d["time"] = pd.to_datetime(d["time"], errors="coerce", utc=True)
        d["close"] = pd.to_numeric(d["close"], errors="coerce")
        d = d.dropna(subset=["time", "close"]).sort_values("time")
        last_time = d["time"].iloc[-1]
        last = safe_float(d["close"].iloc[-1])
        prior_rows = d[d["time"] <= last_time - pd.Timedelta(hours=24)]
        prior = safe_float(prior_rows["close"].iloc[-1]) if not prior_rows.empty else safe_float(d["close"].iloc[max(0, len(d)-2)])
        if prior <= 0 or last <= 0:
            return {"value": 0.0, "percent": 0.0}
        value = last - prior
        pct = (value / prior) * 100
        # Guard against one-off broker outliers so the header never displays absurd percentages.
        if abs(pct) > 20:
            prior = safe_float(d["close"].iloc[-2])
            value = last - prior
            pct = (value / prior) * 100 if prior > 0 else 0.0
        return {"value": round(value, 3), "percent": round(pct, 3)}
    except Exception:
        return {"value": 0.0, "percent": 0.0}

def _load_candles_uncached(asset_key, tf):
    epic = ASSETS[asset_key]["epic"]
    cfg = INTERVALS[tf]

    cst, security = capital_login()

    url = f"{CAPITAL_BASE_URL.rstrip('/')}/prices/{epic}"
    headers = {
        "X-CAP-API-KEY": CAPITAL_API_KEY,
        "CST": cst,
        "X-SECURITY-TOKEN": security,
    }
    params = {
        "resolution": cfg["cap"],
        "max": cfg["max"],
    }

    try:
        with httpx.Client(timeout=12) as client:
            r = client.get(url, headers=headers, params=params)

            if r.status_code in (401, 403):
                CAPITAL_SESSION.update({"cst": "", "security": "", "ts": 0})
                cst, security = capital_login()
                headers["CST"] = cst
                headers["X-SECURITY-TOKEN"] = security
                r = client.get(url, headers=headers, params=params)

            r.raise_for_status()
            data = r.json()
    except Exception as e:
        raise RuntimeError(f"Capital.com candle fetch failed for {epic}: {str(e)[:160]}")

    prices = data.get("prices", [])
    rows = []

    for p in prices:
        t = p.get("snapshotTimeUTC") or p.get("snapshotTime")
        rows.append({
            "time": pd.to_datetime(t, errors="coerce", utc=True),
            "open": _mid_price(p.get("openPrice")),
            "high": _mid_price(p.get("highPrice")),
            "low": _mid_price(p.get("lowPrice")),
            "close": _mid_price(p.get("closePrice")),
            "volume": safe_float(p.get("lastTradedVolume"), 0),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()

    df = clean_market_candles(df, tf)

    if len(df) < 30 or df["close"].nunique() <= 2:
        return pd.DataFrame()

    df = df.tail(420).copy()

    live_price = get_capital_live_price(epic)

    if live_price > 0:
        df.loc[df.index[-1], "close"] = live_price
    
        if live_price > df.loc[df.index[-1], "high"]:
            df.loc[df.index[-1], "high"] = live_price
    
        if live_price < df.loc[df.index[-1], "low"]:
            df.loc[df.index[-1], "low"] = live_price
    
    return df
def load_candles(asset_key, tf):
    key = (asset_key.upper(), tf.upper())

    cached = _cache_get(
        CANDLE_CACHE,
        key,
        CANDLE_TTL_SECONDS
    )

    if cached is not None:
        return cached

    df = _load_candles_uncached(asset_key, tf)

    if df is not None and not df.empty:
        _cache_set(
            CANDLE_CACHE,
            key,
            df
        )

    return df
    
# ============================================================
# INDICATORS + INSTITUTIONAL FILTERS
# ============================================================
def add_indicators(df):
    d = df.copy().sort_values("time")
    d["ema9"] = d["close"].ewm(span=9, adjust=False).mean()
    d["ema21"] = d["close"].ewm(span=21, adjust=False).mean()
    d["ema50"] = d["close"].ewm(span=50, adjust=False).mean()
    d["ema200"] = d["close"].ewm(span=200, adjust=False).mean()

    delta = d["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    d["rsi"] = (100 - (100 / (1 + rs))).fillna(50)

    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]

    tr1 = d["high"] - d["low"]
    tr2 = (d["high"] - d["close"].shift()).abs()
    tr3 = (d["low"] - d["close"].shift()).abs()
    d["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    d["atr"] = d["tr"].ewm(alpha=1/14, adjust=False).mean().fillna(d["tr"].mean())
    d["atr_pct"] = (d["atr"] / d["close"] * 100).replace([np.inf, -np.inf], np.nan).fillna(0)

    # ADX calculation for market regime detection
    up_move = d["high"].diff()
    down_move = -d["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = d["atr"].replace(0, np.nan)
    plus_di = 100 * pd.Series(plus_dm, index=d.index).ewm(alpha=1/14, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=d.index).ewm(alpha=1/14, adjust=False).mean() / atr
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    d["adx"] = dx.ewm(alpha=1/14, adjust=False).mean().fillna(15)
    d["plus_di"] = plus_di.fillna(0)
    d["minus_di"] = minus_di.fillna(0)

    d["vol_ma"] = d["volume"].rolling(20).mean().fillna(d["volume"].mean())
    d["ret"] = d["close"].pct_change().fillna(0)
    return d


def support_resistance(df, lookback=80):
    d = df.tail(lookback).copy()
    price = safe_float(d["close"].iloc[-1])
    supports = []
    resistances = []
    for i in range(2, len(d) - 2):
        lo = d["low"].iloc[i]
        hi = d["high"].iloc[i]
        if lo <= d["low"].iloc[i-2:i+3].min():
            supports.append(float(lo))
        if hi >= d["high"].iloc[i-2:i+3].max():
            resistances.append(float(hi))
    supports = [x for x in supports if x < price]
    resistances = [x for x in resistances if x > price]
    sup = max(supports) if supports else float(d["low"].min())
    res = min(resistances) if resistances else float(d["high"].max())
    return {"support": round(sup, 3), "resistance": round(res, 3)}


def volume_profile(df, bins=24):
    d = df.tail(120).copy()
    typical = (d["high"] + d["low"] + d["close"]) / 3
    vol = d["volume"].replace(0, np.nan)
    if vol.isna().all() or vol.sum(skipna=True) <= 0:
        vol = pd.Series(np.ones(len(d)), index=d.index)
    try:
        cats = pd.cut(typical, bins=bins, duplicates="drop")
        grouped = vol.groupby(cats, observed=False).sum()
        if grouped.empty:
            return {"poc": safe_float(d["close"].iloc[-1]), "bias": 0}
        poc_interval = grouped.idxmax()
        poc = float((poc_interval.left + poc_interval.right) / 2)
        price = safe_float(d["close"].iloc[-1])
        atr = safe_float((d["high"] - d["low"]).tail(14).mean(), price * 0.005)
        bias = clamp((price - poc) / max(atr, price * 0.001) * 10, -15, 15)
        return {"poc": round(poc, 3), "bias": round(bias, 3)}
    except Exception:
        return {"poc": safe_float(d["close"].iloc[-1]), "bias": 0}


def market_regime(df):
    d = add_indicators(df)
    last = d.iloc[-1]
    adx = safe_float(last.adx, 15)
    atr_pct = safe_float(last.atr_pct, 0)
    ema_spread = abs(safe_float(last.ema21 - last.ema50)) / max(safe_float(last.close), 1) * 100
    if adx >= 25 and ema_spread > atr_pct * 0.20:
        regime = "TRENDING"
        multiplier = 1.12
    elif atr_pct > d["atr_pct"].tail(120).quantile(0.75):
        regime = "HIGH VOLATILITY"
        multiplier = 0.88
    elif adx < 17:
        regime = "RANGING"
        multiplier = 0.78
    else:
        regime = "NORMAL"
        multiplier = 1.0
    return {"regime": regime, "adx": round(adx, 2), "atr_pct": round(atr_pct, 3), "multiplier": multiplier}


def raw_technical_model(df):
    d = add_indicators(df)
    if len(d) < 60:
        return 0.0
    last = d.iloc[-1]
    prev = d.iloc[-4] if len(d) > 4 else d.iloc[-2]
    score = 0.0

    # Trend stack + price location
    score += 18 if last.ema9 > last.ema21 else -18
    score += 16 if last.ema21 > last.ema50 else -16
    score += 16 if last.close > last.ema200 else -16
    score += 10 if last.close > last.ema21 else -10

    # Momentum quality
    score += 14 if last.macd > last.macd_signal else -14
    score += clamp(last.macd_hist / max(last.atr, last.close * 0.001) * 25, -12, 12)
    mom10 = (last.close - d["close"].iloc[-10]) / max(d["close"].iloc[-10], 1) * 100
    mom20 = (last.close - d["close"].iloc[-20]) / max(d["close"].iloc[-20], 1) * 100
    score += clamp(mom10 * 18, -12, 12)
    score += clamp(mom20 * 10, -10, 10)

    # RSI: continuation zone is stronger than overbought/oversold alone
    rsi = safe_float(last.rsi, 50)
    if 52 <= rsi <= 68:
        score += 10
    elif 32 <= rsi <= 48:
        score -= 10
    elif rsi > 75:
        score -= 8
    elif rsi < 25:
        score += 8

    # ADX directional confirmation
    if last.adx >= 20:
        score += 8 if last.plus_di > last.minus_di else -8

    # Breakout / breakdown against recent range
    recent_high = d["high"].iloc[-31:-1].max()
    recent_low = d["low"].iloc[-31:-1].min()
    if last.close > recent_high:
        score += 10
    elif last.close < recent_low:
        score -= 10

    # Penalize sudden reversal against the signal
    if score > 0 and last.close < prev.close and last.macd_hist < prev.macd_hist:
        score -= 8
    if score < 0 and last.close > prev.close and last.macd_hist > prev.macd_hist:
        score += 8

    return clamp(score)


def backtest_technical(df, horizon=3):
    """
    Fast Azure-safe backtest approximation.
    Keeps the same output schema but avoids repeatedly recalculating indicators
    inside a Python loop, which was a major reason for slow refreshes.
    """
    d = add_indicators(df).tail(240).reset_index(drop=True)
    if len(d) < 90:
        return {"win_rate": 0.50, "trades": 0, "expectancy": 0.0, "score_adj": 0}

    trend = np.where(d["ema9"] > d["ema21"], 1, -1)
    trend += np.where(d["ema21"] > d["ema50"], 1, -1)
    trend += np.where(d["close"] > d["ema200"], 1, -1)
    momentum = np.where(d["macd"] > d["macd_signal"], 1, -1)
    rsi_ok = np.where(d["rsi"] >= 52, 1, np.where(d["rsi"] <= 48, -1, 0))
    proxy_score = trend * 14 + momentum * 14 + rsi_ok * 8

    entries = np.where(np.abs(proxy_score) >= 24)[0]
    entries = entries[(entries >= 70) & (entries < len(d) - horizon)]
    if len(entries) == 0:
        return {"win_rate": 0.50, "trades": 0, "expectancy": 0.0, "score_adj": 0}

    close = d["close"].to_numpy(dtype=float)
    atr = d["atr"].replace(0, np.nan).fillna(d["close"] * 0.005).to_numpy(dtype=float)
    direction = np.where(proxy_score[entries] > 0, 1, -1)
    entry = close[entries]
    exitp = close[entries + horizon]
    denom = np.maximum(atr[entries], entry * 0.0005)
    r_mult = ((exitp - entry) * direction) / denom

    if len(r_mult) == 0:
        return {"win_rate": 0.50, "trades": 0, "expectancy": 0.0, "score_adj": 0}

    win_rate = float((r_mult > 0).mean())
    expectancy = float(np.mean(r_mult))
    score_adj = clamp((win_rate - 0.50) * 50 + expectancy * 8, -12, 12)
    return {"win_rate": round(win_rate, 3), "trades": int(len(r_mult)), "expectancy": round(expectancy, 3), "score_adj": round(score_adj, 2)}


def multi_timeframe_confirmation(dfs):
    # Signals are still based on 30M; higher frames only confirm or reduce confidence.
    scores = {}
    for tf, df in dfs.items():
        if df is not None and not df.empty:
            scores[tf] = raw_technical_model(df)
    base = scores.get("30M", 0)
    if not scores:
        return {"score": 0, "alignment": 0, "scores": {}}
    signs = []
    for tf in ["15M", "30M", "1H", "1D"]:
        s = scores.get(tf)
        if s is None:
            continue
        signs.append(1 if s > 12 else -1 if s < -12 else 0)
    base_sign = 1 if base > 0 else -1 if base < 0 else 0
    aligned = sum(1 for s in signs if s == base_sign and s != 0)
    opposed = sum(1 for s in signs if s == -base_sign and s != 0)
    alignment = aligned - opposed
    score = clamp(alignment * 7, -18, 18)
    return {"score": round(score, 2), "alignment": alignment, "scores": {k: round(v, 2) for k, v in scores.items()}}


def economic_event_filter(news, asset_key):
    text = " ".join([(n.get("headline", "") + " " + n.get("summary", "")) for n in news]).lower()
    high_impact = ["fed", "fomc", "powell", "cpi", "inflation", "jobs report", "nonfarm", "nfp", "pce", "rate decision", "ecb", "opec", "eia", "inventory", "war", "strike", "hormuz", "sanction"]
    hits = [w for w in high_impact if w in text]
    risk_penalty = min(12, len(hits) * 2)
    if asset_key in ["WTI", "BRENT"] and any(w in text for w in ["opec", "eia", "inventory", "hormuz", "iran"]):
        risk_penalty = max(risk_penalty, 6)
    if asset_key in ["GOLD", "SILVER", "USTEC100", "BTC"] and any(w in text for w in ["fed", "cpi", "inflation", "powell", "rate"]):
        risk_penalty = max(risk_penalty, 6)
    return {"hits": hits[:6], "risk_penalty": risk_penalty}


def optimized_levels(df, direction):
    d = add_indicators(df)
    last = d.iloc[-1]
    price = safe_float(last.close)
    atr = max(safe_float(last.atr, price * 0.006), price * 0.001)
    sr = support_resistance(d)
    vp = volume_profile(d)
    if direction >= 0:
        raw_stop = price - atr * 1.35
        sr_stop = min(raw_stop, sr["support"] - atr * 0.15) if sr["support"] < price else raw_stop
        stop = sr_stop
        target = price + max(atr * 2.15, (price - stop) * 1.65)
    else:
        raw_stop = price + atr * 1.35
        sr_stop = max(raw_stop, sr["resistance"] + atr * 0.15) if sr["resistance"] > price else raw_stop
        stop = sr_stop
        target = price - max(atr * 2.15, (stop - price) * 1.65)
    return {"entry": round(price, 3), "target": round(target, 3), "stop": round(stop, 3), "support": sr["support"], "resistance": sr["resistance"], "poc": vp["poc"]}


def technical_score(df, mtf=None):
    d = add_indicators(df)
    last = d.iloc[-1]
    base = raw_technical_model(d)
    regime = market_regime(d)
    sr = support_resistance(d)
    vp = volume_profile(d)
    bt = backtest_technical(d)
    mtf_score = safe_float((mtf or {}).get("score", 0))

    # Support/resistance and volume profile confirmation
    price = safe_float(last.close)
    sr_bias = 0
    if price > sr["resistance"]:
        sr_bias += 8
    elif price < sr["support"]:
        sr_bias -= 8
    vp_bias = safe_float(vp.get("bias", 0))

    final_score = (base + mtf_score + sr_bias + vp_bias + bt["score_adj"]) * regime["multiplier"]
    final_score = clamp(final_score)
    direction = 1 if final_score >= 0 else -1
    levels = optimized_levels(d, direction)

    return {
        "score": round(final_score, 2),
        "price": round(price, 3),
        "rsi": round(safe_float(last.rsi), 2),
        "macd": round(safe_float(last.macd), 3),
        "atr": round(safe_float(last.atr, price * 0.01), 3),
        "entry": levels["entry"],
        "target": levels["target"],
        "stop": levels["stop"],
        "support": levels["support"],
        "resistance": levels["resistance"],
        "entry_zone": (
            f"{min(levels['support'], levels['resistance']):.3f} - {max(levels['support'], levels['resistance']):.3f}"
            if levels.get("support") and levels.get("resistance") else f"{levels['entry']:.3f}"
        ),
        "poc": levels["poc"],
        "regime": regime,
        "backtest": bt,
        "raw_score": round(base, 2),
    }


# ============================================================
# NEWS + AI - news display kept as original behavior
# ============================================================
async def fetch_finnhub_news(asset_key):
    cache_key = asset_key.upper()
    cached = _cache_get(NEWS_CACHE, cache_key, NEWS_TTL_SECONDS)
    if cached is not None:
        return cached
    if not FINNHUB_API_KEY:
        return []
    url = "https://finnhub.io/api/v1/news"
    params = {"category": "general", "token": FINNHUB_API_KEY}
    try:
        async with httpx.AsyncClient(timeout=6) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            raw = r.json()
    except Exception:
        return []

    keywords = [k.lower() for k in ASSETS[asset_key]["keywords"]]
    geopolitics = ["iran", "hormuz", "gulf", "sanction", "strike", "middle east", "shipping", "war", "ceasefire", "deal"]
    out = []
    for n in raw:
        headline = clean(n.get("headline"))
        summary = clean(n.get("summary"))
        source = clean(n.get("source"))
        text = f"{headline} {summary}".lower()
        relevance = sum(1 for k in keywords if k in text)
        relevance += sum(1 for k in geopolitics if k in text) * 1.3
        if relevance > 0:
            out.append({"headline": headline, "summary": summary[:180], "source": source, "url": n.get("url", ""), "relevance": relevance})
    final_news = sorted(out, key=lambda x: x["relevance"], reverse=True)[:10]
    _cache_set(NEWS_CACHE, cache_key, final_news)
    return final_news


def news_lexicon_score(news, asset_key):
    if not news:
        return 0
    text = " ".join([n["headline"] + " " + n["summary"] for n in news]).lower()
    bullish = ["rally", "surge", "gain", "rise", "rebound", "strong demand", "supply cut", "disruption", "shortage", "drawdown", "inventory draw", "sanctions", "strike", "hormuz", "shipping risk", "safe haven", "rate cut", "weak dollar", "breakout", "record high"]
    bearish = ["drop", "fall", "slump", "selloff", "weak demand", "inventory build", "surplus", "oversupply", "recession", "strong dollar", "rate hike", "peace deal", "ceasefire", "supply restored", "risk off"]
    score = 0
    for w in bullish:
        if w in text:
            score += 8
    for w in bearish:
        if w in text:
            score -= 8
    if asset_key in ["WTI", "BRENT"]:
        if any(x in text for x in ["iran", "hormuz", "gulf", "sanction", "strike", "shipping risk"]):
            score += 20
        if any(x in text for x in ["inventory build", "oversupply", "weak demand"]):
            score -= 20
    return clamp(score)


async def openai_sentiment(asset_key, news, tech, institutional_context=None):
    ai_key = (
        asset_key.upper(),
        round(safe_float(tech.get("score", 0)), 1),
        round(safe_float(tech.get("price", 0)), 1),
        round(safe_float(tech.get("rsi", 50)), 1),
        _news_fingerprint(news),
    )
    cached = _cache_get(AI_CACHE, ai_key, AI_TTL_SECONDS)
    if cached is not None:
        return cached

    if not OPENAI_API_KEY:
        return {"score": 0, "bias": "NEUTRAL", "summary": "OpenAI key missing. Using fallback sentiment.", "risk": "AI sentiment unavailable."}
    headlines = [f"{n['source']}: {n['headline']} - {n['summary']}" for n in news[:8]]
    ctx = institutional_context or {}
    prompt = f"""
Asset: {ASSETS[asset_key]['name']}
30M Technical Score={tech['score']}
Price={tech['price']} RSI={tech['rsi']} MACD={tech['macd']} ATR={tech['atr']}
Regime={tech.get('regime', {}).get('regime')} Backtest={tech.get('backtest')}
Support={tech.get('support')} Resistance={tech.get('resistance')} VolumePOC={tech.get('poc')}
MTF={ctx.get('mtf')} EventRisk={ctx.get('event_risk')}

News:
{chr(10).join(headlines)}

Return only JSON:
{{"score": number between -100 and 100,"bias": "BULLISH" or "BEARISH" or "NEUTRAL","summary": "short reason","risk": "short risk"}}
"""
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "Return valid JSON only. Be conservative and do not overrule strong 30M technical evidence without clear news catalyst."}, {"role": "user", "content": prompt}],
            temperature=0,
        )
        text = response.choices[0].message.content.strip()
        text = re.sub(r"^```json|```$", "", text, flags=re.I).strip()
        data = json.loads(text)
        result = {"score": clamp(data.get("score", 0)), "bias": data.get("bias", "NEUTRAL"), "summary": data.get("summary", ""), "risk": data.get("risk", "")}
        _cache_set(AI_CACHE, ai_key, result)
        return result
    except Exception as e:
        return {"score": 0, "bias": "NEUTRAL", "summary": "OpenAI sentiment failed. Fallback active.", "risk": str(e)[:100]}


def read_tracker():
    try:
        if SIGNAL_TRACKER_FILE.exists():
            return json.loads(SIGNAL_TRACKER_FILE.read_text())
    except Exception:
        pass
    return {"signals": []}


def write_tracker(data):
    try:
        SIGNAL_TRACKER_FILE.parent.mkdir(parents=True, exist_ok=True)
        SIGNAL_TRACKER_FILE.write_text(json.dumps(data)[-250000:])
    except Exception:
        pass


def update_signal_tracker(asset_key, signal, price, confidence):
    # lightweight live paper-trading validation memory; does not alter UI schema
    if signal not in ["BUY SIGNAL", "SELL SIGNAL"]:
        return {"paper_trades": 0, "recent_accuracy": 0.5}
    with TRACKER_LOCK:
        data = read_tracker()
        rows = data.get("signals", [])
        now = datetime.now(timezone.utc).isoformat()
        direction = 1 if signal == "BUY SIGNAL" else -1
        rows.append({"asset": asset_key, "time": now, "signal": signal, "direction": direction, "price": price, "confidence": confidence})
        rows = rows[-300:]
        same = [r for r in rows if r.get("asset") == asset_key]
        # Approximate live tracker from subsequent observed prices on refreshes.
        closed = []
        if len(same) >= 2:
            latest_price = price
            for r in same[:-1][-80:]:
                move = (latest_price - safe_float(r.get("price"))) * int(r.get("direction", 1))
                closed.append(1 if move > 0 else 0)
        acc = float(np.mean(closed)) if closed else 0.5
        data["signals"] = rows
        write_tracker(data)
        return {"paper_trades": len(closed), "recent_accuracy": round(acc, 3)}


def probability_from_inputs(fusion, tech, backtest, event_risk):
    bt_wr = safe_float(backtest.get("win_rate", 0.5), 0.5)
    bt_trades = int(backtest.get("trades", 0) or 0)
    bt_weight = min(1.0, bt_trades / 40.0)
    base = 1 / (1 + math.exp(-abs(fusion) / 24.0))
    prob = (base * 0.72) + ((bt_wr * bt_weight + 0.5 * (1 - bt_weight)) * 0.28)
    prob -= safe_float(event_risk.get("risk_penalty", 0)) / 250.0
    return round(max(0.45, min(0.86, prob)), 3)


def fusion_signal(tech, news_score, ai, mtf=None, event_risk=None, tracker=None):
    tech_s = clamp(tech["score"])
    news_s = clamp(news_score)
    ai_s = clamp(ai.get("score", 0))
    mtf_s = clamp((mtf or {}).get("score", 0))
    event_penalty = safe_float((event_risk or {}).get("risk_penalty", 0))

    # Adaptive fusion: if news/AI are neutral/missing, do not force HOLD; let 30M technical + MTF dominate.
    active_news = abs(news_s) >= 8
    active_ai = abs(ai_s) >= 8
    if active_news and active_ai:
        fusion = tech_s * 0.50 + news_s * 0.18 + ai_s * 0.22 + mtf_s * 0.10
    elif active_ai:
        fusion = tech_s * 0.62 + ai_s * 0.25 + mtf_s * 0.13
    elif active_news:
        fusion = tech_s * 0.64 + news_s * 0.22 + mtf_s * 0.14
    else:
        fusion = tech_s * 0.78 + mtf_s * 0.22

    # Penalize confidence during high-impact event risk, but do not blindly flip direction.
    if fusion > 0:
        fusion -= event_penalty * 0.45
    elif fusion < 0:
        fusion += event_penalty * 0.45
    fusion = clamp(fusion)

    regime_name = tech.get("regime", {}).get("regime", "NORMAL")
    threshold = 22
    if regime_name == "TRENDING":
        threshold = 18
    elif regime_name == "RANGING":
        threshold = 28
    elif regime_name == "HIGH VOLATILITY":
        threshold = 30

    prob = probability_from_inputs(fusion, tech, tech.get("backtest", {}), event_risk or {})

    if fusion >= threshold and prob >= 0.54:
        signal = "BUY SIGNAL"
        label = "BULLISH"
    elif fusion <= -threshold and prob >= 0.54:
        signal = "SELL SIGNAL"
        label = "BEARISH"
    else:
        signal = "HOLD / WAIT"
        label = "NEUTRAL"

    tracker_acc = safe_float((tracker or {}).get("recent_accuracy", 0.5), 0.5)
    tracker_boost = (tracker_acc - 0.5) * 10 if (tracker or {}).get("paper_trades", 0) >= 5 else 0
    confidence = min(100, max(25, abs(fusion) * 1.28 + prob * 25 + tracker_boost))

    return {
        "fusion": round(fusion, 2),
        "confidence": round(confidence, 1),
        "probability": prob,
        "signal": signal,
        "label": label,
        "tech_percent": round((tech_s + 100) / 2, 1),
        "news_percent": round((news_s + 100) / 2, 1),
        "ai_percent": round((ai_s + 100) / 2, 1),
        "threshold": threshold,
    }


# ============================================================
# CHART - UI unchanged, lightweight backend JSON
# ============================================================
def _to_local_chart_times(series):
    """
    Convert all candle timestamps to the configured local timezone before sending
    them to Plotly. Default is Asia/Dubai, override with LOCAL_TIMEZONE if needed.
    """
    try:
        local_tz = ZoneInfo(LOCAL_TIMEZONE)
    except Exception:
        local_tz = ZoneInfo("Asia/Dubai")
    try:
        times = pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(local_tz)
        return times.dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
    except Exception:
        return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S").tolist()


def _chart_tick_arrays(times, max_ticks=4):
    """Return sparse x-axis ticks with two-row date/time labels."""
    vals = [str(t) for t in (times or []) if str(t) and str(t).lower() != "nat"]
    if not vals:
        return [], []
    step = max(1, int(math.ceil(len(vals) / max_ticks)))
    tickvals = vals[::step]
    if vals[-1] not in tickvals:
        tickvals.append(vals[-1])
    ticktext = []
    for v in tickvals:
        try:
            dt = pd.to_datetime(v, errors="coerce")
            if pd.isna(dt):
                ticktext.append(v)
            else:
                ticktext.append(dt.strftime("%d %b %Y<br>%H:%M"))
        except Exception:
            ticktext.append(v)
    return tickvals, ticktext


def build_chart(df, asset_name):
    """
    Azure-light chart builder.
    Keeps the same frontend Plotly.js UI, but removes the heavy Python plotly package.
    This reduces build size and memory usage during Azure Oryx deployment.
    """
    try:
        last_time = str(pd.to_datetime(df["time"].iloc[-1]))
        last_close = round(safe_float(df["close"].iloc[-1]), 3)
        chart_key = (asset_name, len(df), last_time, last_close)
        cached_chart = _cache_get(CHART_CACHE, chart_key, CHART_TTL_SECONDS)
        if cached_chart is not None:
            return cached_chart
    except Exception:
        chart_key = None

    d = add_indicators(df).tail(220)
    times = _to_local_chart_times(d["time"])
    tickvals, ticktext = _chart_tick_arrays(times)

    lows = pd.to_numeric(d["low"], errors="coerce").dropna()
    highs = pd.to_numeric(d["high"], errors="coerce").dropna()
    if lows.empty or highs.empty:
        y_min, y_max = 0, 1
    else:
        y_min = float(lows.tail(200).min())
        y_max = float(highs.tail(200).max())
    pad = max((y_max - y_min) * 0.14, abs(y_max) * 0.003, 0.01)

    data = [
        {
            "type": "candlestick",
            "x": times,
            "open": d["open"].round(3).tolist(),
            "high": d["high"].round(3).tolist(),
            "low": d["low"].round(3).tolist(),
            "close": d["close"].round(3).tolist(),
            "name": asset_name,
            "increasing": {"line": {"width": 1}},
            "decreasing": {"line": {"width": 1}},
        },
        {
            "type": "scatter",
            "mode": "lines",
            "x": times,
            "y": d["ema9"].round(3).tolist(),
            "name": "EMA9",
            "line": {"width": 1},
        },
        {
            "type": "scatter",
            "mode": "lines",
            "x": times,
            "y": d["ema21"].round(3).tolist(),
            "name": "EMA21",
            "line": {"width": 1},
        },
        {
            "type": "scatter",
            "mode": "lines",
            "x": times,
            "y": d["ema50"].round(3).tolist(),
            "name": "EMA50",
            "line": {"width": 1},
        },
    ]

    layout = {
        "template": "plotly_dark",
        "height": 330,
        "margin": {"l": 58, "r": 34, "t": 22, "b": 118},
        "paper_bgcolor": "#111a26",
        "plot_bgcolor": "#111a26",
        "font": {"color": "#dce7f3", "size": 10},
        "xaxis": {
            "type": "date",
            "rangeslider": {"visible": False},
            "showgrid": True,
            "automargin": True,
            "tickmode": "array",
            "tickvals": tickvals,
            "ticktext": ticktext,
            "tickangle": 0,
            "tickfont": {"size": 9},
            "ticklabelstandoff": 10,
            "fixedrange": True,
        },
        "yaxis": {
            "range": [y_min - pad, y_max + pad],
            "fixedrange": True,
            "automargin": True,
            "zeroline": False,
        },
        "legend": {"orientation": "h", "y": 1.02, "x": 0, "font": {"size": 9}},
        "uirevision": "keep",
    }
    result = {"data": data, "layout": layout}
    if chart_key is not None:
        _cache_set(CHART_CACHE, chart_key, result)
    return result


async def process(asset_key, tf):
    response_key = (asset_key.upper(), tf.upper())
    cached_response = _cache_get(RESPONSE_CACHE, response_key, RESPONSE_TTL_SECONDS)
    if cached_response is not None:
        cached_response["updated"] = datetime.now().strftime("%H:%M:%S")
        return cached_response

    loop = asyncio.get_running_loop()

    # Chart follows UI timeframe; signal and all technicals remain fixed to 30M.
    # Performance tuning:
    # - If chart timeframe is 30M, reuse the same dataframe for signal + chart.
    # - Candle/news/AI loaders are cached, so Azure auto-refresh is fast.
    # - Higher-timeframe confirmation remains supported but usually returns from cache.
    if tf == "30M":
        signal_df = await loop.run_in_executor(executor, load_candles, asset_key, "30M")
        chart_df = signal_df.copy()
        h1_future = loop.run_in_executor(executor, load_candles, asset_key, "1H")
        d1_future = loop.run_in_executor(executor, load_candles, asset_key, "1D")
        h1_df, d1_df = await asyncio.gather(h1_future, d1_future)
    else:
        chart_future = loop.run_in_executor(executor, load_candles, asset_key, tf)
        signal_future = loop.run_in_executor(executor, load_candles, asset_key, "30M")
        h1_future = loop.run_in_executor(executor, load_candles, asset_key, "1H")
        d1_future = loop.run_in_executor(executor, load_candles, asset_key, "1D")
        chart_df, signal_df, h1_df, d1_df = await asyncio.gather(chart_future, signal_future, h1_future, d1_future)

    if chart_df.empty and signal_df.empty:
        raise RuntimeError("No candle data returned.")
    if chart_df.empty:
        chart_df = signal_df.copy()
    if signal_df.empty:
        signal_df = chart_df.copy()

    mtf = multi_timeframe_confirmation({"30M": signal_df, "1H": h1_df, "1D": d1_df})
    tech = technical_score(signal_df, mtf=mtf)

    # News and AI are the slowest non-price calls; both are TTL-cached.
    news = await fetch_finnhub_news(asset_key)
    news_score = news_lexicon_score(news, asset_key)
    event_risk = economic_event_filter(news, asset_key)
    ai = await openai_sentiment(asset_key, news, tech, {"mtf": mtf, "event_risk": event_risk})

    fusion_pre = fusion_signal(tech, news_score, ai, mtf=mtf, event_risk=event_risk)
    tracker = update_signal_tracker(asset_key, fusion_pre["signal"], tech["price"], fusion_pre["confidence"])
    fusion = fusion_signal(tech, news_score, ai, mtf=mtf, event_risk=event_risk, tracker=tracker)

    chart = build_chart(chart_df, ASSETS[asset_key]["name"])

    daily_change = compute_daily_change(signal_df)

    result = {
        "asset_key": asset_key,
        "asset": ASSETS[asset_key],
        "data_source": "Capital.com",
        "tf": tf,
        "signal_tf": "30M",
        "tech": tech,
        "news": news,
        "news_score": round(news_score, 2),
        "ai": ai,
        "fusion": fusion,
        "chart": chart,
        "change": daily_change,
        "institutional": {"mtf": mtf, "event_risk": event_risk, "tracker": tracker},
        "updated": datetime.now().strftime("%H:%M:%S"),
    }
    _cache_set(RESPONSE_CACHE, response_key, result)
    return result

@app.get("/api/hit")
async def api_hit():
    with HIT_LOCK:
        hits = read_hit_count() + 1
        write_hit_count(hits)
    return {"hits": hits}


@app.get("/api/hits")
async def api_hits():
    return {"hits": read_hit_count()}


@app.get("/api/signal")
async def api_signal(asset: str = Query("WTI"), tf: str = Query("1M")):
    asset = asset.upper()
    tf = tf.upper()
    if asset not in ASSETS:
        return JSONResponse({"error": "Invalid asset"}, status_code=400)
    if tf not in INTERVALS:
        return JSONResponse({"error": "Invalid timeframe"}, status_code=400)
    try:
        return await process(asset, tf)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JSONResponse({"error": str(e)}, status_code=200)

HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Market Navigator - By Syed Abbas</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
:root{
  --bg:#f4f7fb;--surface:#ffffff;--surface2:#f9fbff;--ink:#101828;--muted:#667085;--line:#d9e2ef;
  --blue:#2563eb;--violet:#7c3aed;--green:#16a34a;--red:#e11d48;--amber:#f59e0b;--cyan:#0891b2;
  --shadow:0 18px 45px rgba(15,23,42,.09);--radius:22px;
}
*{box-sizing:border-box}html,body{margin:0;min-height:100%;background:var(--bg);color:var(--ink);font-family:Inter,Segoe UI,Arial,sans-serif}body{overflow:auto}.app{min-height:100vh;padding:18px;display:flex;flex-direction:column;gap:14px;background:radial-gradient(circle at 4% 0%,rgba(124,58,237,.12),transparent 30%),radial-gradient(circle at 98% 8%,rgba(37,99,235,.10),transparent 30%),linear-gradient(180deg,#fbfdff,#f4f7fb)}
.topbar{display:grid;grid-template-columns:minmax(280px,1fr) minmax(390px,.9fr) 128px;gap:14px;align-items:center}.brand{display:flex;align-items:center;gap:14px;min-width:0}.logo{width:54px;height:54px;border-radius:18px;display:grid;place-items:center;color:white;font-weight:1000;font-size:25px;background:linear-gradient(135deg,var(--violet),var(--blue));box-shadow:0 12px 30px rgba(124,58,237,.25)}.brand h1{margin:0;font-size:24px;letter-spacing:-.03em}.brand b{display:block;color:var(--violet);font-size:14px;margin-top:3px}.brand p{margin:5px 0 0;color:var(--muted);font-size:12px;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.livebox,.hitbox,.card,.control,.tickerCard{background:rgba(255,255,255,.88);border:1px solid var(--line);box-shadow:var(--shadow);border-radius:var(--radius);backdrop-filter:blur(14px)}.livebox{height:58px;display:flex;align-items:center;justify-content:center;gap:14px;font-size:13px;font-weight:900;color:#344054;white-space:nowrap;overflow:hidden}.dot{width:10px;height:10px;border-radius:50%;background:var(--green);display:inline-block;box-shadow:0 0 0 6px rgba(22,163,74,.12)}.hitbox{height:58px;display:flex;flex-direction:column;align-items:center;justify-content:center;background:linear-gradient(135deg,#f5f3ff,#eef5ff)}.hitbox span{color:var(--muted);font-size:10px;font-weight:1000;letter-spacing:.10em}.hitbox b{font-size:25px;color:var(--violet);line-height:1}.controls{display:grid;grid-template-columns:1.05fr 1.15fr .9fr .9fr;gap:14px}.control{height:84px;padding:15px 18px;min-width:0}.control label,.miniLabel{display:block;color:#667085;font-size:11px;font-weight:1000;letter-spacing:.08em;text-transform:uppercase;margin-bottom:8px}.controlLine{display:flex;align-items:flex-end;justify-content:space-between;gap:12px}.priceLine{display:flex;align-items:baseline;gap:10px;min-width:0}.mainPrice{font-size:25px;font-weight:1000;color:var(--blue);white-space:nowrap}.dailyChange{font-size:13px;font-weight:1000;color:var(--green);white-space:nowrap}.statusText{font-size:18px;font-weight:900}.subNote{margin-top:5px;color:var(--muted);font-size:12px;font-weight:700}select{width:100%;border:0;background:transparent;color:var(--ink);font-size:17px;font-weight:950;outline:0}option{background:#fff}.error{min-height:15px;color:var(--red);font-size:12px;font-weight:900;padding-left:4px}.workspace{display:grid;grid-template-columns:minmax(240px,280px) minmax(520px,1fr) minmax(270px,360px);gap:14px;align-items:stretch}.card{min-width:0;overflow:hidden}.signalCard{padding:18px;background:linear-gradient(160deg,#fff,#fff4f7);border-color:#ffd1dc;box-shadow:0 24px 55px rgba(225,29,72,.22),0 8px 20px rgba(15,23,42,.08)}.signalTop{display:flex;align-items:center;gap:12px}.assetBadge{width:46px;height:46px;border-radius:16px;display:grid;place-items:center;color:white;background:linear-gradient(135deg,var(--violet),var(--blue));font-size:22px}.assetTitle{font-size:17px;font-weight:1000}.assetSub{color:var(--muted);font-size:12px;font-weight:800;margin-top:3px}.timeMini{margin-left:auto;color:var(--muted);font-size:11px;font-weight:900}.signalWord{text-align:center;margin:28px 0 14px}.signalWord div{font-size:52px;line-height:.92;font-weight:1000;letter-spacing:-.05em;color:var(--red)}.signalWord span{display:inline-block;margin-top:9px;color:#be123c;background:#fff1f2;border:1px solid #fecdd3;border-radius:999px;padding:7px 12px;font-size:12px;font-weight:1000}.gauge{height:12px;border-radius:999px;background:#ffe4e6;margin:22px 0 18px;overflow:hidden}.gauge i{display:block;height:100%;width:0;background:linear-gradient(90deg,#fb7185,var(--red));border-radius:999px}.signalRows{background:rgba(255,255,255,.74);border:1px solid #ffe0e7;border-radius:18px;padding:8px}.signalRow{display:grid;grid-template-columns:24px 1fr;gap:8px;align-items:center;padding:11px 4px;border-bottom:1px solid #eef2f7}.signalRow:last-child{border-bottom:0}.signalRow label{color:var(--muted);font-size:11px;font-weight:1000;text-transform:uppercase}.signalRow b{grid-column:2;font-size:14px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.chartCard{min-height:520px}.head{height:52px;display:flex;align-items:center;justify-content:space-between;padding:0 16px;border-bottom:1px solid #eef2f7}.head b{font-size:14px;letter-spacing:.02em}.tools{display:flex;align-items:center;gap:8px}.toolBtn{border:1px solid var(--line);background:#fff;border-radius:12px;padding:8px 10px;font-size:12px;font-weight:900;color:#344054}.chartWrap{height:calc(100% - 52px);min-height:455px;padding:10px 12px 8px}#chart{height:100%;width:100%}.rightStack{display:grid;grid-template-rows:205px 1fr;gap:14px;min-width:0}.aiBody,.driverBody{padding:14px;height:calc(100% - 52px);overflow:hidden}.aiText{color:#344054;line-height:1.55;font-size:14px;font-weight:650;display:-webkit-box;-webkit-line-clamp:4;-webkit-box-orient:vertical;overflow:hidden}.aiBadges{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:14px}.aiBadge{border:1px solid var(--line);background:linear-gradient(135deg,#fff,#f7faff);border-radius:16px;padding:12px;text-align:center}.aiBadge span{display:block;color:var(--muted);font-size:11px;font-weight:1000;text-transform:uppercase}.aiBadge b{display:block;margin-top:5px;font-size:18px}.driverBody{display:flex;flex-direction:column;gap:10px}.driverItem{display:grid;grid-template-columns:48px minmax(120px,1fr) minmax(90px,34%);align-items:center;gap:10px;min-width:0}.driverBadge{border-radius:12px;text-align:center;padding:7px 0;font-size:13px;font-weight:1000;background:#ecfdf3;color:var(--green)}.driverBadge.neg{background:#fff1f2;color:var(--red)}.driverName{font-size:13px;font-weight:900;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.driverTrack{height:9px;background:#eef2f7;border-radius:999px;overflow:hidden}.driverFill{height:100%;background:linear-gradient(90deg,var(--green),#22c55e);border-radius:999px}.driverFill.neg{background:linear-gradient(90deg,#fb7185,#f97316)}.insights{display:grid;grid-template-columns:minmax(520px,1.35fr) minmax(330px,.75fr);gap:14px}.planBody{padding:14px}.planGrid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.kpi{min-height:86px;border:1px solid #e8eef7;background:linear-gradient(145deg,#fff,#f8fbff);border-radius:18px;padding:13px;display:flex;flex-direction:column;justify-content:center;min-width:0}.kpi span{color:var(--muted);font-size:11px;font-weight:1000;letter-spacing:.05em;text-transform:uppercase}.kpi b{font-size:18px;margin-top:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.green{color:var(--green)}.red{color:var(--red)}.amber{color:var(--amber)}.blue{color:var(--blue)}.newsBody{height:calc(100% - 52px);padding:12px;display:flex;flex-direction:column;gap:10px;overflow:hidden}.newsItem{display:grid;grid-template-columns:1fr auto;gap:10px;border:1px solid #eef2f7;border-radius:16px;background:#fff;padding:12px;min-width:0}.newsItem a{color:var(--ink);text-decoration:none;font-size:13px;font-weight:800;line-height:1.3;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}.newsItem small{color:var(--blue);font-size:11px;font-weight:1000}.ticker{display:grid;grid-template-columns:repeat(5,minmax(160px,1fr));gap:12px}.tickerCard{min-height:76px;padding:12px;display:grid;grid-template-columns:40px 1fr 70px;align-items:center;gap:10px;min-width:0}.tickerIcon{width:38px;height:38px;border-radius:14px;display:grid;place-items:center;color:white;font-size:20px;background:linear-gradient(135deg,var(--violet),var(--blue))}.tickerName{font-size:12px;font-weight:1000}.tickerPrice{font-size:12px;color:var(--muted);font-weight:800}.spark{height:28px;width:70px;border-radius:12px;background:linear-gradient(135deg,#ecfdf3,#fff)}.loading{opacity:.60;filter:saturate(.7)}
@media(max-width:1320px){.topbar,.controls,.workspace,.insights,.ticker{grid-template-columns:1fr}.chartCard{min-height:520px}.rightStack{grid-template-rows:auto}.aiBody,.driverBody,.newsBody{overflow:auto}.signalCard{min-height:auto}.planGrid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:680px){.app{padding:10px}.livebox{justify-content:flex-start;overflow:auto}.brand h1{font-size:20px}.mainPrice{font-size:21px}.signalWord div{font-size:44px}.planGrid,.aiBadges{grid-template-columns:1fr}.chartWrap{min-height:390px}.ticker{grid-template-columns:1fr}.toolBtn{padding:7px}.workspace{gap:10px}}
</style>
</head>
<body>
<div class="app">
  <header class="topbar">
    <div class="brand"><div class="logo">N</div><div><h1>AI Market Navigator</h1><b>By Syed Abbas</b><p>AI + Technicals + Market News Fusion Engine • Capital.com Prices</p></div></div>
    <div class="livebox"><span><i class="dot"></i> LIVE</span><span>↻ Auto-refresh: <b id="countdown">30</b>s</span><span>◷ Updated <b id="headerUpdated">--</b></span><span>Signal: 30M</span></div>
    <div class="hitbox"><span>TOTAL HITS</span><b id="hitCounter">0</b></div>
  </header>

  <section class="controls">
    <div class="control"><label>Asset</label><select id="assetSelect"></select></div>
    <div class="control"><label>Price</label><div class="controlLine"><div class="priceLine"><b class="mainPrice" id="controlPrice">--</b><span class="dailyChange" id="dailyChange">--</span></div></div></div>
    <div class="control"><label>Timeframe</label><select id="tfSelect"></select></div>
    <div class="control"><label>Status</label><div class="statusText" id="statusText">● Ready</div><div class="subNote">Protected responsive layout</div></div>
  </section>
  <div class="error" id="error"></div>

  <main class="workspace" id="dashboard">
    <section class="card signalCard" id="decisionCard">
      <div class="signalTop"><div class="assetBadge" id="assetIcon">📈</div><div><div class="assetTitle" id="assetName">USTEC 100</div><div class="assetSub"><span id="assetSymbol">US100</span> • <span id="assetPriceInline">--</span></div></div><span class="timeMini" id="signalTime">--</span></div>
      <div class="signalWord"><div id="signalText">WAIT</div><span id="signalSub">AI FUSION SIGNAL</span></div>
      <div class="gauge"><i id="confidenceBar"></i></div>
      <div class="signalRows">
        <div class="signalRow"><span>◎</span><label>Confidence</label><b id="meterValue">--%</b></div>
        <div class="signalRow"><span>↔</span><label>Range</label><b class="blue" id="entryVal">--</b></div>
        <div class="signalRow"><span>⊕</span><label>Target</label><b class="green" id="targetVal">--</b></div>
        <div class="signalRow"><span>⊗</span><label>Stop Loss</label><b class="red" id="stopVal">--</b></div>
      </div>
    </section>

    <section class="card chartCard">
      <div class="head"><b id="chartLabel">Price Action Chart</b><div class="tools"><button class="toolBtn">Indicators</button><button class="toolBtn">Fixed Zoom</button><button class="toolBtn">No Overlap</button></div></div>
      <div class="chartWrap"><div id="chart"></div></div>
    </section>

    <div class="rightStack">
      <section class="card"><div class="head"><b>AI Market View</b><span class="miniLabel">30M Engine</span></div><div class="aiBody"><div class="aiText" id="aiSummary">Loading market narrative...</div><div class="aiBadges"><div class="aiBadge"><span>Bias</span><b id="biasTag">--</b></div><div class="aiBadge"><span>Confidence</span><b><span id="confTag">--</span>%</b></div></div></div></section>
      <section class="card"><div class="head"><b>Signal Drivers</b><span class="miniLabel">Weighted Impact</span></div><div class="driverBody" id="drivers"></div></section>
    </div>
  </main>

  <section class="insights">
    <section class="card"><div class="head"><b>Market Intelligence KPIs</b><span class="miniLabel">Refreshed Live</span></div><div class="planBody"><div class="planGrid" id="planGrid"></div></div></section>
    <section class="card"><div class="head"><b>News Stream</b><span class="miniLabel" id="newsCount">0 items</span></div><div class="newsBody" id="news"></div></section>
  </section>

  <section class="ticker" id="tickerStrip"></section>
</div>
<script>
const ASSETS={GOLD:["🟡","Gold Spot","XAUUSD"],SILVER:["⚪","Silver","XAGUSD"],WTI:["🛢️","Crude Oil WTI","OIL_CRUDE"],BRENT:["🛢️","Brent Crude","OIL_BRENT"],BTC:["₿","Bitcoin","BTCUSD"],USTEC100:["📈","USTEC 100","US100"]};
const TFS=["1M","15M","30M","1H","1D"];let asset="USTEC100",tf="1H",busy=false,count=30;
function init(){Object.keys(ASSETS).forEach(k=>assetSelect.innerHTML+=`<option value="${k}">${ASSETS[k][0]} ${ASSETS[k][1]}</option>`);TFS.forEach(t=>tfSelect.innerHTML+=`<option value="${t}">${t}</option>`);assetSelect.value=asset;tfSelect.value=tf;assetSelect.onchange=()=>{asset=assetSelect.value;loadData(true)};tfSelect.onchange=()=>{tf=tfSelect.value;loadData(true)};updateTicker();updateHits();loadData(true);setInterval(()=>loadData(false),30000);setInterval(()=>{count=count<=1?30:count-1;countdown.innerText=count},1000)}
async function updateHits(){try{const r=await fetch(`/api/hit?_=${Date.now()}`);const d=await r.json();hitCounter.innerText=Number(d.hits||0).toLocaleString()}catch(e){hitCounter.innerText="--"}}
function colorFor(label){if(label==="BULLISH")return "#16a34a";if(label==="BEARISH")return "#e11d48";return "#f59e0b"}
function setLoading(v,manual=false){busy=v;dashboard.classList.toggle('loading',v&&manual);statusText.innerText=v?'● Loading':'● Ready'}
async function loadData(manual=false){if(busy&&!manual)return;setLoading(true,manual);let controller=new AbortController();let timer=setTimeout(()=>controller.abort(),manual?22000:12000);try{error.innerText="";count=30;const r=await fetch(`/api/signal?asset=${asset}&tf=${tf}&_=${Date.now()}`,{signal:controller.signal});const d=await r.json();if(d.error)throw new Error(d.error);render(d)}catch(e){if(manual)error.innerText="Error: "+e.message}finally{clearTimeout(timer);setLoading(false,manual)}}
function num(x){let n=Number(x||0);return isFinite(n)?n:0}
function fmt(x,maxDp=3){let n=Number(String(x??'').replace(/,/g,''));if(!isFinite(n))return String(x??'--');return n.toLocaleString(undefined,{minimumFractionDigits:3,maximumFractionDigits:3})}
function fmtZone(z){return String(z??'--').split(' - ').map(v=>fmt(v)).join(' - ')}
function calcChange(d){try{if(d.change&&isFinite(Number(d.change.percent))){return {p:Number(d.change.percent||0),v:Number(d.change.value||0)}}let closes=(d.chart.data||[])[0].close||[];let first=Number(closes[Math.max(0,closes.length-25)]||closes[0]);let last=Number(closes[closes.length-1]);if(!isFinite(first)||!isFinite(last)||first===0)return {p:0,v:0};let pct=(last-first)/first*100;if(Math.abs(pct)>20){first=Number(closes[closes.length-2]||first);pct=(last-first)/first*100}return {p:pct,v:last-first}}catch(e){return {p:0,v:0}}}
function updateTicker(){tickerStrip.innerHTML=Object.keys(ASSETS).slice(0,5).map(k=>`<div class="tickerCard"><div class="tickerIcon">${ASSETS[k][0]}</div><div><div class="tickerName">${ASSETS[k][1]}</div><div class="tickerPrice" id="ticker_${k}">Loading</div></div><div class="spark"></div></div>`).join('')}
function render(d){const label=d.fusion.label||'NEUTRAL',sig=d.fusion.signal||'HOLD / WAIT',conf=num(d.fusion.confidence),fusion=num(d.fusion.fusion),col=colorFor(label),isSell=label==='BEARISH',isHold=label==='NEUTRAL';
const ch=calcChange(d);const chTxt=`${ch.v>=0?'+':''}${fmt(ch.v)} (${ch.p>=0?'+':''}${ch.p.toFixed(2)}%)`;dailyChange.innerText=chTxt;dailyChange.style.color=ch.p>=0?'#16a34a':'#e11d48';
const entryZone=d.tech.entry_zone||(d.tech.support&&d.tech.resistance?`${Math.min(num(d.tech.support),num(d.tech.resistance))} - ${Math.max(num(d.tech.support),num(d.tech.resistance))}`:d.tech.entry);
assetIcon.innerText=d.asset.icon;assetName.innerText=(d.asset.name||'').toUpperCase();assetSymbol.innerText=ASSETS[d.asset_key]?.[2]||d.asset.epic||'';controlPrice.innerText=fmt(d.tech.price);assetPriceInline.innerText=fmt(d.tech.price);headerUpdated.innerText=d.updated;signalTime.innerText=d.updated;signalText.innerText=sig.replace(' SIGNAL','').replace(' / WAIT','');signalText.style.color=col;signalSub.innerText=label==='NEUTRAL'?'WAIT FOR CONFIRMATION':(label==='BEARISH'?'STRONG BEARISH':'STRONG BULLISH');meterValue.innerText=Math.round(conf)+'% CONFIDENCE';confidenceBar.style.width=Math.max(5,Math.min(100,conf))+'%';confidenceBar.style.background=isSell?'linear-gradient(90deg,#fb7185,#e11d48)':isHold?'linear-gradient(90deg,#fde68a,#f59e0b)':'linear-gradient(90deg,#86efac,#16a34a)';
entryVal.innerText=fmtZone(entryZone);targetVal.innerText=fmt(d.tech.target);stopVal.innerText=fmt(d.tech.stop);targetVal.className=isSell?'red':'green';stopVal.className=isSell?'green':'red';decisionCard.style.background=isSell?'linear-gradient(155deg,#ffffff 0%,#ffe4ea 58%,#fecdd3 100%)':isHold?'linear-gradient(155deg,#ffffff 0%,#fff7d6 100%)':'linear-gradient(155deg,#ffffff 0%,#dcfce7 58%,#bbf7d0 100%)';decisionCard.style.borderColor=isSell?'#fb7185':isHold?'#fbbf24':'#22c55e';decisionCard.style.boxShadow=isSell?'0 26px 65px rgba(225,29,72,.30), 0 0 0 1px rgba(225,29,72,.12)':isHold?'0 24px 55px rgba(245,158,11,.22), 0 0 0 1px rgba(245,158,11,.10)':'0 26px 65px rgba(22,163,74,.30), 0 0 0 1px rgba(22,163,74,.12)';
aiSummary.innerText=d.ai.summary||'No AI summary returned.';biasTag.innerText=label;biasTag.style.color=col;confTag.innerText=Math.round(conf);chartLabel.innerText=`${d.asset.name} • ${d.tf} • Signal Fixed ${d.signal_tf||'30M'} • Candles + EMA 9/21/50`;
let layout=d.chart.layout||{};layout.autosize=true;layout.height=null;layout.margin={l:56,r:22,t:12,b:60};layout.paper_bgcolor='rgba(0,0,0,0)';layout.plot_bgcolor='#ffffff';layout.font={color:'#344054',size:10};layout.legend={orientation:'h',y:1.04,x:0,font:{size:10,color:'#344054'}};layout.dragmode=false;layout.xaxis={...(layout.xaxis||{}),type:'date',rangeslider:{visible:false},gridcolor:'rgba(15,23,42,.08)',linecolor:'#e5e7eb',automargin:true,tickangle:0,tickfont:{size:9,color:'#475467'},ticklabelstandoff:8,fixedrange:true,rangebreaks:['1M','15M','30M','1H','1D'].includes(d.tf)?[{bounds:['sat','mon']}]:[]};layout.yaxis={...(layout.yaxis||{}),gridcolor:'rgba(15,23,42,.08)',linecolor:'#e5e7eb',automargin:true,zeroline:false,fixedrange:true};Plotly.react('chart',d.chart.data,layout,{displayModeBar:false,responsive:true,scrollZoom:false,editable:false,doubleClick:false});setTimeout(()=>Plotly.Plots.resize('chart'),150);
const rows=[['Technical Analysis',Math.round(num(d.fusion.tech_percent)-50)],['Fusion Momentum',Math.round(fusion)],['AI Sentiment',Math.round(num(d.fusion.ai_percent)-50)],['News Sentiment',Math.round(num(d.fusion.news_percent)-50)],['Risk Adjustment',-Math.round(num(d.institutional?.event_risk?.risk_penalty)||0)]];
drivers.innerHTML=rows.map(r=>{const v=Math.trunc(r[1]);const pctWidth=Math.max(6,Math.min(100,Math.abs(v)/70*100));const sign=v>0?'+':'';const neg=v<0;return `<div class="driverItem"><div class="driverBadge ${neg?'neg':''}">${sign}${v}</div><div class="driverName">${r[0]}</div><div class="driverTrack"><div class="driverFill ${neg?'neg':''}" style="width:${pctWidth}%"></div></div></div>`}).join('');
newsCount.innerText=(d.news||[]).length+' items';news.innerHTML=(d.news||[]).slice(0,4).map(n=>`<div class="newsItem"><div><small>${n.source||'News'}</small><br><a href="${n.url||'#'}" target="_blank">${n.headline||''}</a></div><span>↗</span></div>`).join('')||'<div class="newsItem"><a>No matching news returned.</a></div>';
const regime=(d.tech.regime?.regime||'NORMAL'),adx=num(d.tech.regime?.adx),atrPct=num(d.tech.regime?.atr_pct),rsi=num(d.tech.rsi),prob=Math.round(num(d.fusion.probability)*100),trendStrength=Math.min(10,Math.max(1,Math.round((Math.abs(fusion)/10 + adx/10)*10)/10));const riskLevel=(isHold||atrPct>1.2||num(d.institutional?.event_risk?.risk_penalty)>6)?'ELEVATED':(atrPct>.65?'MEDIUM':'LOW');const volStatus=atrPct>1.2?'HIGH':(atrPct>.65?'MEDIUM':'CALM');const volumeContext=num(d.tech.price)>=num(d.tech.poc)?'ABOVE POC':'BELOW POC';const mtfAlign=num(d.institutional?.mtf?.alignment);
planGrid.innerHTML=[['Trend Strength',fmt(trendStrength)+' / 10',trendStrength>=7?'green':trendStrength>=4?'amber':'red'],['Market Regime',regime,regime==='TRENDING'?'green':regime==='RANGING'?'amber':'green'],['Volatility',volStatus+' '+fmt(atrPct)+'% ',volStatus==='HIGH'?'red':volStatus==='MEDIUM'?'amber':'green'],['Volume Profile',volumeContext,'blue'],['Risk Level',riskLevel,riskLevel==='LOW'?'green':riskLevel==='MEDIUM'?'amber':'red'],['Probability',prob+'%','green'],['MTF Alignment',mtfAlign>0?'ALIGNED':mtfAlign<0?'MIXED':'NEUTRAL',mtfAlign>0?'green':mtfAlign<0?'red':'amber'],['RSI Zone',rsi>=70?'OVERBOUGHT':rsi<=30?'OVERSOLD':'NORMAL',rsi>=70||rsi<=30?'amber':'green']].map(x=>`<div class="kpi"><span>${x[0]}</span><b class="${x[2]}">${x[1]}</b></div>`).join('');
try{document.getElementById('ticker_'+d.asset_key).innerText=fmt(d.tech.price)+'  '+chTxt}catch(e){}
}
window.addEventListener('resize',()=>{try{Plotly.Plots.resize('chart')}catch(e){}});init();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(HTML)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8001)),
        reload=False,
    )
