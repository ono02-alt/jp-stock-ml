"""
株価データ取得モジュール
- 1分足データ（yfinance、最大7日）
- 日足データ（yfinance、長期）
- レート制限対策込み
"""
import os
import time
import logging
import json
import hashlib
from datetime import datetime, timedelta, date
from typing import Optional, List, Dict
import pandas as pd
import numpy as np
import yfinance as yf

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cache")
os.makedirs(DATA_DIR, exist_ok=True)

# yfinance レート制限対策
REQUEST_DELAY = 1.5   # 通常リクエスト間隔（秒）
BATCH_DELAY = 5.0     # バッチ間隔（秒）
MAX_RETRIES = 3
RETRY_WAIT = 60       # 429エラー時の待機（秒）

# Japan market hours (JST = UTC+9)
MARKET_OPEN_MORNING = (9, 0)
MARKET_CLOSE_MORNING = (11, 30)
MARKET_OPEN_AFTERNOON = (12, 30)
MARKET_CLOSE_AFTERNOON = (15, 30)


def is_market_open() -> bool:
    """現在日本株市場が開いているか"""
    now_jst = datetime.utcnow() + timedelta(hours=9)
    # 土日は休場
    if now_jst.weekday() >= 5:
        return False
    h, m = now_jst.hour, now_jst.minute
    time_val = h * 60 + m
    morning_open = MARKET_OPEN_MORNING[0] * 60 + MARKET_OPEN_MORNING[1]
    morning_close = MARKET_CLOSE_MORNING[0] * 60 + MARKET_CLOSE_MORNING[1]
    afternoon_open = MARKET_OPEN_AFTERNOON[0] * 60 + MARKET_OPEN_AFTERNOON[1]
    afternoon_close = MARKET_CLOSE_AFTERNOON[0] * 60 + MARKET_CLOSE_AFTERNOON[1]
    return (morning_open <= time_val <= morning_close) or (afternoon_open <= time_val <= afternoon_close)


def get_market_session() -> str:
    """
    現在のセッションを返す
    Returns: 'pre_market' | 'morning' | 'lunch' | 'afternoon' | 'post_market'
    """
    now_jst = datetime.utcnow() + timedelta(hours=9)
    if now_jst.weekday() >= 5:
        return "post_market"
    h, m = now_jst.hour, now_jst.minute
    time_val = h * 60 + m
    if time_val < 9 * 60:
        return "pre_market"
    elif time_val <= 11 * 60 + 30:
        return "morning"
    elif time_val < 12 * 60 + 30:
        return "lunch"
    elif time_val <= 15 * 60 + 30:
        return "afternoon"
    else:
        return "post_market"


def _cache_key(ticker: str, interval: str, period: str) -> str:
    key = f"{ticker}_{interval}_{period}"
    return hashlib.md5(key.encode()).hexdigest()


def _get_cache_path(ticker: str, interval: str) -> str:
    safe_ticker = ticker.replace(".", "_")
    return os.path.join(DATA_DIR, f"{safe_ticker}_{interval}.parquet")


def fetch_1min_data(
    ticker: str,
    period: str = "7d",
    use_cache: bool = True,
    cache_minutes: int = 5
) -> Optional[pd.DataFrame]:
    """
    1分足データを取得する（最大7日間）

    Parameters
    ----------
    ticker : str  例: "7203.T"
    period : str  "1d", "2d", "5d", "7d"
    use_cache : bool  キャッシュ使用
    cache_minutes : int  キャッシュ有効分数

    Returns
    -------
    DataFrame with Open, High, Low, Close, Volume（JST timezone）
    """
    cache_path = _get_cache_path(ticker, "1m")

    if use_cache and os.path.exists(cache_path):
        mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
        if datetime.now() - mtime < timedelta(minutes=cache_minutes):
            try:
                df = pd.read_parquet(cache_path)
                return df
            except Exception:
                pass

    for attempt in range(MAX_RETRIES):
        try:
            df = yf.download(
                ticker,
                period=period,
                interval="1m",
                auto_adjust=True,
                multi_level_index=False,
                progress=False,
            )
            if df is None or df.empty:
                logger.warning(f"{ticker}: 1分足データが空です")
                return None

            # タイムゾーンをJSTに変換
            if df.index.tz is not None:
                df.index = df.index.tz_convert("Asia/Tokyo")
            else:
                df.index = df.index.tz_localize("UTC").tz_convert("Asia/Tokyo")

            # 日本市場時間のみ（9:00-15:30）
            df = df[
                ((df.index.hour == 9) |
                 (df.index.hour == 10) |
                 ((df.index.hour == 11) & (df.index.minute <= 30)) |
                 ((df.index.hour == 12) & (df.index.minute >= 30)) |
                 (df.index.hour == 13) |
                 (df.index.hour == 14) |
                 ((df.index.hour == 15) & (df.index.minute <= 30)))
            ]

            df.to_parquet(cache_path)
            time.sleep(REQUEST_DELAY)
            return df

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate" in err_str.lower():
                wait = RETRY_WAIT * (attempt + 1)
                logger.warning(f"{ticker}: レート制限。{wait}秒待機...")
                time.sleep(wait)
            else:
                logger.error(f"{ticker} 1分足取得失敗（試行{attempt+1}）: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(REQUEST_DELAY * 3)

    return None


def fetch_daily_data(
    ticker: str,
    start: str = "1991-01-01",
    use_cache: bool = True,
    cache_hours: int = 12
) -> Optional[pd.DataFrame]:
    """
    日足データを取得する（バブル崩壊後から）

    Parameters
    ----------
    ticker : str  例: "7203.T"
    start : str  取得開始日（バブル崩壊: 1991-01-01）
    use_cache : bool
    cache_hours : int  キャッシュ有効時間

    Returns
    -------
    DataFrame with Open, High, Low, Close, Volume
    """
    cache_path = _get_cache_path(ticker, "1d")

    if use_cache and os.path.exists(cache_path):
        mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
        if datetime.now() - mtime < timedelta(hours=cache_hours):
            try:
                return pd.read_parquet(cache_path)
            except Exception:
                pass

    for attempt in range(MAX_RETRIES):
        try:
            df = yf.download(
                ticker,
                start=start,
                interval="1d",
                auto_adjust=True,
                multi_level_index=False,
                progress=False,
            )
            if df is None or df.empty:
                return None

            df.to_parquet(cache_path)
            time.sleep(REQUEST_DELAY)
            return df

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate" in err_str.lower():
                wait = RETRY_WAIT * (attempt + 1)
                logger.warning(f"{ticker}: レート制限。{wait}秒待機...")
                time.sleep(wait)
            else:
                logger.error(f"{ticker} 日足取得失敗（試行{attempt+1}）: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(REQUEST_DELAY * 3)

    return None


def fetch_batch_1min(
    tickers: List[str],
    period: str = "7d",
    batch_size: int = 5
) -> Dict[str, pd.DataFrame]:
    """
    複数銘柄の1分足データを一括取得（レート制限対策付き）
    """
    results = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i: i + batch_size]
        for ticker in batch:
            df = fetch_1min_data(ticker, period=period)
            if df is not None and not df.empty:
                results[ticker] = df
        logger.info(f"1分足取得進捗: {min(i+batch_size, len(tickers))}/{len(tickers)}")
        time.sleep(BATCH_DELAY)
    return results


def get_latest_price(ticker: str) -> Optional[dict]:
    """直近の株価情報を取得（yfinance 1.x対応）"""
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info
        # yfinance 1.x では last_price / last_volume が利用可能
        price = getattr(info, "last_price", None)
        volume = getattr(info, "last_volume", None)
        # フォールバック: 属性が取れない場合はhistoryから取得
        if price is None:
            hist = t.history(period="1d", interval="1m")
            if hist is not None and not hist.empty:
                price = float(hist["Close"].iloc[-1])
                volume = int(hist["Volume"].iloc[-1])
        return {
            "ticker": ticker,
            "price": float(price) if price is not None else None,
            "volume": int(volume) if volume is not None else None,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"{ticker} 直近価格取得失敗: {e}")
        return None
