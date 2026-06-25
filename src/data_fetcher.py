"""
株価データ取得モジュール
- 1分足データ（yfinance、最大7日）
- 日足データ（yfinance、長期）
- キャッシュは CSV 形式（追加依存なし）
- レート制限対策込み
"""
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cache")
os.makedirs(DATA_DIR, exist_ok=True)

# yfinance レート制限対策
REQUEST_DELAY = 1.5  # 通常リクエスト間隔（秒）
MAX_RETRIES   = 3
RETRY_WAIT    = 60   # 429エラー時の待機（秒）

# Japan market hours (JST = UTC+9)
MARKET_OPEN_MORNING    = (9,  0)
MARKET_CLOSE_MORNING   = (11, 30)
MARKET_OPEN_AFTERNOON  = (12, 30)
MARKET_CLOSE_AFTERNOON = (15, 30)


def is_market_open() -> bool:
    """現在日本株市場が開いているか"""
    now_jst = datetime.utcnow() + timedelta(hours=9)
    if now_jst.weekday() >= 5:
        return False
    t = now_jst.hour * 60 + now_jst.minute
    mo = MARKET_OPEN_MORNING[0]    * 60 + MARKET_OPEN_MORNING[1]
    mc = MARKET_CLOSE_MORNING[0]   * 60 + MARKET_CLOSE_MORNING[1]
    ao = MARKET_OPEN_AFTERNOON[0]  * 60 + MARKET_OPEN_AFTERNOON[1]
    ac = MARKET_CLOSE_AFTERNOON[0] * 60 + MARKET_CLOSE_AFTERNOON[1]
    return (mo <= t <= mc) or (ao <= t <= ac)


def get_market_session() -> str:
    """
    現在のセッションを返す
    Returns: 'pre_market' | 'morning' | 'lunch' | 'afternoon' | 'post_market'
    """
    now_jst = datetime.utcnow() + timedelta(hours=9)
    if now_jst.weekday() >= 5:
        return "post_market"
    t = now_jst.hour * 60 + now_jst.minute
    if t < 9 * 60:
        return "pre_market"
    elif t <= 11 * 60 + 30:
        return "morning"
    elif t < 12 * 60 + 30:
        return "lunch"
    elif t <= 15 * 60 + 30:
        return "afternoon"
    else:
        return "post_market"


def _cache_path(ticker: str, interval: str) -> str:
    """CSVキャッシュのパスを返す（pyarrow 不要）"""
    safe = ticker.replace(".", "_")
    return os.path.join(DATA_DIR, f"{safe}_{interval}.csv")


def _read_cache(path: str, max_age: timedelta) -> Optional[pd.DataFrame]:
    """CSVキャッシュを読み込む。期限切れ・存在しない場合は None"""
    if not os.path.exists(path):
        return None
    if datetime.now() - datetime.fromtimestamp(os.path.getmtime(path)) > max_age:
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def _write_cache(df: pd.DataFrame, path: str) -> None:
    """DataFrameをCSVキャッシュに書き込む"""
    try:
        df.to_csv(path)
    except Exception as e:
        logger.warning(f"キャッシュ書き込み失敗: {e}")


def fetch_1min_data(
    ticker: str,
    period: str = "7d",
    use_cache: bool = True,
    cache_minutes: int = 5,
) -> Optional[pd.DataFrame]:
    """
    1分足データを取得する（最大7日間）

    Parameters
    ----------
    ticker        : 例 "7203.T"
    period        : "1d" | "2d" | "5d" | "7d"
    use_cache     : キャッシュを使うか
    cache_minutes : キャッシュ有効分数

    Returns
    -------
    DataFrame with Open/High/Low/Close/Volume（JST index）、取得失敗時は None
    """
    path = _cache_path(ticker, "1m")
    if use_cache:
        cached = _read_cache(path, timedelta(minutes=cache_minutes))
        if cached is not None:
            return cached

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

            # タイムゾーンを JST に変換
            if df.index.tz is not None:
                df.index = df.index.tz_convert("Asia/Tokyo")
            else:
                df.index = df.index.tz_localize("UTC").tz_convert("Asia/Tokyo")

            # 日本市場時間帯のみ抽出（9:00–11:30、12:30–15:30）
            h, m = df.index.hour, df.index.minute
            mask = (
                (h == 9) |
                (h == 10) |
                ((h == 11) & (m <= 30)) |
                ((h == 12) & (m >= 30)) |
                (h == 13) |
                (h == 14) |
                ((h == 15) & (m <= 30))
            )
            df = df[mask]

            if use_cache:
                _write_cache(df, path)

            time.sleep(REQUEST_DELAY)
            return df

        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = RETRY_WAIT * (attempt + 1)
                logger.warning(f"{ticker}: レート制限。{wait}秒待機...")
                time.sleep(wait)
            else:
                logger.error(f"{ticker} 1分足取得失敗（試行{attempt + 1}）: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(REQUEST_DELAY * 3)

    return None


def fetch_daily_data(
    ticker: str,
    start: str = "1991-01-01",
    use_cache: bool = True,
    cache_hours: int = 12,
) -> Optional[pd.DataFrame]:
    """
    日足データを取得する（バブル崩壊後から）

    Parameters
    ----------
    ticker      : 例 "7203.T"
    start       : 取得開始日（バブル崩壊: "1991-01-01"）
    use_cache   : キャッシュを使うか
    cache_hours : キャッシュ有効時間

    Returns
    -------
    DataFrame with Open/High/Low/Close/Volume、取得失敗時は None
    """
    path = _cache_path(ticker, "1d")
    if use_cache:
        cached = _read_cache(path, timedelta(hours=cache_hours))
        if cached is not None:
            return cached

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

            if use_cache:
                _write_cache(df, path)

            time.sleep(REQUEST_DELAY)
            return df

        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = RETRY_WAIT * (attempt + 1)
                logger.warning(f"{ticker}: レート制限。{wait}秒待機...")
                time.sleep(wait)
            else:
                logger.error(f"{ticker} 日足取得失敗（試行{attempt + 1}）: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(REQUEST_DELAY * 3)

    return None
