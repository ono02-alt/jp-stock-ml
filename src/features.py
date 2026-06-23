"""
テクニカル指標の計算とML特徴量エンジニアリング
1分足OHLCVデータから特徴量を生成する
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    logger.warning("pandas_ta が見つかりません。手動計算にフォールバックします")


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _bollinger_bands(series: pd.Series, period=20, std_dev=2):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower


def _vwap(df: pd.DataFrame) -> pd.Series:
    """セッション内VWAPを計算"""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vwap = (typical * df["Volume"]).cumsum() / df["Volume"].cumsum().replace(0, np.nan)
    return vwap


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    1分足OHLCVデータからML特徴量を計算する

    Parameters
    ----------
    df : DataFrame with columns Open, High, Low, Close, Volume

    Returns
    -------
    DataFrame with feature columns added
    """
    df = df.copy()
    close = df["Close"]
    volume = df["Volume"]
    high = df["High"]
    low = df["Low"]
    open_ = df["Open"]

    # === ローソク足の特性 ===
    df["body"] = close - open_
    df["upper_shadow"] = high - pd.concat([close, open_], axis=1).max(axis=1)
    df["lower_shadow"] = pd.concat([close, open_], axis=1).min(axis=1) - low
    df["body_ratio"] = df["body"] / (high - low + 1e-10)
    df["is_bullish"] = (close > open_).astype(int)

    # === リターン ===
    df["ret_1"] = close.pct_change(1)
    df["ret_3"] = close.pct_change(3)
    df["ret_5"] = close.pct_change(5)
    df["ret_10"] = close.pct_change(10)

    # === 移動平均 ===
    for p in [5, 10, 20, 25, 50]:
        df[f"ma_{p}"] = close.rolling(p).mean()
        df[f"ma_ratio_{p}"] = close / (df[f"ma_{p}"] + 1e-10) - 1

    # === EMA ===
    for p in [5, 12, 26]:
        df[f"ema_{p}"] = _ema(close, p)

    # === VWAP ===
    df["vwap"] = _vwap(df)
    df["vwap_ratio"] = close / (df["vwap"] + 1e-10) - 1

    # === RSI ===
    df["rsi_14"] = _rsi(close, 14)
    df["rsi_9"] = _rsi(close, 9)

    # === MACD ===
    macd_line, signal_line, hist = _macd(close)
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = hist
    df["macd_cross"] = (
        (macd_line > signal_line).astype(int) -
        (macd_line.shift(1) > signal_line.shift(1)).astype(int)
    )

    # === ボリンジャーバンド ===
    bb_upper, bb_mid, bb_lower = _bollinger_bands(close, 20, 2)
    df["bb_upper"] = bb_upper
    df["bb_lower"] = bb_lower
    df["bb_mid"] = bb_mid
    df["bb_width"] = (bb_upper - bb_lower) / (bb_mid + 1e-10)
    df["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)

    # === ATR ===
    df["atr"] = _atr(df, 14)
    df["atr_ratio"] = df["atr"] / (close + 1e-10)

    # === 出来高 ===
    df["vol_ma_5"] = volume.rolling(5).mean()
    df["vol_ma_20"] = volume.rolling(20).mean()
    df["vol_ratio_5"] = volume / (df["vol_ma_5"] + 1e-10)
    df["vol_ratio_20"] = volume / (df["vol_ma_20"] + 1e-10)
    df["vol_price"] = volume * close  # 売買代金

    # === 価格レンジ ===
    df["high_low_ratio"] = (high - low) / (close + 1e-10)

    # === 時間特徴量（日本市場: 9:00-11:30, 12:30-15:30）===
    if hasattr(df.index, "hour"):
        df["hour"] = df.index.hour
        df["minute"] = df.index.minute
        df["time_in_session"] = (df["hour"] * 60 + df["minute"]) / 900.0
        # 前場・後場フラグ
        df["is_morning"] = (
            ((df["hour"] == 9) | (df["hour"] == 10) | ((df["hour"] == 11) & (df["minute"] <= 30)))
        ).astype(int)
        df["is_afternoon"] = (
            ((df["hour"] == 12) & (df["minute"] >= 30)) |
            (df["hour"] == 13) | (df["hour"] == 14) |
            ((df["hour"] == 15) & (df["minute"] <= 30))
        ).astype(int)
        # 寄り・引け付近フラグ
        df["near_open"] = (
            (df["hour"] == 9) & (df["minute"] <= 15)
        ).astype(int)
        df["near_close"] = (
            (df["hour"] == 15) & (df["minute"] >= 15)
        ).astype(int)

    # === ターゲット: 次の1分後が陽線か ===
    next_close = close.shift(-1)
    next_open = open_.shift(-1)
    df["target"] = (next_close > next_open).astype(int)

    return df


def get_feature_columns() -> list:
    """ML学習に使用する特徴量カラム名のリスト"""
    return [
        "body", "upper_shadow", "lower_shadow", "body_ratio", "is_bullish",
        "ret_1", "ret_3", "ret_5", "ret_10",
        "ma_ratio_5", "ma_ratio_10", "ma_ratio_20", "ma_ratio_25", "ma_ratio_50",
        "vwap_ratio",
        "rsi_14", "rsi_9",
        "macd", "macd_signal", "macd_hist", "macd_cross",
        "bb_width", "bb_position",
        "atr_ratio",
        "vol_ratio_5", "vol_ratio_20",
        "high_low_ratio",
      ]
