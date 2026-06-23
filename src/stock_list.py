"""
JPX上場銘柄リストを動的に取得するモジュール
日本取引所グループ公式XLSから全上場銘柄を取得し、
デイトレ・スイング可能な銘柄（低位株除外）をフィルタリングする
"""
import os
import time
import logging
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from io import BytesIO
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# JPX公式銘柄一覧URL
JPX_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"

# 低位株除外の最低株価閾値（円）
MIN_PRICE = 300
# デイトレ可能な最低出来高（株/日）
MIN_VOLUME_DAYTRADE = 500_000
# スイング可能な最低出来高（株/日）
MIN_VOLUME_SWING = 100_000

CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cache", "stock_list.csv")
CACHE_EXPIRY_HOURS = 12


def fetch_jpx_stock_list(use_cache: bool = True) -> pd.DataFrame:
    """JPX公式XLSから全上場銘柄リストを取得する"""
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

    # キャッシュ確認
    if use_cache and os.path.exists(CACHE_PATH):
        mtime = datetime.fromtimestamp(os.path.getmtime(CACHE_PATH))
        if datetime.now() - mtime < timedelta(hours=CACHE_EXPIRY_HOURS):
            logger.info("キャッシュから銘柄リストを読み込みます")
            return pd.read_csv(CACHE_PATH, dtype={"コード": str})

    logger.info("JPX公式サイトから銘柄リストを取得中...")
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; stock-screener/1.0)"
    }

    for attempt in range(3):
        try:
            resp = requests.get(JPX_URL, headers=headers, timeout=30)
            resp.raise_for_status()
            df = pd.read_excel(BytesIO(resp.content), engine="xlrd", dtype={"コード": str})
            break
        except Exception as e:
            logger.warning(f"JPX取得試行{attempt+1}失敗: {e}")
            if attempt < 2:
                time.sleep(10 * (attempt + 1))
            else:
                # フォールバック: キャッシュがあれば使用
                if os.path.exists(CACHE_PATH):
                    logger.warning("フォールバック: 古いキャッシュを使用")
                    return pd.read_csv(CACHE_PATH, dtype={"コード": str})
                raise

    # 列名の正規化
    df.columns = df.columns.str.strip()

    # コードを4桁文字列に正規化
    if "コード" in df.columns:
        df["コード"] = df["コード"].astype(str).str.zfill(4).str[:4]

    # キャッシュ保存
    df.to_csv(CACHE_PATH, index=False)
    logger.info(f"銘柄リスト取得完了: {len(df)}銘柄")
    return df


def get_tradeable_stocks(
    mode: str = "daytrade",
    sample_days: int = 5,
    max_stocks: int = 200
) -> pd.DataFrame:
    """
    デイトレ・スイング可能な銘柄を絞り込む

    Parameters
    ----------
    mode : "daytrade" | "swing"
    sample_days : 出来高サンプリング日数（yfinance上限: 7日）
    max_stocks : 最大返却銘柄数

    Returns
    -------
    DataFrame with columns: code, ticker, name, price, avg_volume, market
    """
    min_vol = MIN_VOLUME_DAYTRADE if mode == "daytrade" else MIN_VOLUME_SWING

    jpx_df = fetch_jpx_stock_list()

    # 市場区分を確認（プライム・スタンダード・グロース等）
    market_col = None
    for col in ["市場・商品区分", "市場区分", "部門"]:
        if col in jpx_df.columns:
            market_col = col
            break

    # 内国普通株式のみ（ETF・REITを除外）
    kind_col = None
    for col in ["種別", "規模区分"]:
        if col in jpx_df.columns:
            kind_col = col
            break

    # ティッカー作成（コード + .T）
    codes = jpx_df["コード"].dropna().unique().tolist()
    # 4桁の純粋な数字コードのみ
    codes = [c for c in codes if c.isdigit() and len(c) == 4]
    tickers = [f"{c}.T" for c in codes]

    logger.info(f"全上場銘柄数: {len(tickers)}")

    # バッチで株価・出来高取得（yfinance レート制限対策）
    results = []
    batch_size = 20
    sleep_between = 3.0

    for i in range(0, min(len(tickers), 2000), batch_size):
        batch = tickers[i: i + batch_size]
        try:
            data = yf.download(
                batch,
                period=f"{sample_days}d",
                interval="1d",
                auto_adjust=True,
                multi_level_index=False,
                progress=False,
                threads=False,
            )

            if data.empty:
                continue

            # マルチティッカーの場合は MultiIndex
            if isinstance(data.columns, pd.MultiIndex):
                for ticker in batch:
                    try:
                        if ticker not in data["Close"].columns:
                            continue
                        close = data["Close"][ticker].dropna()
                        vol = data["Volume"][ticker].dropna()
                        if len(close) == 0:
                            continue
                        avg_price = close.mean()
                        avg_vol = vol.mean()
                        if avg_price >= MIN_PRICE and avg_vol >= min_vol:
                            code = ticker.replace(".T", "")
                            name_row = jpx_df[jpx_df["コード"] == code]
                            name = name_row["銘柄名"].values[0] if len(name_row) > 0 and "銘柄名" in name_row.columns else ""
                            market = name_row[market_col].values[0] if market_col and len(name_row) > 0 else ""
                            results.append({
                                "code": code,
                                "ticker": ticker,
                                "name": name,
                                "price": round(avg_price, 0),
                                "avg_volume": int(avg_vol),
                                "market": market,
                            })
                    except Exception:
                        pass
            else:
                # 単一銘柄
                ticker = batch[0]
                close = data["Close"].dropna()
                vol = data["Volume"].dropna()
                if len(close) == 0:
                    continue
                avg_price = close.mean()
                avg_vol = vol.mean()
                if avg_price >= MIN_PRICE and avg_vol >= min_vol:
                    code = ticker.replace(".T", "")
                    name_row = jpx_df[jpx_df["コード"] == code]
                    name = name_row["銘柄名"].values[0] if len(name_row) > 0 and "銘柄名" in name_row.columns else ""
                    market = name_row[market_col].values[0] if market_col and len(name_row) > 0 else ""
                    results.append({
                        "code": code,
                        "ticker": ticker,
                        "name": name,
                        "price": round(avg_price, 0),
                        "avg_volume": int(avg_vol),
                        "market": market,
                    })

        except Exception as e:
            logger.warning(f"バッチ取得失敗 {i}-{i+batch_size}: {e}")

        time.sleep(sleep_between)

        if len(results) >= max_stocks:
            break

    result_df = pd.DataFrame(results)
    if result_df.empty:
        return result_df

    # 出来高降順でソート
    result_df = result_df.sort_values("avg_volume", ascending=False).reset_index(drop=True)
    logger.info(f"フィルタ後銘柄数: {len(result_df)}")
    return result_df
