"""
取引履歴分析モジュール
SBI証券のCSV取引履歴を読み込み、
エントリー・決済位置の最適化とテクニカル指標アドバイスを生成する
"""
import os
import io
import logging
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np

from .features import compute_features, _rsi, _ema, _bollinger_bands, _macd
from .data_fetcher import fetch_daily_data, fetch_1min_data

logger = logging.getLogger(__name__)

# SBI証券CSV列定義（国内株式・信用取引）
SBI_COLUMNS_MAPPING = {
    # 現物
    "約定日": "date",
    "銘柄名": "name",
    "銘柄コード": "code",
    "売買": "side",
    "約定数量": "qty",
    "約定単価": "price",
    "手数料": "fee",
    "税額": "tax",
    "受渡金額": "settlement",
    # 信用取引
    "建玉日": "open_date",
    "返済日": "close_date",
    "建単価": "open_price",
    "返済単価": "close_price",
    "建玉数量": "qty",
    "売買区分": "side",
}


def parse_sbi_csv(file_content: bytes, encoding: str = "shift_jis") -> Optional[pd.DataFrame]:
    """
    SBI証券の取引履歴CSVをパースする

    Parameters
    ----------
    file_content : bytes  CSVファイルの内容
    encoding : str  文字コード

    Returns
    -------
    DataFrame with standardized columns
    """
    for enc in [encoding, "utf-8", "cp932", "utf-8-sig"]:
        try:
            text = file_content.decode(enc)
            break
        except (UnicodeDecodeError, AttributeError):
            continue
    else:
        try:
            text = file_content.decode("utf-8", errors="replace")
        except Exception:
            return None

    # ヘッダー行を探す
    lines = text.split("\n")
    header_idx = 0
    for i, line in enumerate(lines):
        if "銘柄" in line or "約定" in line or "コード" in line:
            header_idx = i
            break

    try:
        df = pd.read_csv(
            io.StringIO("\n".join(lines[header_idx:])),
            dtype=str,
            on_bad_lines="skip",
        )
    except Exception as e:
        logger.error(f"CSV解析失敗: {e}")
        return None

    # 列名の正規化
    df.columns = df.columns.str.strip()
    rename_map = {}
    for orig, new in SBI_COLUMNS_MAPPING.items():
        for col in df.columns:
            if orig in col:
                rename_map[col] = new
                break
    df = df.rename(columns=rename_map)

    # 数値変換
    for col in ["qty", "price", "fee", "tax", "settlement", "open_price", "close_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "").str.replace("円", "").str.strip(),
                errors="coerce"
            )

    # 日付変換
    for col in ["date", "open_date", "close_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df = df.dropna(how="all")
    return df


def calculate_trade_metrics(trades: pd.DataFrame) -> Dict[str, Any]:
    """
    取引メトリクスを計算する
    """
    if trades.empty:
        return {}

    metrics = {
        "total_trades": len(trades),
    }

    # PnL計算（信用取引の場合）
    if "open_price" in trades.columns and "close_price" in trades.columns:
        # 買い建て: close_price - open_price
        # 売り建て: open_price - close_price
        buy_trades = trades[trades["side"].astype(str).str.contains("買|BUY|buy", na=False)]
        sell_trades = trades[trades["side"].astype(str).str.contains("売|SELL|sell", na=False)]

        pnl_list = []
        for _, row in trades.iterrows():
            try:
                qty = float(row.get("qty", 1) or 1)
                open_p = float(row.get("open_price", 0) or 0)
                close_p = float(row.get("close_price", 0) or 0)
                side_str = str(row.get("side", "")).lower()
                if "売" in side_str or "sell" in side_str:
                    pnl = (open_p - close_p) * qty
                else:
                    pnl = (close_p - open_p) * qty
                fee = float(row.get("fee", 0) or 0)
                pnl_list.append(pnl - fee)
            except Exception:
                continue

        if pnl_list:
            wins = [p for p in pnl_list if p > 0]
            losses = [p for p in pnl_list if p < 0]
            metrics.update({
                "total_pnl": sum(pnl_list),
                "win_rate": len(wins) / len(pnl_list) if pnl_list else 0,
                "avg_win": np.mean(wins) if wins else 0,
                "avg_loss": np.mean(losses) if losses else 0,
                "risk_reward": abs(np.mean(wins) / np.mean(losses)) if wins and losses else 0,
                "profit_factor": sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 0,
                "max_win": max(pnl_list) if pnl_list else 0,
                "max_loss": min(pnl_list) if pnl_list else 0,
                "n_wins": len(wins),
                "n_losses": len(losses),
            })

    return metrics


def analyze_entry_exit_optimization(
    trade_row: pd.Series,
    ohlcv: pd.DataFrame,
) -> Dict[str, Any]:
    """
    1トレードのエントリー・決済位置を分析し、最適なポイントを提案する

    Parameters
    ----------
    trade_row : 1トレードの行
    ohlcv : そのトレード期間の価格データ

    Returns
    -------
    dict with optimization suggestions
    """
    result = {}

    if ohlcv.empty:
        return result

    try:
        open_price = float(trade_row.get("open_price", 0) or 0)
        close_price = float(trade_row.get("close_price", 0) or 0)
        side = str(trade_row.get("side", "")).lower()
        is_long = not ("売" in side or "sell" in side)

        # 特徴量計算
        feat = compute_features(ohlcv)

        # 高値・安値
        period_high = ohlcv["High"].max()
        period_low = ohlcv["Low"].min()

        # ボリンジャーバンド（期間終了時点）
        bb_upper, bb_mid, bb_lower = _bollinger_bands(ohlcv["Close"], 20)

        # RSI
        rsi = _rsi(ohlcv["Close"], 14)

        # MACD
        macd_line, macd_signal, macd_hist = _macd(ohlcv["Close"])

        # 移動平均
        ma5 = ohlcv["Close"].rolling(5).mean()
        ma25 = ohlcv["Close"].rolling(25).mean()

        actual_pnl = (close_price - open_price) * (1 if is_long else -1)
        best_entry = period_low if is_long else period_high
        best_exit = period_high if is_long else period_low
        best_pnl = (best_exit - best_entry) * (1 if is_long else -1)

        result = {
            "actual_entry": open_price,
            "actual_exit": close_price,
            "actual_pnl_per_share": round(actual_pnl, 2),
            "optimal_entry": round(best_entry, 2),
            "optimal_exit": round(best_exit, 2),
            "optimal_pnl_per_share": round(best_pnl, 2),
            "improvement_potential": round(best_pnl - actual_pnl, 2),
            "period_high": round(float(period_high), 2),
            "period_low": round(float(period_low), 2),
            "rsi_at_entry": round(float(rsi.iloc[0]), 1) if len(rsi) > 0 else None,
            "rsi_at_exit": round(float(rsi.iloc[-1]), 1) if len(rsi) > 0 else None,
            "ma5_at_exit": round(float(ma5.iloc[-1]), 2) if len(ma5) > 0 else None,
            "ma25_at_exit": round(float(ma25.iloc[-1]), 2) if len(ma25) > 0 else None,
            "bb_upper_at_exit": round(float(bb_upper.iloc[-1]), 2) if len(bb_upper) > 0 else None,
            "bb_lower_at_exit": round(float(bb_lower.iloc[-1]), 2) if len(bb_lower) > 0 else None,
        }

        # エントリータイミングのアドバイス
        if is_long:
            entry_advice = []
            if open_price > float(ma25.iloc[0]) * 1.05:
                entry_advice.append("エントリー時に25MAから5%以上乖離しており過熱気味でした。25MAへの押し目を待つとより良いエントリーが可能でした。")
            if float(rsi.iloc[0]) > 70:
                entry_advice.append(f"エントリー時のRSIが{float(rsi.iloc[0]):.0f}と過買い圏でした。RSI50〜60付近でのエントリーがより有利です。")
            if float(bb_lower.iloc[0]) > open_price * 0.99:
                entry_advice.append("ボリンジャーバンド下限付近でのエントリーはリスクが高い可能性があります。")
        else:
            entry_advice = []
            if open_price < float(ma25.iloc[0]) * 0.95:
                entry_advice.append("売り建て時に25MAから5%以上下に乖離しており、リバウンドリスクがありました。")
            if float(rsi.iloc[0]) < 30:
                entry_advice.append(f"売り建て時のRSIが{float(rsi.iloc[0]):.0f}と過売り圏でした。")

        result["entry_advice"] = entry_advice if entry_advice else ["エントリータイミングは適切でした。"]

        # 決済タイミングのアドバイス
        exit_advice = []
        if is_long and close_price < period_high * 0.97:
            exit_advice.append(f"高値{period_high:.0f}円に対して{(1-close_price/period_high)*100:.1f}%早く利益確定しました。トレーリングストップの活用で利益拡大が可能でした。")
        if is_long and actual_pnl < 0 and close_price < float(ma25.iloc[-1]):
            exit_advice.append("25MA割れを損切りサインとして活用することで、損失を抑制できた可能性があります。")

        result["exit_advice"] = exit_advice if exit_advice else ["決済タイミングは適切でした。"]

    except Exception as e:
        logger.error(f"エントリー・決済分析失敗: {e}")

    return result


def generate_technical_advice(
    ticker: str,
    trades: pd.DataFrame,
    ohlcv: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    銘柄ごとのテクニカル分析アドバイスを生成する

    Returns
    -------
    dict with advice per indicator
    """
    advice = {
        "ticker": ticker,
        "indicators_advice": [],
        "recommended_strategies": [],
    }

    if ohlcv is None:
        ohlcv = fetch_daily_data(ticker)
    if ohlcv is None or ohlcv.empty:
        return advice

    close = ohlcv["Close"]
    feat = compute_features(ohlcv)

    # 移動平均の有効性
    ma5 = close.rolling(5).mean()
    ma25 = close.rolling(25).mean()
    ma75 = close.rolling(75).mean()

    golden_cross_profit = 0
    cross_count = 0
    for i in range(1, len(ma5)):
        if ma5.iloc[i] > ma25.iloc[i] and ma5.iloc[i-1] <= ma25.iloc[i-1]:
            future_idx = min(i + 5, len(close) - 1)
            profit = (close.iloc[future_idx] - close.iloc[i]) / close.iloc[i] * 100
            golden_cross_profit += profit
            cross_count += 1

    if cross_count > 0:
        avg_golden_cross_profit = golden_cross_profit / cross_count
        if avg_golden_cross_profit > 1.0:
            advice["indicators_advice"].append({
                "indicator": "移動平均線（5MA/25MAゴールデンクロス）",
                "effectiveness": "高い",
                "detail": f"この銘柄では5MA・25MAのゴールデンクロス後5日間の平均収益が{avg_golden_cross_profit:.1f}%と有効です。クロス発生時のエントリーを検討してください。",
                "suggested_params": {"fast": 5, "slow": 25},
            })

    # RSIの有効性
    rsi = _rsi(close, 14)
    oversold_signals = (rsi < 30).astype(int).diff()
    recovery_profits = []
    for idx in oversold_signals[oversold_signals == 1].index:
        try:
            pos = close.index.get_loc(idx)
            future_pos = min(pos + 5, len(close) - 1)
            profit = (close.iloc[future_pos] - close.iloc[pos]) / close.iloc[pos] * 100
            recovery_profits.append(profit)
        except Exception:
            continue

    if recovery_profits:
        avg_rsi_profit = np.mean(recovery_profits)
        if avg_rsi_profit > 0.5:
            advice["indicators_advice"].append({
                "indicator": "RSI（14期間）",
                "effectiveness": "高い" if avg_rsi_profit > 2.0 else "中程度",
                "detail": f"RSI30以下の過売りシグナル後の平均5日収益が{avg_rsi_profit:.1f}%です。逆張りエントリーに有効です。",
                "suggested_params": {"period": 14, "oversold": 30, "overbought": 70},
            })

    # ボリンジャーバンド
    bb_upper, bb_mid, bb_lower = _bollinger_bands(close, 20)
    bb_squeeze = (bb_upper - bb_lower) / bb_mid
    squeeze_threshold = bb_squeeze.quantile(0.2)

    advice["indicators_advice"].append({
        "indicator": "ボリンジャーバンド（20期間、2σ）",
        "effectiveness": "中程度",
        "detail": f"バンド幅の縮小（スクイーズ）後にブレイクアウトが発生しやすい傾向があります。現在のバンド幅は{float(bb_squeeze.iloc[-1]):.3f}（下位20%閾値: {squeeze_threshold:.3f}）。",
        "suggested_params": {"period": 20, "std": 2},
    })

    # MACD
    macd_line, macd_signal, macd_hist = _macd(close)
    macd_buy = ((macd_line > macd_signal) & (macd_line.shift(1) <= macd_signal.shift(1)))
    macd_profits = []
    for idx in macd_buy[macd_buy].index:
        try:
            pos = close.index.get_loc(idx)
            future_pos = min(pos + 5, len(close) - 1)
            profit = (close.iloc[future_pos] - close.iloc[pos]) / close.iloc[pos] * 100
            macd_profits.append(profit)
        except Exception:
            continue

    if macd_profits:
        avg_macd_profit = np.mean(macd_profits)
        advice["indicators_advice"].append({
            "indicator": "MACD（12,26,9）",
            "effectiveness": "高い" if avg_macd_profit > 1.0 else "低い",
            "detail": f"MACDゴールデンクロス後5日の平均収益{avg_macd_profit:.1f}%。シグナルライン交差をエントリートリガーとして活用できます。",
            "suggested_params": {"fast": 12, "slow": 26, "signal": 9},
        })

    # 推奨戦略
    metrics = calculate_trade_metrics(trades)
    win_rate = metrics.get("win_rate", 0.5)
    rr = metrics.get("risk_reward", 1.0)

    if win_rate < 0.5 and rr > 1.5:
        advice["recommended_strategies"].append({
            "strategy": "トレンドフォロー",
            "reason": f"勝率{win_rate*100:.0f}%だがリスクリワード{rr:.1f}倍と高い。エントリー精度より利益伸長に集中する戦略が合っています。",
        })
    elif win_rate > 0.6 and rr < 1.0:
        advice["recommended_strategies"].append({
            "strategy": "損切りルールの厳格化",
            "reason": f"勝率{win_rate*100:.0f}%と高いが、損失が利益を上回っています（RR={rr:.1f}）。ATRの1.5倍を損切りラインにする等、損失コントロールが必要です。",
        })
    else:
        advice["recommended_strategies"].append({
            "strategy": "現状維持・微調整",
            "reason": f"勝率{win_rate*100:.0f}%、RR{rr:.1f}倍とバランスが取れています。エントリーポイントの精度を上げることで安定性が向上します。",
        })

    return advice


def full_trade_analysis(
    file_content: bytes,
    use_ai: bool = True,
    anthropic_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    取引履歴CSVの完全分析を実行する

    Parameters
    ----------
    file_content : CSVファイルバイト
    use_ai : Claude APIでアドバイス生成するか
    anthropic_api_key : Anthropic APIキー

    Returns
    -------
    完全な分析結果dict
    """
    trades = parse_sbi_csv(file_content)
    if trades is None or trades.empty:
        return {"error": "CSVの解析に失敗しました。SBI証券の取引履歴CSVをアップロードしてください。"}

    # 全体メトリクス
    metrics = calculate_trade_metrics(trades)

    # 銘柄別分析
    per_ticker_analysis = {}
    code_col = next((c for c in ["code", "銘柄コード", "コード"] if c in trades.columns), None)

    if code_col:
        for code in trades[code_col].dropna().unique()[:10]:  # 最大10銘柄
            ticker = f"{str(code).strip().zfill(4)}.T"
            ticker_trades = trades[trades[code_col] == code]
            ohlcv = fetch_daily_data(ticker)
            tech_advice = generate_technical_advice(ticker, ticker_trades, ohlcv)
            per_ticker_analysis[ticker] = {
                "trade_count": len(ticker_trades),
                "technical_advice": tech_advice,
            }

            # エントリー・決済分析（最新トレード）
            if not ticker_trades.empty and ohlcv is not None:
                last_trade = ticker_trades.iloc[-1]
                entry_exit = analyze_entry_exit_optimization(last_trade, ohlcv)
                per_ticker_analysis[ticker]["entry_exit_analysis"] = entry_exit

    result = {
        "summary_metrics": metrics,
        "per_ticker_analysis": per_ticker_analysis,
        "total_trades_analyzed": len(trades),
        "analysis_timestamp": datetime.now().isoformat(),
    }

    # AI分析（オプション）
    if use_ai and anthropic_api_key:
        ai_advice = generate_ai_advice(metrics, per_ticker_analysis, anthropic_api_key)
        result["ai_advice"] = ai_advice

    return result


def generate_ai_advice(
    metrics: dict,
    per_ticker: dict,
    api_key: str
) -> str:
    """Claude APIを使って包括的なアドバイスを生成"""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        summary = f"""
取引サマリー:
- 総取引数: {metrics.get('total_trades', 0)}
- 勝率: {metrics.get('win_rate', 0)*100:.1f}%
- 平均利益: {metrics.get('avg_win', 0):.0f}円
- 平均損失: {metrics.get('avg_loss', 0):.0f}円
- リスクリワード比: {metrics.get('risk_reward', 0):.2f}
- プロフィットファクター: {metrics.get('profit_factor', 0):.2f}
- 最大利益: {metrics.get('max_win', 0):.0f}円
- 最大損失: {metrics.get('max_loss', 0):.0f}円
"""
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": f"""あなたはプロのトレーディングコーチです。
以下の取引履歴データを分析し、日本語で具体的な改善アドバイスを提供してください。

{summary}

以下の観点から分析してください：
1. 現在の強みと弱み
2. リスク管理の改善点
3. エントリー・決済タイミングの改善提案
4. 推奨するテクニカル指標と使い方
5. 具体的な行動計画（3つ以内）

簡潔かつ実践的なアドバイスをお願いします。"""
            }]
        )
        return msg.content[0].text
    except Exception as e:
        logger.error(f"AI分析失敗: {e}")
        return f"AI分析は利用できません（APIキーを設定するか、後でお試しください）"
