"""
メインオーケストレーター
市場時間に応じてML学習・予測を制御する
"""
import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# ルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_fetcher import (
    get_market_session, is_market_open,
    fetch_1min_data, fetch_daily_data,
    fetch_batch_1min
)
from src.stock_list import get_tradeable_stocks, fetch_jpx_stock_list
from src.ml_trainer import (
    train_model, load_model, predict_next_candle,
    batch_train, load_progress, save_progress
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(os.path.dirname(__file__), "logs", "orchestrator.log"),
            mode="a", encoding="utf-8"
        ),
    ]
)
logger = logging.getLogger("orchestrator")

HISTORICAL_START = "1991-01-01"  # バブル崩壊後
PROGRESS_FILE = os.path.join(os.path.dirname(__file__), "models", "training_progress.json")
HISTORICAL_DONE_FILE = os.path.join(os.path.dirname(__file__), "models", "historical_training_done.json")

MAX_HISTORICAL_STOCKS = 300  # 歴史的学習の最大銘柄数
MAX_INTRADAY_STOCKS = 100    # ザラバ学習の最大銘柄数


def is_historical_training_done() -> bool:
    """初回の歴史的学習が完了しているか"""
    return os.path.exists(HISTORICAL_DONE_FILE)


def mark_historical_training_done(trained_count: int) -> None:
    """歴史的学習完了をマーク"""
    with open(HISTORICAL_DONE_FILE, "w") as f:
        json.dump({
            "completed_at": datetime.now().isoformat(),
            "trained_count": trained_count,
        }, f)


def run_historical_training(max_time_sec: int = 20000) -> None:
    """
    バブル崩壊後〜前日までの日足データで一度だけ学習する
    （完了後は繰り返さない）
    """
    if is_historical_training_done():
        logger.info("歴史的学習は完了済みです。スキップします。")
        return

    logger.info("=== 歴史的学習（バブル崩壊後〜前日）を開始します ===")

    # 銘柄取得（出来高でフィルタ）
    stocks_df = get_tradeable_stocks(mode="swing", max_stocks=MAX_HISTORICAL_STOCKS)
    if stocks_df.empty:
        logger.error("銘柄リストの取得に失敗しました")
        return

    tickers = stocks_df["ticker"].tolist()
    logger.info(f"対象銘柄数: {len(tickers)}")

    # 前回の進捗確認
    progress = load_progress()
    trained_set = set(progress.get("trained", [])) if progress else set()
    remaining = [t for t in tickers if t not in trained_set]
    logger.info(f"学習残り: {len(remaining)}/{len(tickers)}")

    # 日足データ取得 & 学習
    trained_this_run = list(trained_set)
    start_time = time.time()

    for ticker in remaining:
        elapsed = time.time() - start_time
        if elapsed > max_time_sec:
            logger.warning(f"時間制限到達。進捗保存して停止: {len(trained_this_run)}/{len(tickers)}")
            save_progress(trained_this_run, tickers, "historical")
            return

        logger.info(f"日足データ取得: {ticker}")
        df = fetch_daily_data(ticker, start=HISTORICAL_START)
        if df is None or len(df) < 100:
            logger.warning(f"{ticker}: データ不足スキップ")
            continue

        result = train_model(ticker, df)
        if result:
            trained_this_run.append(ticker)
            save_progress(trained_this_run, tickers, "historical")

        time.sleep(2.0)

    logger.info(f"=== 歴史的学習完了: {len(trained_this_run)}銘柄 ===")
    mark_historical_training_done(len(trained_this_run))


def run_intraday_training(max_time_sec: int = 3000) -> None:
    """
    ザラバ中の1分足データで継続的に学習する
    """
    logger.info("=== ザラバML学習を開始します ===")

    stocks_df = get_tradeable_stocks(mode="daytrade", max_stocks=MAX_INTRADAY_STOCKS)
    if stocks_df.empty:
        logger.warning("デイトレ可能銘柄が見つかりませんでした")
        return

    tickers = stocks_df["ticker"].tolist()
    logger.info(f"対象銘柄数: {len(tickers)}")

    start_time = time.time()
    for ticker in tickers:
        if time.time() - start_time > max_time_sec:
            break

        df = fetch_1min_data(ticker, period="7d")
        if df is None or len(df) < 100:
            continue

        result = train_model(ticker, df)
        if result:
            logger.info(f"{ticker}: 1分足学習完了 精度={result['metrics']['cv_accuracy_mean']:.3f}")

        time.sleep(1.5)


def run_prediction_update() -> Dict[str, dict]:
    """
    現在のザラバデータで全銘柄の予測を更新する
    """
    logger.info("予測更新開始")
    stocks_df = get_tradeable_stocks(mode="daytrade", max_stocks=MAX_INTRADAY_STOCKS)
    if stocks_df.empty:
        return {}

    predictions = {}
    for _, row in stocks_df.iterrows():
        ticker = row["ticker"]
        df = fetch_1min_data(ticker, period="1d", cache_minutes=2)
        if df is None or len(df) < 30:
            continue

        pred = predict_next_candle(ticker, df)
        if pred:
            pred["name"] = row.get("name", "")
            pred["price"] = row.get("price", 0)
            pred["avg_volume"] = row.get("avg_volume", 0)
            predictions[ticker] = pred

    # 予測結果を保存
    pred_path = os.path.join(os.path.dirname(__file__), "data", "cache", "predictions.json")
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    logger.info(f"予測更新完了: {len(predictions)}銘柄")
    return predictions


def main():
    parser = argparse.ArgumentParser(description="JP Stock ML Orchestrator")
    parser.add_argument(
        "--mode",
        choices=["auto", "historical", "intraday", "predict", "serve"],
        default="auto",
        help="実行モード"
    )
    parser.add_argument("--max-time", type=int, default=20000, help="最大実行時間（秒）")
    args = parser.parse_args()

    os.makedirs(os.path.join(os.path.dirname(__file__), "logs"), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), "models"), exist_ok=True)

    if args.mode == "auto":
        session = get_market_session()
        logger.info(f"現在のセッション: {session}")

        if session in ("morning", "afternoon"):
            # ザラバ中: 1分足学習 + 予測更新
            run_intraday_training(max_time_sec=min(args.max_time, 2400))
            run_prediction_update()
        else:
            # 開場前・閉場後: 歴史的学習（初回のみ）
            run_historical_training(max_time_sec=args.max_time)

    elif args.mode == "historical":
        run_historical_training(max_time_sec=args.max_time)

    elif args.mode == "intraday":
        run_intraday_training(max_time_sec=args.max_time)

    elif args.mode == "predict":
        predictions = run_prediction_update()
        print(json.dumps(predictions, ensure_ascii=False, indent=2))

    elif args.mode == "serve":
        # Webサーバー起動
        from web.app import create_app
        app = create_app()
        port = int(os.environ.get("PORT", 5000))
        app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
