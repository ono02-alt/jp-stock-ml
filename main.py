"""
メインオーケストレーター
市場時間に応じてML学習・予測を制御する
"""
import os
import sys
import json
import time
import shutil
import logging
import argparse
from datetime import datetime
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_fetcher import (
    get_market_session, is_market_open,
    fetch_1min_data, fetch_daily_data,
)
from src.stock_list import get_tradeable_stocks
from src.ml_trainer import (
    train_model, predict_next_candle,
    load_progress, save_progress,
)

_ROOT = os.path.dirname(__file__)

# ログ設定
_log_dir = os.path.join(_ROOT, "logs")
os.makedirs(_log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(_log_dir, "orchestrator.log"),
            mode="a", encoding="utf-8"
        ),
    ]
)
logger = logging.getLogger("orchestrator")

HISTORICAL_START      = "1991-01-01"
MAX_HISTORICAL_STOCKS = 300
MAX_INTRADAY_STOCKS   = 100
HISTORICAL_DONE_FILE  = os.path.join(_ROOT, "models", "historical_training_done.json")

# GitHub Pages 用出力ディレクトリ
DOCS_DIR      = os.path.join(_ROOT, "docs")
DOCS_DATA_DIR = os.path.join(DOCS_DIR, "data")


def is_historical_training_done() -> bool:
    return os.path.exists(HISTORICAL_DONE_FILE)


def mark_historical_training_done(trained_count: int) -> None:
    with open(HISTORICAL_DONE_FILE, "w") as f:
        json.dump({
            "completed_at": datetime.now().isoformat(),
            "trained_count": trained_count,
        }, f)


def run_historical_training(max_time_sec: int = 20000) -> None:
    """バブル崩壊後〜前日の日足データで一度だけ学習（完了後は繰り返さない）"""
    if is_historical_training_done():
        logger.info("歴史的学習は完了済みです。スキップします。")
        return

    logger.info("=== 歴史的学習（バブル崩壊後〜前日）を開始します ===")

    stocks_df = get_tradeable_stocks(mode="swing", max_stocks=MAX_HISTORICAL_STOCKS)
    if stocks_df.empty:
        logger.error("銘柄リストの取得に失敗しました")
        return

    tickers = stocks_df["ticker"].tolist()
    logger.info(f"対象銘柄数: {len(tickers)}")

    progress     = load_progress()
    trained_set  = set(progress.get("trained", [])) if progress else set()
    remaining    = [t for t in tickers if t not in trained_set]
    logger.info(f"学習残り: {len(remaining)}/{len(tickers)}")

    trained_this_run = list(trained_set)
    start_time = time.time()

    for ticker in remaining:
        if time.time() - start_time > max_time_sec:
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
    """ザラバ中の1分足データで継続的に学習する"""
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
    """現在のザラバデータで全銘柄の予測を更新する"""
    logger.info("予測更新開始")

    stocks_df = get_tradeable_stocks(mode="daytrade", max_stocks=MAX_INTRADAY_STOCKS)
    if stocks_df.empty:
        return {}

    predictions: Dict[str, dict] = {}
    for _, row in stocks_df.iterrows():
        ticker = row["ticker"]
        df = fetch_1min_data(ticker, period="1d", cache_minutes=2)
        if df is None or len(df) < 30:
            continue
        pred = predict_next_candle(ticker, df)
        if pred:
            pred["name"]       = str(row.get("name", ""))
            pred["price"]      = float(row.get("price", 0))
            pred["avg_volume"] = int(row.get("avg_volume", 0))
            predictions[ticker] = pred

    # data/cache/ に保存（Flask 用）
    pred_path = os.path.join(_ROOT, "data", "cache", "predictions.json")
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    logger.info(f"予測更新完了: {len(predictions)}銘柄")
    return predictions


def publish_to_pages(predictions: Dict[str, dict]) -> None:
    """
    GitHub Pages 用に docs/data/ へ予測結果・学習進捗を書き出す
    GitHub Actions が commit/push することで Pages に反映される
    """
    os.makedirs(DOCS_DATA_DIR, exist_ok=True)

    # --- 予測結果 ---
    out_pred = os.path.join(DOCS_DATA_DIR, "predictions.json")
    with open(out_pred, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    logger.info(f"Pages用予測データ書き出し: {out_pred} ({len(predictions)}銘柄)")

    # --- 学習進捗・ステータス ---
    progress = load_progress()
    models_dir   = os.path.join(_ROOT, "models")
    model_count  = len([f for f in os.listdir(models_dir) if f.endswith("_model.pkl")]) if os.path.exists(models_dir) else 0
    jst_now      = (datetime.utcnow()).isoformat()  # UTC で記録、UI 側で変換

    status = {
        "session":           get_market_session(),
        "market_open":       is_market_open(),
        "trained_models":    model_count,
        "training_progress": progress,
        "updated_at_utc":    jst_now,
    }
    out_status = os.path.join(DOCS_DATA_DIR, "status.json")
    with open(out_status, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)
    logger.info(f"Pages用ステータス書き出し: {out_status}")

    # --- docs/index.html をルートの web/index.html からコピー ---
    src_html = os.path.join(_ROOT, "web", "index.html")
    dst_html = os.path.join(DOCS_DIR, "index.html")
    if os.path.exists(src_html):
        shutil.copy2(src_html, dst_html)
        logger.info(f"index.html コピー: {dst_html}")

    # GitHub Pages 用 .nojekyll（Jekyll 処理をスキップ）
    nojekyll = os.path.join(DOCS_DIR, ".nojekyll")
    if not os.path.exists(nojekyll):
        open(nojekyll, "w").close()


def main() -> None:
    parser = argparse.ArgumentParser(description="JP Stock ML Orchestrator")
    parser.add_argument(
        "--mode",
        choices=["auto", "historical", "intraday", "predict", "pages", "serve"],
        default="auto",
        help=(
            "auto: 市場時間で自動切替 / historical: 歴史的学習 / "
            "intraday: ザラバ学習 / predict: 予測更新 / "
            "pages: GitHub Pages 用データ書き出し / serve: Flask サーバー起動"
        ),
    )
    parser.add_argument("--max-time", type=int, default=20000, help="最大実行時間（秒）")
    args = parser.parse_args()

    os.makedirs(os.path.join(_ROOT, "models"), exist_ok=True)

    if args.mode == "auto":
        session = get_market_session()
        logger.info(f"現在のセッション: {session}")
        if session in ("morning", "afternoon"):
            run_intraday_training(max_time_sec=min(args.max_time, 2400))
            preds = run_prediction_update()
            publish_to_pages(preds)
        else:
            run_historical_training(max_time_sec=args.max_time)

    elif args.mode == "historical":
        run_historical_training(max_time_sec=args.max_time)

    elif args.mode == "intraday":
        run_intraday_training(max_time_sec=args.max_time)

    elif args.mode == "predict":
        preds = run_prediction_update()
        publish_to_pages(preds)
        print(json.dumps(preds, ensure_ascii=False, indent=2))

    elif args.mode == "pages":
        # 既存の predictions.json から書き出すだけ（再学習なし）
        pred_path = os.path.join(_ROOT, "data", "cache", "predictions.json")
        if os.path.exists(pred_path):
            with open(pred_path, encoding="utf-8") as f:
                preds = json.load(f)
        else:
            preds = {}
        publish_to_pages(preds)

    elif args.mode == "serve":
        from web.app import create_app
        app = create_app()
        port = int(os.environ.get("PORT", 5000))
        app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()
