"""
Flaskウェブアプリケーション
- 銘柄検索、予測表示
- 取引履歴CSVアップロード・分析
"""
import os
import sys
import json
import logging
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory
from datetime import timedelta as _timedelta

try:
    from flask_cors import CORS
except ImportError:
    # flask-cors がない場合のフォールバック
    class CORS:
        def __init__(self, app, **kwargs): pass

# パス設定
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data_fetcher import fetch_1min_data, fetch_daily_data, get_market_session, is_market_open
from src.ml_trainer import predict_next_candle, load_model, load_progress
from src.trade_analyzer import full_trade_analysis, parse_sbi_csv, calculate_trade_metrics
from src.stock_list import fetch_jpx_stock_list

logger = logging.getLogger(__name__)

PREDICTIONS_CACHE = os.path.join(ROOT, "data", "cache", "predictions.json")


def create_app() -> Flask:
    app = Flask(
        __name__,
        static_folder=os.path.join(ROOT, "web", "static"),
        template_folder=os.path.join(ROOT, "web", "templates"),
    )
    CORS(app)

    # ======= 静的ファイル =======
    @app.route("/")
    def index():
        return send_from_directory(os.path.join(ROOT, "web"), "index.html")

    @app.route("/<path:filename>")
    def static_files(filename):
        return send_from_directory(os.path.join(ROOT, "web"), filename)

    # ======= API: 銘柄検索 =======
    @app.route("/api/search")
    def search_stock():
        query = request.args.get("q", "").strip()
        if not query:
            return jsonify({"error": "検索クエリを入力してください"}), 400

        try:
            jpx_df = fetch_jpx_stock_list()
            # コードまたは名前で検索
            mask = (
                jpx_df["コード"].astype(str).str.contains(query, na=False) |
                jpx_df["銘柄名"].astype(str).str.contains(query, na=False)
            )
            results = jpx_df[mask].head(20)

            stocks = []
            for _, row in results.iterrows():
                code = str(row.get("コード", "")).zfill(4)
                stocks.append({
                    "code": code,
                    "ticker": f"{code}.T",
                    "name": str(row.get("銘柄名", "")),
                    "market": str(row.get("市場・商品区分", row.get("市場区分", ""))),
                })
            return jsonify({"results": stocks, "count": len(stocks)})

        except Exception as e:
            logger.error(f"検索エラー: {e}")
            return jsonify({"error": str(e)}), 500

    # ======= API: 銘柄予測 =======
    @app.route("/api/predict/<ticker>")
    def predict_ticker(ticker):
        if not ticker.endswith(".T"):
            ticker = f"{ticker.zfill(4)}.T"

        session = get_market_session()
        market_open = is_market_open()

        # キャッシュから予測を読む
        cached_pred = None
        if os.path.exists(PREDICTIONS_CACHE):
            try:
                with open(PREDICTIONS_CACHE, "r", encoding="utf-8") as f:
                    all_preds = json.load(f)
                cached_pred = all_preds.get(ticker)
            except Exception:
                pass

        # 1分足データ取得
        df_1m = fetch_1min_data(ticker, period="7d")

        if df_1m is None or df_1m.empty:
            if cached_pred:
                return jsonify({
                    "prediction": cached_pred,
                    "session": session,
                    "market_open": market_open,
                    "data_source": "cache",
                })
            return jsonify({"error": f"{ticker}のデータを取得できませんでした"}), 404

        # リアルタイム予測
        pred = predict_next_candle(ticker, df_1m)
        if pred is None:
            return jsonify({"error": "予測モデルの実行に失敗しました"}), 500

        # 最新のローソク足情報を追加
        latest = df_1m.iloc[-1]
        pred["latest_candle"] = {
            "time": str(df_1m.index[-1]),
            "open": round(float(latest["Open"]), 2),
            "high": round(float(latest["High"]), 2),
            "low": round(float(latest["Low"]), 2),
            "close": round(float(latest["Close"]), 2),
            "volume": int(latest["Volume"]),
        }
        pred["session"] = session
        pred["market_open"] = market_open

        # チャートデータ（直近100本）
        chart_data = []
        for ts, row in df_1m.tail(100).iterrows():
            chart_data.append({
                "time": str(ts),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
            })
        pred["chart_data"] = chart_data

        return jsonify(pred)

    # ======= API: 全銘柄予測一覧 =======
    @app.route("/api/predictions")
    def get_all_predictions():
        if os.path.exists(PREDICTIONS_CACHE):
            try:
                with open(PREDICTIONS_CACHE, "r", encoding="utf-8") as f:
                    preds = json.load(f)
                return jsonify({
                    "predictions": preds,
                    "count": len(preds),
                    "session": get_market_session(),
                    "market_open": is_market_open(),
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        return jsonify({
            "predictions": {},
            "count": 0,
            "session": get_market_session(),
            "market_open": is_market_open(),
            "message": "予測データがまだ生成されていません。機械学習の完了をお待ちください。",
        })

    # ======= API: 取引履歴分析 =======
    @app.route("/api/analyze-trades", methods=["POST"])
    def analyze_trades():
        if "file" not in request.files:
            return jsonify({"error": "CSVファイルをアップロードしてください"}), 400

        file = request.files["file"]
        if not file.filename or not file.filename.endswith(".csv"):
            return jsonify({"error": ".csvファイルをアップロードしてください"}), 400

        content = file.read()
        if len(content) == 0:
            return jsonify({"error": "空のファイルです"}), 400

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        use_ai = bool(api_key)

        try:
            result = full_trade_analysis(content, use_ai=use_ai, anthropic_api_key=api_key)
            return jsonify(result)
        except Exception as e:
            logger.error(f"取引分析エラー: {e}", exc_info=True)
            return jsonify({"error": f"分析中にエラーが発生しました: {str(e)}"}), 500

    # ======= API: モデル状況 =======
    @app.route("/api/status")
    def get_status():
        progress = load_progress()
        models_dir = os.path.join(ROOT, "models")
        model_files = [f for f in os.listdir(models_dir) if f.endswith("_model.pkl")] if os.path.exists(models_dir) else []

        return jsonify({
            "session": get_market_session(),
            "market_open": is_market_open(),
            "trained_models": len(model_files),
            "training_progress": progress,
            "server_time_jst": (datetime.utcnow().replace(tzinfo=None) + _timedelta(hours=9)).isoformat(),
        })

    # ======= API: ヘルスチェック =======
    @app.route("/api/health")
    def health():
        return jsonify({"status": "ok", "timestamp": datetime.utcnow().isoformat()})

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
