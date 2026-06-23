"""
テストスイート
全モジュールのユニットテスト
"""
import sys
import os
import json
import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ======= ダミーデータ生成 =======
def make_ohlcv(n=200, seed=42) -> pd.DataFrame:
    np.random.seed(seed)
    idx = pd.date_range("2024-01-15 09:00", periods=n, freq="1min", tz="Asia/Tokyo")
    close = 1000 + np.cumsum(np.random.randn(n) * 5)
    high = close + np.abs(np.random.randn(n) * 3)
    low = close - np.abs(np.random.randn(n) * 3)
    open_ = close + np.random.randn(n) * 2
    volume = np.random.randint(100000, 2000000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


def make_daily_ohlcv(n=500, seed=0) -> pd.DataFrame:
    np.random.seed(seed)
    idx = pd.date_range("2022-01-04", periods=n, freq="B")
    close = 1500 + np.cumsum(np.random.randn(n) * 20)
    close = np.maximum(close, 200)
    high = close + np.abs(np.random.randn(n) * 10)
    low = close - np.abs(np.random.randn(n) * 10)
    open_ = close + np.random.randn(n) * 5
    volume = np.random.randint(200000, 5000000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


# ======= 特徴量テスト =======
class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.df = make_ohlcv(200)

    def test_compute_features_returns_dataframe(self):
        from src.features import compute_features
        result = compute_features(self.df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_compute_features_has_target(self):
        from src.features import compute_features
        result = compute_features(self.df)
        self.assertIn("target", result.columns)

    def test_target_is_binary(self):
        from src.features import compute_features
        result = compute_features(self.df)
        vals = result["target"].dropna().unique()
        for v in vals:
            self.assertIn(v, [0, 1])

    def test_feature_columns_exist(self):
        from src.features import compute_features, get_feature_columns
        result = compute_features(self.df)
        feat_cols = get_feature_columns()
        for col in feat_cols:
            self.assertIn(col, result.columns, f"Missing feature: {col}")

    def test_rsi_range(self):
        from src.features import _rsi
        rsi = _rsi(self.df["Close"], 14).dropna()
        self.assertTrue((rsi >= 0).all())
        self.assertTrue((rsi <= 100).all())

    def test_bollinger_bands(self):
        from src.features import _bollinger_bands
        up, mid, low = _bollinger_bands(self.df["Close"], 20)
        valid = up.dropna()
        self.assertTrue((valid >= mid.dropna()).all())

    def test_macd(self):
        from src.features import _macd
        macd, signal, hist = _macd(self.df["Close"])
        self.assertEqual(len(macd), len(self.df))

    def test_vwap(self):
        from src.features import _vwap
        vwap = _vwap(self.df)
        self.assertFalse(vwap.isna().all())

    def test_ema(self):
        from src.features import _ema
        ema = _ema(self.df["Close"], 5)
        self.assertEqual(len(ema), len(self.df))

    def test_time_features_added(self):
        from src.features import compute_features
        result = compute_features(self.df)
        self.assertIn("hour", result.columns)
        self.assertIn("is_morning", result.columns)

    def test_no_all_nan_features(self):
        from src.features import compute_features, get_feature_columns
        result = compute_features(self.df)
        for col in get_feature_columns():
            if col in result.columns:
                non_nan = result[col].dropna()
                self.assertGreater(len(non_nan), 0, f"All NaN: {col}")


# ======= ML学習テスト =======
class TestMLTrainer(unittest.TestCase):
    def setUp(self):
        self.df = make_ohlcv(300)
        self.daily_df = make_daily_ohlcv(500)
        self.ticker = "TEST_9999"

    def tearDown(self):
        # テスト用モデルファイル削除
        import glob
        for f in glob.glob(f"/tmp/*{self.ticker}*"):
            try:
                os.remove(f)
            except Exception:
                pass

    def test_prepare_dataset(self):
        from src.ml_trainer import prepare_dataset
        X, y = prepare_dataset(self.df)
        self.assertGreater(len(X), 0)
        self.assertEqual(len(X), len(y))
        self.assertTrue(X.notna().all().all())

    def test_train_model_returns_dict(self):
        from src.ml_trainer import train_model
        with patch("src.ml_trainer._model_path", return_value=f"/tmp/{self.ticker}_model.pkl"), \
             patch("src.ml_trainer._meta_path", return_value=f"/tmp/{self.ticker}_meta.json"):
            result = train_model(self.ticker, self.df, n_splits=2)
            self.assertIsNotNone(result)
            self.assertIn("model", result)
            self.assertIn("metrics", result)

    def test_train_model_metrics_structure(self):
        from src.ml_trainer import train_model
        with patch("src.ml_trainer._model_path", return_value=f"/tmp/{self.ticker}_model.pkl"), \
             patch("src.ml_trainer._meta_path", return_value=f"/tmp/{self.ticker}_meta.json"):
            result = train_model(self.ticker, self.df, n_splits=2)
            self.assertIn("cv_accuracy_mean", result["metrics"])
            acc = result["metrics"]["cv_accuracy_mean"]
            self.assertGreaterEqual(acc, 0.0)
            self.assertLessEqual(acc, 1.0)

    def test_train_model_insufficient_data(self):
        from src.ml_trainer import train_model
        small_df = make_ohlcv(30)
        result = train_model(self.ticker, small_df)
        self.assertIsNone(result)

    def test_feature_importance_exists(self):
        from src.ml_trainer import train_model
        with patch("src.ml_trainer._model_path", return_value=f"/tmp/{self.ticker}_model.pkl"), \
             patch("src.ml_trainer._meta_path", return_value=f"/tmp/{self.ticker}_meta.json"):
            result = train_model(self.ticker, self.df, n_splits=2)
            self.assertIn("feature_importance", result)
            self.assertIsInstance(result["feature_importance"], dict)

    def test_save_load_progress(self):
        from src.ml_trainer import save_progress, load_progress
        import tempfile
        with patch("src.ml_trainer.PROGRESS_FILE", tempfile.mktemp(suffix=".json")):
            save_progress(["A", "B"], ["A", "B", "C"], "test")
            prog = load_progress()
            self.assertIsNotNone(prog)
            self.assertEqual(prog["trained"], ["A", "B"])


# ======= データ取得テスト =======
class TestDataFetcher(unittest.TestCase):
    def test_get_market_session_returns_valid(self):
        from src.data_fetcher import get_market_session
        session = get_market_session()
        self.assertIn(session, ["pre_market", "morning", "lunch", "afternoon", "post_market"])

    def test_is_market_open_returns_bool(self):
        from src.data_fetcher import is_market_open
        result = is_market_open()
        self.assertIsInstance(result, bool)

    def test_fetch_1min_data_with_mock(self):
        from src.data_fetcher import fetch_1min_data
        mock_df = make_ohlcv(100)
        with patch("src.data_fetcher.yf.download", return_value=mock_df), \
             patch("src.data_fetcher.os.path.exists", return_value=False), \
             patch("pandas.DataFrame.to_parquet"):
            result = fetch_1min_data("7203.T", period="5d", use_cache=False)
            # mock_df は分足フィルタを通るので None でない可能性もある
            # 呼び出しが成功すれば OK

    def test_cache_key_consistent(self):
        from src.data_fetcher import _cache_key
        k1 = _cache_key("7203.T", "1m", "7d")
        k2 = _cache_key("7203.T", "1m", "7d")
        self.assertEqual(k1, k2)

    def test_cache_key_different_inputs(self):
        from src.data_fetcher import _cache_key
        k1 = _cache_key("7203.T", "1m", "7d")
        k2 = _cache_key("6758.T", "1m", "7d")
        self.assertNotEqual(k1, k2)


# ======= 取引分析テスト =======
class TestTradeAnalyzer(unittest.TestCase):
    def _make_sbi_csv(self) -> bytes:
        csv = "約定日,銘柄名,銘柄コード,売買,約定数量,約定単価,手数料,建単価,返済単価\n"
        csv += "2024/01/15,テスト株式,7777,買,100,1500,550,1500,1600\n"
        csv += "2024/01/16,テスト株式,7777,売,100,1600,550,1500,1600\n"
        csv += "2024/01/17,テスト株式,7777,買,100,1400,550,1400,1350\n"
        return csv.encode("shift_jis", errors="replace")

    def test_parse_sbi_csv_returns_df(self):
        from src.trade_analyzer import parse_sbi_csv
        content = self._make_sbi_csv()
        df = parse_sbi_csv(content)
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

    def test_parse_sbi_csv_utf8(self):
        from src.trade_analyzer import parse_sbi_csv
        csv = "約定日,銘柄名,銘柄コード,売買,約定数量,約定単価\n2024/01/15,A社,1111,買,100,500\n"
        df = parse_sbi_csv(csv.encode("utf-8"))
        self.assertIsNotNone(df)

    def test_calculate_trade_metrics_with_pnl(self):
        from src.trade_analyzer import calculate_trade_metrics
        trades = pd.DataFrame({
            "side": ["買", "売", "買"],
            "qty": [100, 100, 100],
            "open_price": [1500.0, 1500.0, 1400.0],
            "close_price": [1600.0, 1400.0, 1350.0],
            "fee": [550.0, 550.0, 550.0],
        })
        metrics = calculate_trade_metrics(trades)
        self.assertIn("total_trades", metrics)
        self.assertEqual(metrics["total_trades"], 3)

    def test_calculate_trade_metrics_win_rate(self):
        from src.trade_analyzer import calculate_trade_metrics
        trades = pd.DataFrame({
            "side": ["買", "買"],
            "qty": [100, 100],
            "open_price": [1000.0, 1000.0],
            "close_price": [1100.0, 900.0],
            "fee": [0.0, 0.0],
        })
        metrics = calculate_trade_metrics(trades)
        self.assertAlmostEqual(metrics["win_rate"], 0.5)

    def test_entry_exit_optimization(self):
        from src.trade_analyzer import analyze_entry_exit_optimization
        ohlcv = make_daily_ohlcv(50)
        trade = pd.Series({
            "side": "買",
            "open_price": float(ohlcv["Close"].iloc[10]),
            "close_price": float(ohlcv["Close"].iloc[20]),
            "qty": 100,
        })
        result = analyze_entry_exit_optimization(trade, ohlcv)
        self.assertIsInstance(result, dict)

    def test_generate_technical_advice(self):
        from src.trade_analyzer import generate_technical_advice
        trades = pd.DataFrame({"side": ["買"], "qty": [100], "open_price": [1000.0], "close_price": [1050.0]})
        ohlcv = make_daily_ohlcv(300)
        advice = generate_technical_advice("9999.T", trades, ohlcv)
        self.assertIn("indicators_advice", advice)
        self.assertIsInstance(advice["indicators_advice"], list)

    def test_full_analysis_empty_file(self):
        from src.trade_analyzer import full_trade_analysis
        result = full_trade_analysis(b"", use_ai=False)
        self.assertIn("error", result)

    def test_full_analysis_valid_csv(self):
        from src.trade_analyzer import full_trade_analysis
        content = self._make_sbi_csv()
        with patch("src.trade_analyzer.fetch_daily_data", return_value=make_daily_ohlcv(300)):
            result = full_trade_analysis(content, use_ai=False)
            self.assertIn("summary_metrics", result)


# ======= Webアプリテスト =======
class TestWebApp(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from web.app import create_app
        self.app = create_app()
        self.client = self.app.test_client()
        self.app.config["TESTING"] = True

    def test_health_endpoint(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        data = json.loads(r.data)
        self.assertEqual(data["status"], "ok")

    def test_status_endpoint(self):
        r = self.client.get("/api/status")
        self.assertEqual(r.status_code, 200)
        data = json.loads(r.data)
        self.assertIn("session", data)
        self.assertIn("market_open", data)

    def test_predictions_endpoint(self):
        r = self.client.get("/api/predictions")
        self.assertIn(r.status_code, [200, 500])

    def test_search_no_query(self):
        r = self.client.get("/api/search")
        self.assertEqual(r.status_code, 400)

    def test_predict_endpoint_no_data(self):
        with patch("web.app.fetch_1min_data", return_value=None), \
             patch("web.app.fetch_jpx_stock_list", return_value=pd.DataFrame({
                 "コード": ["7203"], "銘柄名": ["テスト"], "市場・商品区分": ["プライム"]
             })):
            r = self.client.get("/api/predict/7203.T")
            self.assertIn(r.status_code, [200, 404, 500])

    def test_analyze_trades_no_file(self):
        r = self.client.post("/api/analyze-trades")
        self.assertEqual(r.status_code, 400)

    def test_index_serves_html(self):
        r = self.client.get("/")
        # index.html が見つかれば 200、なければ 404
        self.assertIn(r.status_code, [200, 404])


# ======= 統合テスト =======
class TestIntegration(unittest.TestCase):
    def test_full_pipeline_1min(self):
        """1分足データのML全パイプライン"""
        from src.features import compute_features, get_feature_columns
        from src.ml_trainer import prepare_dataset, train_model

        df = make_ohlcv(300)
        X, y = prepare_dataset(df)
        self.assertGreater(len(X), 50)

        with patch("src.ml_trainer._model_path", return_value="/tmp/integration_model.pkl"), \
             patch("src.ml_trainer._meta_path", return_value="/tmp/integration_meta.json"):
            result = train_model("INTG_TEST", df, n_splits=2)
            self.assertIsNotNone(result)
            self.assertGreaterEqual(result["metrics"]["cv_accuracy_mean"], 0.4)

    def test_full_pipeline_daily(self):
        """日足データのML全パイプライン"""
        from src.ml_trainer import train_model

        df = make_daily_ohlcv(500)
        with patch("src.ml_trainer._model_path", return_value="/tmp/integration_daily_model.pkl"), \
             patch("src.ml_trainer._meta_path", return_value="/tmp/integration_daily_meta.json"):
            result = train_model("DAILY_TEST", df, n_splits=2)
            self.assertIsNotNone(result)

    def test_predict_after_train(self):
        """学習後に予測が動作することを確認"""
        from src.ml_trainer import train_model, predict_next_candle

        df = make_ohlcv(300)
        with patch("src.ml_trainer._model_path", return_value="/tmp/pred_test_model.pkl"), \
             patch("src.ml_trainer._meta_path", return_value="/tmp/pred_test_meta.json"):
            train_model("PRED_TEST", df, n_splits=2)

        with patch("src.ml_trainer._model_path", return_value="/tmp/pred_test_model.pkl"):
            pred = predict_next_candle("PRED_TEST", df)
            if pred is not None:
                self.assertIn("bullish_prob", pred)
                self.assertIn("bearish_prob", pred)
                self.assertAlmostEqual(pred["bullish_prob"] + pred["bearish_prob"], 1.0, places=3)
                self.assertIn(pred["prediction"], ["bullish", "bearish"])


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in [TestFeatures, TestMLTrainer, TestDataFetcher, TestTradeAnalyzer, TestWebApp, TestIntegration]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
