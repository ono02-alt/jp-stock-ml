"""
機械学習トレーニングモジュール
LightGBMを使って次の1分後が陽線か陰線かを予測する
"""
import os
import json
import time
import logging
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_GBM = True
except ImportError:
    HAS_GBM = False

from .features import compute_features, get_feature_columns

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PROGRESS_FILE = os.path.join(MODELS_DIR, "training_progress.json")
os.makedirs(MODELS_DIR, exist_ok=True)


def _model_path(ticker: str) -> str:
    safe = ticker.replace(".", "_")
    return os.path.join(MODELS_DIR, f"{safe}_model.pkl")


def _meta_path(ticker: str) -> str:
    safe = ticker.replace(".", "_")
    return os.path.join(MODELS_DIR, f"{safe}_meta.json")


def save_progress(trained: List[str], total: List[str], mode: str) -> None:
    progress = {
        "trained": trained,
        "total": total,
        "mode": mode,
        "last_updated": datetime.now().isoformat(),
        "completion_pct": len(trained) / max(len(total), 1) * 100,
    }
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def load_progress() -> Optional[dict]:
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


def prepare_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """特徴量計算してX, yを返す"""
    feat_df = compute_features(df)
    feat_cols = get_feature_columns()

    # 時間特徴量があれば追加
    time_cols = [c for c in ["hour", "minute", "time_in_session", "is_morning", "is_afternoon",
                               "near_open", "near_close"] if c in feat_df.columns]
    use_cols = feat_cols + time_cols

    available_cols = [c for c in use_cols if c in feat_df.columns]
    X = feat_df[available_cols].dropna()
    y = feat_df.loc[X.index, "target"]

    # 最後の行はターゲットが未来なので除去
    X = X.iloc[:-1]
    y = y.iloc[:-1]

    return X, y


def train_model(
    ticker: str,
    df: pd.DataFrame,
    n_splits: int = 5
) -> Optional[dict]:
    """
    単一銘柄のモデルを学習する

    Parameters
    ----------
    ticker : 銘柄コード
    df : 1分足 or 日足 DataFrame
    n_splits : 時系列クロスバリデーション分割数

    Returns
    -------
    dict with model, metrics, feature_importance
    """
    try:
        X, y = prepare_dataset(df)

        if len(X) < 100:
            logger.warning(f"{ticker}: データ不足 ({len(X)}行)")
            return None

        # 時系列分割CV
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []

        # 最後のfoldでモデル確定
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            if HAS_LGB:
                model = lgb.LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=6,
                    num_leaves=31,
                    min_child_samples=20,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    random_state=42,
                    verbose=-1,
                )
            else:
                model = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=4,
                    random_state=42,
                )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            score = accuracy_score(y_val, y_pred)
            cv_scores.append(score)

        # 全データで最終モデル学習
        if HAS_LGB:
            final_model = lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1,
            )
        else:
            final_model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                random_state=42,
            )

        final_model.fit(X, y)

        # 特徴量重要度
        feat_imp = {}
        if HAS_LGB and hasattr(final_model, "feature_importances_"):
            for col, imp in zip(X.columns, final_model.feature_importances_):
                feat_imp[col] = float(imp)
        elif hasattr(final_model, "feature_importances_"):
            for col, imp in zip(X.columns, final_model.feature_importances_):
                feat_imp[col] = float(imp)

        feat_imp_sorted = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:20])

        metrics = {
            "cv_accuracy_mean": float(np.mean(cv_scores)),
            "cv_accuracy_std": float(np.std(cv_scores)),
            "n_samples": len(X),
            "n_features": len(X.columns),
            "trained_at": datetime.now().isoformat(),
            "feature_columns": list(X.columns),
        }

        # モデル保存
        model_data = {
            "model": final_model,
            "feature_columns": list(X.columns),
            "metrics": metrics,
            "feature_importance": feat_imp_sorted,
        }
        with open(_model_path(ticker), "wb") as f:
            pickle.dump(model_data, f)

        # メタ情報保存
        with open(_meta_path(ticker), "w", encoding="utf-8") as f:
            json.dump({
                "ticker": ticker,
                "metrics": metrics,
                "feature_importance": feat_imp_sorted,
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"{ticker}: 学習完了 精度={np.mean(cv_scores):.3f}±{np.std(cv_scores):.3f}")
        return model_data

    except Exception as e:
        logger.error(f"{ticker}: 学習失敗: {e}", exc_info=True)
        return None


def load_model(ticker: str) -> Optional[dict]:
    """保存済みモデルを読み込む"""
    path = _model_path(ticker)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"{ticker}: モデル読み込み失敗: {e}")
        return None


def predict_next_candle(
    ticker: str,
    df: pd.DataFrame
) -> Optional[dict]:
    """
    次の1分足が陽線か陰線かを予測する

    Returns
    -------
    dict: {
        "ticker": str,
        "bullish_prob": float,  # 陽線確率 0-1
        "bearish_prob": float,  # 陰線確率 0-1
        "prediction": "bullish" | "bearish",
        "confidence": float,
        "model_accuracy": float,
    }
    """
    model_data = load_model(ticker)
    if model_data is None:
        # モデルがなければその場で学習
        logger.info(f"{ticker}: モデルなし、即時学習を実行")
        model_data = train_model(ticker, df)
        if model_data is None:
            return None

    model = model_data["model"]
    feature_cols = model_data["feature_columns"]

    try:
        feat_df = compute_features(df)
        available = [c for c in feature_cols if c in feat_df.columns]
        last_row = feat_df[available].dropna().iloc[-1:]

        if last_row.empty:
            return None

        proba = model.predict_proba(last_row)[0]
        bullish_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
        bearish_prob = 1.0 - bullish_prob

        return {
            "ticker": ticker,
            "bullish_prob": round(bullish_prob, 4),
            "bearish_prob": round(bearish_prob, 4),
            "prediction": "bullish" if bullish_prob > 0.5 else "bearish",
            "confidence": round(abs(bullish_prob - 0.5) * 2, 4),
            "model_accuracy": model_data["metrics"].get("cv_accuracy_mean", 0),
            "model_trained_at": model_data["metrics"].get("trained_at", ""),
            "feature_importance": model_data.get("feature_importance", {}),
        }

    except Exception as e:
        logger.error(f"{ticker}: 予測失敗: {e}")
        return None


def batch_train(
    tickers_and_data: Dict[str, pd.DataFrame],
    resume: bool = True,
    max_time_seconds: int = 20000  # 6時間制限の余裕を持たせた上限
) -> Dict[str, dict]:
    """
    複数銘柄を一括学習（進捗保存・再開機能付き）

    Parameters
    ----------
    tickers_and_data : {ticker: df}
    resume : 前回の続きから再開するか
    max_time_seconds : この時間を超えたら停止して進捗保存

    Returns
    -------
    {ticker: result_dict}
    """
    total_tickers = list(tickers_and_data.keys())
    trained = []
    results = {}

    # 再開処理
    if resume:
        progress = load_progress()
        if progress:
            previously_trained = set(progress.get("trained", []))
            trained = [t for t in total_tickers if t in previously_trained]
            logger.info(f"前回の進捗から再開: {len(trained)}/{len(total_tickers)}完了済み")
        else:
            previously_trained = set()
    else:
        previously_trained = set()

    start_time = time.time()

    for ticker in total_tickers:
        if ticker in previously_trained:
            continue

        elapsed = time.time() - start_time
        if elapsed > max_time_seconds:
            logger.warning(f"時間制限到達。{len(trained)}/{len(total_tickers)}完了で停止")
            save_progress(trained, total_tickers, "batch")
            break

        df = tickers_and_data.get(ticker)
        if df is None or df.empty:
            continue

        result = train_model(ticker, df)
        if result:
            results[ticker] = result
            trained.append(ticker)
            save_progress(trained, total_tickers, "batch")

    logger.info(f"バッチ学習完了: {len(trained)}/{len(total_tickers)}")
    return results
