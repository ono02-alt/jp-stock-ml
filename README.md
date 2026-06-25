# JP Stock ML - 日本株ML予測システム

機械学習で日本株の**次の1分足が陽線か陰線か**を予測するシステムです。  
GitHub Actions で完全自動動作し、**GitHub Pages** でスマホブラウザから閲覧できます。完全無料で運用できます。

---

## 機能

| 機能 | 説明 |
|---|---|
| 🔍 銘柄予測 | 銘柄を検索して次の1分後の陽線/陰線確率を表示 |
| 📊 予測ランキング | 全対象銘柄の陽線確率ランキング |
| 🤖 ML自動学習 | 前場・後場中は1分足、開場前・閉場後は日足で自動学習 |
| 📋 取引履歴分析 | SBI証券CSV をアップロードしてエントリー・決済の改善提案（ブラウザ内処理） |
| 💡 AIアドバイス | Anthropic Claude API で取引データの包括的な改善アドバイス（Oracle Cloud 版のみ） |

---

## アクセス方法

### GitHub Pages（メイン・完全無料）

```
https://あなたのユーザー名.github.io/jp-stock-ml/
```

- GitHub Actions が予測データを `docs/data/` に書き出し、自動で Pages に反映されます
- 静的ページのためサーバー不要・常時アクセス可能です

### Oracle Cloud Always Free（Flask 版・AIアドバイス利用時）

```
http://your-oracle-ip:5000
```

---

## ファイル構成

```
jp-stock-ml/
│
├── main.py                          メインオーケストレーター
├── requirements.txt                 依存パッケージ一覧
├── .gitignore
│
├── src/                             ロジック層
│   ├── stock_list.py                JPX公式XLSから全上場銘柄を動的取得・フィルタリング
│   ├── data_fetcher.py              yfinance データ取得（1分足/日足、CSVキャッシュ）
│   ├── features.py                  テクニカル指標34種の特徴量エンジニアリング
│   ├── ml_trainer.py                LightGBM 学習・予測・進捗保存・再開
│   └── trade_analyzer.py            SBI証券CSV分析・エントリー最適化・AIアドバイス
│
├── web/                             Web 層
│   ├── app.py                       Flask API サーバー（Oracle Cloud 用）
│   └── index.html                   スマホ対応 SPA（GitHub Pages / Flask 兼用）
│
├── docs/                            GitHub Pages 公開ディレクトリ ★
│   ├── .nojekyll                    Jekyll 処理スキップ用
│   ├── index.html                   web/index.html のコピー（Actions が自動更新）
│   └── data/
│       ├── predictions.json         ML予測結果（Actions が定期更新）
│       └── status.json              システム状態（Actions が定期更新）
│
├── .github/workflows/
│   ├── ml_train.yml                 メイン（学習・予測・docs/ 書き出し・Pages 反映）
│   └── training.yml                 歴史的学習継続用（6時間チェーン）
│
├── models/                          学習済みモデル（.gitignore / Actions cache 管理）
├── data/cache/                      データキャッシュ（.gitignore / Actions cache 管理）
└── logs/                            実行ログ（.gitignore / Actions cache 管理）
```

---

## セットアップ（スマホのみ）

### 1. GitHub にリポジトリを作成

⚠️ **必ず Public（公開）にしてください**（無料 GitHub Actions・GitHub Pages のため）

### 2. ファイルをアップロード

GitHub の Web エディタでファイルを貼り付けます（または git push）。

### 3. GitHub Pages を有効化

リポジトリ → Settings → Pages → Source: **Deploy from a branch** → Branch: `main` / Folder: **`/docs`** → Save

### 4. シークレットを設定（任意）

Settings → Secrets and variables → Actions → New repository secret

| シークレット名 | 内容 | 用途 |
|---|---|---|
| `ANTHROPIC_API_KEY` | Anthropic の API キー | 取引分析の AI アドバイス（Oracle Cloud 版のみ） |

### 5. GitHub Actions を有効化

Actions タブ → 「I understand my workflows, go ahead and enable them」をクリック

### 6. 初回実行

Actions → `JP Stock ML - Training & Prediction` → `Run workflow` → `Run workflow`

実行後、`docs/data/predictions.json` が更新され、GitHub Pages に自動反映されます。

---

## 動作スケジュール（JST）

| 時刻 | 動作 |
|---|---|
| 08:30 | 開場前：歴史的学習チェック（初回のみ） |
| 09:05 | 前場開始：1分足学習 + 予測更新 + Pages 反映 |
| 10:30 | 前場後半：予測更新 + Pages 反映 |
| 12:35 | 後場開始：1分足学習 + 予測更新 + Pages 反映 |
| 14:00 | 後場後半：予測更新 + Pages 反映 |
| 16:00 | 閉場後：歴史的学習チェック |

---

## 6時間制限への対応

GitHub Actions は1ジョブ最大6時間（360分）です。以下の仕組みで対処します。

- 各ジョブを **350分上限**で停止
- 学習進捗を `models/training_progress.json` に都度保存
- `actions/cache` でモデル・進捗をジョブ間で引き継ぎ
- 歴史的学習未完了なら `training.yml` を自動起動してチェーン継続
- 次回起動時は前回の続きから再開

---

## 銘柄フィルタ条件

低位株を除外し、デイトレ・スイング可能な銘柄のみを対象とします。

| 条件 | デイトレ | スイング |
|---|---|---|
| 最低株価 | 300円以上 | 300円以上 |
| 最低出来高 | 500,000株/日以上 | 100,000株/日以上 |

銘柄リストは JPX 公式 XLS から動的取得します（12時間キャッシュ）。

---

## 機械学習モデル

### アルゴリズム
- **主**: LightGBM 4.x（`LGBMClassifier`）
- **フォールバック**: scikit-learn `GradientBoostingClassifier`

### 依存パッケージ（2026年6月時点）

| パッケージ | バージョン | 用途 |
|---|---|---|
| yfinance | >=1.0.0（最新: 1.4.1） | 株価データ取得 |
| lightgbm | >=4.0.0 | ML メイン |
| pandas-ta-openbb | >=0.4.20（最新: 0.4.24） | テクニカル指標（Python 3.11+ 対応版） |
| scikit-learn | >=1.3.0 | ML ユーティリティ・フォールバック |
| flask | >=3.0.0 | Web サーバー（Oracle Cloud 版のみ） |

> ⚠️ 旧来の `pandas-ta>=0.3.14b` は Python 3.11 非対応です。  
> データキャッシュは **CSV 形式**を使用します（pyarrow / fastparquet 不要）。

### 特徴量（34種）

- ローソク足特性（実体・ヒゲ・実体比率・陽線フラグ）
- リターン（1/3/5/10 分）
- 移動平均乖離率（5/10/20/25/50 期間）
- VWAP 乖離率 / RSI（9/14 期間）
- MACD（12/26/9）・ヒストグラム・クロス
- ボリンジャーバンド幅・位置 / ATR 比率
- 出来高比率（5/20 期間）
- 時間特徴量（時・分・前場/後場フラグ・寄り/引け付近）

### 評価方法

時系列クロスバリデーション（`TimeSeriesSplit`, n_splits=5）

### 歴史的学習について

- バブル崩壊後（1991年1月1日）〜 初回起動前日の日足データを使用
- **一度完了すると繰り返さない**（`models/historical_training_done.json` でフラグ管理）

---

## Oracle Cloud での Flask サーバー起動

```bash
# リポジトリをクローン
git clone https://github.com/あなたのユーザー名/jp-stock-ml.git
cd /home/ubuntu/jp-stock-ml

# 依存パッケージインストール
pip3 install -r /home/ubuntu/jp-stock-ml/requirements.txt

# 環境変数設定（任意）
export ANTHROPIC_API_KEY="your-key-here"

# Flask サーバー起動
python3 /home/ubuntu/jp-stock-ml/main.py --mode serve
# → http://your-oracle-ip:5000 でアクセス可能
```

---

## 注意事項

- **GitHub Pages**: 静的ファイルのみ配信。Flask は動作しません。本プロジェクトは予測 JSON を事前生成して対応しています。
- **yfinance**: 1分足データは最大7日間のみ。歴史的学習には日足を使用します。
- **データキャッシュ**: pyarrow / fastparquet は不要です（CSV 形式）。
- **データ遅延**: Yahoo Finance のデータは実際の株価より数分遅延する場合があります。
- **免責事項**: 本システムの予測は参考情報です。投資判断はご自身の責任でお願いします。
- **GitHub Actions**: 公開リポジトリのみ無料・無制限。プライベートリポジトリは月2,000分の制限あり。

---

## ライセンス

MIT License
