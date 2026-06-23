# JP Stock ML - 日本株ML予測システム

機械学習で日本株の次の1分足が**陽線か陰線か**を予測するシステムです。
GitHub Actions上で完全無料・自動動作し、スマホブラウザから閲覧できます。

---

## 機能一覧

| 機能 | 説明 |
|------|------|
| 🔍 銘柄予測 | 銘柄を検索して次の1分後の方向性（陽線/陰線確率）をリアルタイム表示 |
| 📊 予測ランキング | 全対象銘柄の陽線確率ランキング表示 |
| 🤖 ML自動学習 | 前場・後場中は1分足で学習、開場前・閉場後は日足で学習 |
| 📋 取引履歴分析 | SBI証券CSVをアップロードしてエントリー・決済位置のRR改善提案 |
| 💡 AIアドバイス | Anthropic Claude APIで取引データの包括的な改善アドバイス |

---

## アーキテクチャ

```
GitHub Actions（無料・公開リポジトリ）
├── ml_train.yml     # スケジュール実行（前場/後場/閉場後）
├── training.yml     # 歴史的学習継続（6時間制限チェーン）
├── test.yml         # push時テスト
└── web_server.yml   # Webサーバー起動

データフロー:
JPX公式XLS → 銘柄フィルタ → yfinance 1分足/日足
→ テクニカル指標計算 → LightGBM学習 → 予測 → Web表示
```

---

## ファイル構成

```
/jp-stock-ml/
├── /jp-stock-ml/main.py                          # メインオーケストレーター
├── /jp-stock-ml/requirements.txt                 # 依存パッケージ
├── /jp-stock-ml/.gitignore
│
├── /jp-stock-ml/src/
│   ├── /jp-stock-ml/src/__init__.py
│   ├── /jp-stock-ml/src/stock_list.py            # JPX全上場銘柄取得・フィルタ
│   ├── /jp-stock-ml/src/data_fetcher.py          # yfinanceデータ取得（レート制限対策）
│   ├── /jp-stock-ml/src/features.py              # テクニカル指標・特徴量エンジニアリング
│   ├── /jp-stock-ml/src/ml_trainer.py            # LightGBM学習・予測・進捗管理
│   └── /jp-stock-ml/src/trade_analyzer.py        # SBI証券CSV分析・RR改善提案
│
├── /jp-stock-ml/web/
│   ├── /jp-stock-ml/web/__init__.py
│   ├── /jp-stock-ml/web/app.py                   # Flask APIサーバー
│   └── /jp-stock-ml/web/index.html               # スマホ対応UI
│
├── /jp-stock-ml/tests/
│   ├── /jp-stock-ml/tests/__init__.py
│   └── /jp-stock-ml/tests/test_all.py            # 全モジュールテスト
│
├── /jp-stock-ml/.github/workflows/
│   ├── /jp-stock-ml/.github/workflows/ml_train.yml    # メイン学習ワークフロー
│   ├── /jp-stock-ml/.github/workflows/training.yml    # 歴史的学習継続ワークフロー
│   ├── /jp-stock-ml/.github/workflows/test.yml        # テストワークフロー
│   └── /jp-stock-ml/.github/workflows/web_server.yml  # Webサーバーワークフロー
│
├── /jp-stock-ml/models/                          # 学習済みモデル（.gitignore対象）
│   └── /jp-stock-ml/models/.gitkeep
├── /jp-stock-ml/data/cache/                      # キャッシュデータ（.gitignore対象）
│   └── /jp-stock-ml/data/cache/.gitkeep
└── /jp-stock-ml/logs/                            # ログ（.gitignore対象）
    └── /jp-stock-ml/logs/.gitkeep
```

---

## セットアップ手順（スマホのみ）

### 1. リポジトリを作成・プッシュ

GitHubアプリ（またはブラウザ）で新しいリポジトリを作成します。  
**⚠️ 必ず Public（公開）にしてください**（無料・無制限のGitHub Actionsのため）

```bash
# PCがある場合（スマホのみの場合はGitHub Webエディタで各ファイルを貼り付け）
git clone https://github.com/あなたのユーザー名/jp-stock-ml.git
cd jp-stock-ml
# ファイルをコピー
git add .
git commit -m "initial commit"
git push origin main
```

### 2. シークレット設定（任意・AI分析に必要）

GitHub → リポジトリ → Settings → Secrets and variables → Actions → New repository secret

| シークレット名 | 値 | 用途 |
|---|---|---|
| `ANTHROPIC_API_KEY` | AnthropicのAPIキー | 取引分析のAIアドバイス（任意） |

※ APIキーなしでも機械学習予測・取引分析の基本機能は動作します

### 3. GitHub Actions を有効化

リポジトリ → Actions タブ → "I understand my workflows, go ahead and enable them" をクリック

### 4. 初回実行

Actions → `JP Stock ML - Training & Prediction` → `Run workflow` → `Run workflow`

---

## 動作スケジュール（JST）

| 時刻 | 動作 |
|------|------|
| 08:30 | 開場前ML学習（日足）/ 歴史的学習チェック |
| 09:05 | 前場開始 → 1分足学習 + 予測更新 |
| 10:30 | 前場中 → 予測更新 |
| 12:35 | 後場開始 → 1分足学習 + 予測更新 |
| 14:00 | 後場中 → 予測更新 |
| 16:00 | 閉場後ML学習（日足）|

---

## 6時間制限の対応

GitHub Actions は1ジョブ最大6時間（360分）です。本システムでは以下の対策を取っています：

1. **各ジョブを最大350分**に制限し、6時間超え前に安全停止
2. **進捗をJSON保存**（`/jp-stock-ml/models/training_progress.json`）
3. **キャッシュで引き継ぎ**（`actions/cache`でモデルと進捗を保持）
4. **自動チェーン**：歴史的学習未完了なら自動で `training.yml` を起動
5. **再開機能**：`load_progress()` で前回の続きから学習を再開

---

## 銘柄フィルタ条件

低位株を除外し、デイトレ・スイング可能な銘柄のみを対象とします：

| 条件 | デイトレ | スイング |
|------|----------|----------|
| 最低株価 | 300円以上 | 300円以上 |
| 最低出来高 | 500,000株/日以上 | 100,000株/日以上 |

銘柄リストはJPX公式XLS（`https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls`）から動的取得します。

---

## 機械学習モデル詳細

### アルゴリズム
- **主**: LightGBM（`LGBMClassifier`）
- **フォールバック**: scikit-learn `GradientBoostingClassifier`

### 特徴量（約30種）
- ローソク足特性（実体・ヒゲ・実体比率）
- リターン（1/3/5/10分）
- 移動平均乖離率（5/10/20/25/50期間）
- VWAP乖離率
- RSI（9/14期間）
- MACD（12/26/9）・ヒストグラム・クロス
- ボリンジャーバンド幅・位置
- ATR比率
- 出来高比率（5/20期間）
- 時間特徴量（時・分・前場/後場フラグ・寄り/引け付近）

### 評価方法
時系列クロスバリデーション（`TimeSeriesSplit`, n_splits=5）

### 歴史的学習について
- バブル崩壊後（1991年1月1日）から初回起動前日までの日足データを使用
- **一度完了すると繰り返さない**（`/jp-stock-ml/models/historical_training_done.json` でフラグ管理）
- yfinanceの1分足は最大7日間のみ取得可能なため、長期学習には日足を使用

---

## Webサーバーについて

本システムのWebサーバーはGitHub Actions上で動作します。  
GitHub ActionsはインターネットからアクセスできるURLを自動提供しないため、以下の方法でアクセスしてください：

### ローカルでの確認（Termius / Oracle Cloud）

```bash
# Oracle Cloud Ubuntu にSSH接続後
cd /path/to/jp-stock-ml
pip install -r requirements.txt
python main.py --mode serve
# → http://your-oracle-ip:5000 でアクセス可能
```

### ngrok経由（無料トンネル）

```bash
# GitHub Actions のステップに追加可能（workflow_dispatchで手動実行）
wget -q https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar xzf ngrok-v3-stable-linux-amd64.tgz
./ngrok http 5000 &
python main.py --mode serve
```

---

## 取引履歴分析の使い方

1. SBI証券 → 口座管理 → 取引履歴 → CSV出力
2. Web画面の「取引分析」タブ → CSVをアップロード
3. 以下の分析結果が表示されます：
   - **総損益・勝率・リスクリワード・PF**
   - **エントリー・決済の最適ポイント比較**（期間高値・安値、テクニカル指標）
   - **有効なテクニカル指標の提案**（移動平均、RSI、MACD、BB）
   - **AIアドバイス**（ANTHROPIC_API_KEY設定時）

---

## ローカル実行（Oracle Cloud Ubuntu）

```bash
# 1. リポジトリをクローン
git clone https://github.com/あなたのユーザー名/jp-stock-ml.git
cd /home/ubuntu/jp-stock-ml

# 2. 依存パッケージインストール
pip3 install -r /home/ubuntu/jp-stock-ml/requirements.txt

# 3. 環境変数設定（任意）
export ANTHROPIC_API_KEY="your-key-here"

# 4. 自動モードで実行（市場時間に応じて自動切替）
python3 /home/ubuntu/jp-stock-ml/main.py --mode auto

# 5. Webサーバー起動
python3 /home/ubuntu/jp-stock-ml/main.py --mode serve

# 6. テスト実行
python3 -m pytest /home/ubuntu/jp-stock-ml/tests/test_all.py -v
```

### cron設定例（Oracle Cloud）

```bash
crontab -e

# 開場前
30 23 * * 0-4 cd /home/ubuntu/jp-stock-ml && python3 main.py --mode auto >> /home/ubuntu/jp-stock-ml/logs/cron.log 2>&1
# 前場中
5,30 0,1 * * 1-5 cd /home/ubuntu/jp-stock-ml && python3 main.py --mode predict >> /home/ubuntu/jp-stock-ml/logs/cron.log 2>&1
# 後場中
35 3,4,5 * * 1-5 cd /home/ubuntu/jp-stock-ml && python3 main.py --mode predict >> /home/ubuntu/jp-stock-ml/logs/cron.log 2>&1
```

---

## 注意事項

- **yfinance制限**: 1分足データは最大7日間のみ。過去の長期1分足データは取得不可のため、歴史的学習は日足データを使用します。
- **レート制限**: yfinanceのAPIレート制限を考慮し、リクエスト間に適切な待機時間を設けています。429エラー時は自動的に待機・リトライします。
- **免責事項**: 本システムの予測は参考情報です。投資判断はご自身の責任でお願いします。
- **データの遅延**: Yahoo Finance（yfinance）のデータは実際の株価より数分遅延する場合があります。
- **GitHub Actionsの制限**: 公開リポジトリのみ無料・無制限。プライベートリポジトリは月2,000分の制限あり。

---

## ライセンス

MIT License - 自由に使用・改変できます。
