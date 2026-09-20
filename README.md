# jreit_score — J-REIT 統合指標プロトタイプ（MIMIC型潜在因子モデル）

## 構造
- `jreit_score/features.py` 価格・分配金・10年債利回りのパネルから6本の目的変数を作る
  - `ret_6m`, `ret_12m`（将来トータルリターン）
  - `dpu_stab`（-CV）, `dpu_growth`（直近6期の対数成長）
  - `rate_resil`（-金利ベータ）, `dd_resil`（-利上げ週のドローダウン）
  - 全て「高いほど良い」向き
- `jreit_score/model.py` semopy で `quality =~ 6指標`, `quality ~ 財務指標` を推定。
  運用スコアは構造方程式のみ `η̂ = β̂·X`（将来情報を使わない）
- `jreit_score/validation.py` 時系列分割。目的変数が実現済みの期だけで学習し、
  Spearman IC と五分位スプレッドを Newey–West t 値で評価
- `jreit_score/synthetic.py` 実データが揃うまでの合成パネル
- `run_local.py` 合成データで end-to-end を回す（`--opposite` で安定性が逆向きに載るケース）

## データ取得（プロトタイプ）
- `jreit_score/ingest/japan_reit.py` JAPAN-REIT.COM の銘柄ランキング（全銘柄×11指標）を
  日付付き Parquet として蓄積する。サイトは現在値しか出さないため定期実行で履歴を自作する
  （`data/japan_reit_ranking/asof=YYYY-MM-DD/part.parquet`）
- `jreit_score/ingest/dpu_history.py` 銘柄別 DPU 履歴。取得元ページの表構造は未確認のため
  「決算期」「分配金」を含む表を自動検出する汎用パーサ。まず `--inspect` で表のヘッダを確認する
- `jreit_score/panel.py` スナップショットを period ごとの説明変数（`PROTO_CAUSES`）に整形
- `tests/test_ingest.py` ネットワーク無しでパーサを検証するフィクスチャ

```
python -m jreit_score.ingest.japan_reit --out data
python -m jreit_score.ingest.dpu_history 8985 8951 --source japan_reit --inspect
python -m jreit_score.ingest.dpu_history 8985 8951 --source japan_reit --out data
PYTHONPATH=. python tests/test_ingest.py
```

利用規約: JAPAN-REIT.COM は転載・複製を禁じている。個人利用のプロトタイプに限定し、
アクセス間隔を空け（2秒以上）、取得データはリポジトリにコミットしない（`data/` は .gitignore）。
公開段階では一次情報（TDnet/EDINET）に切り替える。

## 実行
```
pip install -r requirements.txt
python run_local.py
python run_local.py --opposite
```

## 実データ接続時に差し替える箇所
- `features.build_outcomes(prices, dpu, jgb10, periods)` に long 形式の DataFrame を渡す
- 財務指標（`DEFAULT_CAUSES`）は `code, period` をキーに外部で結合する
- 期ごとに `cross_sectional_standardize` を掛けてから `fit_mimic` / `rolling_validation`

## 設計上の注意
- 6指標は 1因子モデルの適合度検定（自由度>0）のために目的3つを各2本に分割している
- 適合度が良くても負荷量の符号が割れる場合は「1因子で統合できない」と判断し、2因子に落とす。
  適合度指標だけでは符号の割れを検出できない（合成データで確認済み）
- 合成データの IC は信号を強めに作っているため参考値にならない。実データでは 0.05〜0.15 程度を想定
