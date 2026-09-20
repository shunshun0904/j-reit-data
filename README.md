# jreit_score — J-REIT 統合指標プロトタイプ（MIMIC型潜在因子モデル）

## 構造
- `jreit_score/features.py` 価格・分配金・10年債利回りのパネルから6本の目的変数を作る
  - `ret_6m`, `ret_12m`（将来トータルリターン）
  - `dpu_stab`（-CV）, `dpu_growth`（直近6期の対数成長）
  - `rate_resil`（-金利ベータ）, `dd_resil`（-利上げ週のドローダウン）
  - 全て「高いほど良い」向き
- `jreit_score/model.py` semopy で `quality =~ 6指標`, `quality ~ 財務指標` を推定。
  運用スコアは構造方程式のみ `η̂ = β̂·X`（将来情報を使わない）
  - `fit_with_sign_branch` が符号割れを判定し、割れていれば2因子モデルに落とす。
    判定に適合度は使わない（適合度は割れていても良いままになる）
- `jreit_score/validation.py` 時系列分割。目的変数が実現済みの期だけで学習し、
  Spearman IC と五分位スプレッドを Newey–West t 値で評価。
  `branch=True` で期ごとに符号判定を行い、2因子になった期は因子ごとに列を分ける
- `jreit_score/synthetic.py` 実データが揃うまでの合成パネル
- `run_local.py` 合成データで end-to-end を回す（`--opposite` で安定性が逆向きに載るケース）

## データ取得（プロトタイプ）
- `jreit_score/ingest/japan_reit.py` JAPAN-REIT.COM の銘柄ランキング（全銘柄×11指標）を
  日付付き Parquet として蓄積する。サイトは現在値しか出さないため定期実行で履歴を自作する
  （`data/japan_reit_ranking/asof=YYYY-MM-DD/part.parquet`）
- `jreit_score/ingest/dpu_history.py` 銘柄別 DPU 履歴。取得元ページの表構造は未確認のため
  「決算期」「分配金」を含む表を自動検出する汎用パーサ。まず `--inspect` で表のヘッダを確認する
- `jreit_score/ingest/jgb.py` 財務省「国債金利情報」CSV から国債利回りを取得する。
  和暦（`S49.9.24` / `令和6年4月1日`）と西暦の両方、全角の年限列、欠損記号 `-` を吸収する。
  - 当月分 `jgbcm.csv` / 過去分 `data/jgbcm_all.csv`（1974-09-24〜、13,290行、年限15本）
  - 過去分は当月を含まないため `fetch_jgb10_full()` が両方を結合して
    `[date, yield]` を返す（`features.build_outcomes` の `jgb10` 入力）
  - 取得先は `--list` で一覧ページから確認する。URL を推測しない
- `jreit_score/panel.py` スナップショットを period ごとの説明変数（`PROTO_CAUSES`）に整形
- `tests/test_ingest.py` ネットワーク無しでパーサを検証するフィクスチャ

```
python -m jreit_score.ingest.japan_reit --out data
python -m jreit_score.ingest.dpu_history 8985 8951 --source japan_reit --inspect
python -m jreit_score.ingest.dpu_history 8985 8951 --source japan_reit --out data
python -m jreit_score.ingest.jgb --list              # 取得先の一覧
python -m jreit_score.ingest.jgb --source all --inspect
python -m jreit_score.ingest.jgb --source full --out data   # all + current を結合
PYTHONPATH=. python tests/test_ingest.py
PYTHONPATH=. python tests/test_jgb.py
```

開発セッションから japan-reit.com / api.jquants.com / mof.go.jp に到達できない場合は、
`.github/workflows/inspect-sources.yml` / `inspect-jgb.yml` を手動実行して構造を確認する。

利用規約: JAPAN-REIT.COM は転載・複製を禁じている。個人利用のプロトタイプに限定し、
アクセス間隔を空け（2秒以上）、取得データはリポジトリにコミットしない（`data/` は .gitignore）。
公開段階では一次情報（TDnet/EDINET）に切り替える。

## 実行
```
pip install -r requirements.txt
python run_local.py            # 6指標が同じ向き → 1因子で統合
python run_local.py --opposite # 安定性2指標が逆向き → 2因子へ分岐
PYTHONPATH=. python tests/test_ingest.py
PYTHONPATH=. python tests/test_model.py
```

## 実データ接続時に差し替える箇所
- `features.build_outcomes(prices, dpu, jgb10, periods)` に long 形式の DataFrame を渡す
- 財務指標（`DEFAULT_CAUSES`）は `code, period` をキーに外部で結合する
- 期ごとに `cross_sectional_standardize` を掛けてから `fit_with_sign_branch` /
  `rolling_validation(..., branch=True)`

## 設計上の注意
- 6指標は 1因子モデルの適合度検定（自由度>0）のために目的3つを各2本に分割している
- 適合度が良くても負荷量の符号が割れる場合は「1因子で統合できない」と判断し、2因子に落とす。
  適合度指標だけでは符号の割れを検出できない（合成データで確認済み）
- 符号判定は標準化負荷量 |λ|>=0.15 かつ p<0.05 の指標だけで行う。弱い負荷量や非有意な
  負荷量の符号はノイズで反転しうるため、これらを判定に入れると過剰に2因子へ落ちる
- 2因子仕様には残差共分散 `q1 ~~ q2` が必須。semopy は内生潜在変数どうしの共分散を
  自動追加しないため、省くと2因子が独立と制約され適合が崩れる
  （合成データ `--opposite`: 省略時 chi2 p=0.000 / CFI=0.895 / RMSEA=0.048、
    明示時 chi2 p=0.996 / CFI=1.010 / RMSEA=0.000）
- 潜在因子の符号は識別上任意なので、各因子は負荷量の合計が正になる向きに揃えている
- 2因子に落ちた時点で1本のスコアに束ねる根拠が無いため、合成スコアは作らない
- 合成データの IC は信号を強めに作っているため参考値にならない。実データでは 0.05〜0.15 程度を想定
