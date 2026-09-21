# jreit_score — J-REIT 目的別スコア（MIMIC型潜在因子モデル）

3つの目的（将来リターン・分配金の安定性/成長・金利上昇耐性）の共通因子を推定して統合スコアを作る
計画だったが、実データでは目的をまたぐ共通因子が確認できなかった（6指標の目的間相関はすべて 0.10 以下）。
そのため統合スコアは作らず、目的別のスコアを推定して並べる（2026-09-21 決定。経緯は CLAUDE.md）。

## 構造
- `jreit_score/features.py` 価格・分配金・10年債利回りのパネルから6本の目的変数を作る
  - `ret_6m`, `ret_12m`（将来トータルリターン）
  - `dpu_stab`（-CV）, `dpu_growth`（直近6期の対数成長）
  - `rate_resil`（-金利ベータ）, `dd_resil`（-利上げ週のドローダウン）
  - 全て「高いほど良い」向き
- `jreit_score/model.py` semopy による MIMIC の推定。運用スコアは構造方程式のみ `η̂ = β̂·X`
  （将来情報を使わない）
  - `fit_objective_factors` が目的ごとに別々に推定する（`features.OBJECTIVES`）。
    `q_ret =~ ret_6m + ret_12m` と `q_rate =~ rate_resil + dd_resil` は 2 指標の 1 因子 MIMIC、
    `q_stab`（dpu_stab）と `q_grow`（dpu_growth）は 1 指標なので説明変数への回帰。
    目的ごとに `ok / sign_split / weak（識別不能）/ improper / no_signal / fit_failed` を判定し、
    `ok` 以外はスコアを出さない
  - `fit_objective_model_joint` は全目的の同時推定（適合度の参考のみ）
  - `fit_with_sign_branch` は 1 因子の符号割れ判定と 2 因子への分岐。合成データの検証用に残している。
    符号が割れないことと共通因子があることは別で、実データでは後者が成り立たなかった
- `jreit_score/validation.py` 時系列分割。目的変数が実現済みの期だけで学習し、
  Spearman IC と五分位スプレッドを Newey–West t 値で評価。
  `objectives=OBJECTIVES` で目的ごとに評価し、期ごとの判定を `status_<目的>` に残す
- `jreit_score/run_real.py` 実データ（J-Quants store + 財務省 10 年債）での推定と診断
- `jreit_score/publish.py` 公開用の目的別スコア payload（銘柄別 z 値・五分位、係数、検証 IC。派生値のみ）
- `jreit_score/site.py` / `site/index.html` 公開ダッシュボード（GitHub Pages）
- `jreit_score/synthetic.py` 合成パネル
- `run_local.py` 合成データで end-to-end を回す（`--objectives` で目的別、`--opposite` で符号割れのケース）

## データ取得（プロトタイプ）
- `jreit_score/ingest/japan_reit.py` JAPAN-REIT.COM の銘柄ランキング（全銘柄×11指標）を
  日付付き Parquet として蓄積する。サイトは現在値しか出さないため定期実行で履歴を自作する
  （`data/japan_reit_ranking/asof=YYYY-MM-DD/part.parquet`）
- `jreit_score/ingest/dpu_history.py` 銘柄別 DPU 履歴。**DPU 履歴の取得には使えない**。
  JAPAN-REIT.COM の銘柄ページは「前期/当期/次期」の3期分しか持たず（次期は予想）、
  haitoukabu.com も履歴を持たないことを Actions で確認した（2026-09-20）。
  参考として残すが、DPU は `jquants.py` から取る
- `jreit_score/ingest/jquants.py` J-Quants V2 から価格（`/equities/bars/daily`）と決算短信サマリ
  （`/fins/summary`。DPU・BPS）を取得する。`jquants_store.py` が Actions cache に差分保存し、
  `jquants_panel.py` が説明変数（NAV 倍率・対数時価総額）を as-of で作る
  （API キーは GitHub Secrets の `JQUANTS_API_KEY` または `JQUANTS_API`）
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
python -m jreit_score.ingest.dpu_history 8985 --source japan_reit --inspect  # 構造確認のみ
JQUANTS_API_KEY=... python -m jreit_score.ingest.jquants --probe
python -m jreit_score.ingest.jgb --list              # 取得先の一覧
python -m jreit_score.ingest.jgb --source all --inspect
python -m jreit_score.ingest.jgb --source full --out data   # all + current を結合
PYTHONPATH=. python tests/test_ingest.py
PYTHONPATH=. python tests/test_jgb.py
PYTHONPATH=. python tests/test_jquants.py
```

開発セッションから japan-reit.com / api.jquants.com / mof.go.jp に到達できない場合は、
`.github/workflows/` の inspect-sources / inspect-jgb / probe-jquants を手動実行して確認する。
取得先 URL は推測しない（`jgbcm_all.csv` を推測して 404、JAPAN-REIT.COM の DPU 履歴表は
そもそも存在せず、いずれも実行して初めて分かった）。

利用規約: JAPAN-REIT.COM は転載・複製を禁じている。個人利用のプロトタイプに限定し、
アクセス間隔を空け（2秒以上）、取得データはリポジトリにコミットしない（`data/` は .gitignore）。
公開段階では一次情報（TDnet/EDINET）に切り替える。

## 公開ダッシュボード
- `jreit_score/site.py` 財務省の10年債利回りを月末値に落として `site/data.json` を作る
- `site/index.html` それを描画する静的ページ。接続済みの系列だけを表示し、
  未接続の指標は「未接続」と明示する。合成データの数値は公開しない
- `.github/workflows/pages.yml` 生成してデプロイする。公開前に
  「jgb10 以外の数値系列が無いこと」を検証する

```
PYTHONPATH=. python -m jreit_score.site --out site   # data.json を生成（要ネットワーク）
PYTHONPATH=. python tests/test_site.py
```

## 実行
```
pip install -r requirements.txt
python run_local.py --objectives # 目的別（合成データでは全目的が使える）
python run_local.py              # 1因子 + 符号判定（合成データ）
python run_local.py --opposite   # 安定性2指標が逆向き → 2因子へ分岐
PYTHONPATH=. python tests/test_ingest.py
PYTHONPATH=. python tests/test_model.py
PYTHONPATH=. python tests/test_publish.py
```

## 公開（GitHub Pages）
- `pages.yml`（手動実行）が財務省 CSV と J-Quants の store（cache）から `site/data.json` を生成する
- 掲載するのは派生値だけ: 10 年債利回りの月末値、銘柄別スコアの z 値（掲載銘柄間で標準化）と
  五分位（5 が上位）、係数 β̂、時系列検証の IC。価格・分配金・BPS・時価総額・NAV 倍率は出さない
- 掲載するのは「推定可」かつ「時系列検証の IC の Newey–West t >= 2」の目的だけ
  （`publish.PUBLISH_MIN_T`）。推定できない目的や検証で予測力が無い目的は非掲載と表示する
- スコアは「モデル推定値」と明記し、売買推奨の表現はしない

## 実データの流れ
- `run_real.build_panel(store, jgb10, periods)` が `features.build_outcomes` と
  `jquants_panel.causes_panel` を結合する。説明変数は今のところ `nav_ratio` と `log_mcap` の 2 本
  （LTV 等は JAPAN-REIT.COM の日次蓄積待ち）
- 期ごとに `cross_sectional_standardize` を掛けてから `fit_objective_factors` /
  `rolling_validation(..., objectives=OBJECTIVES)`

## 設計上の注意
- 目的別に推定するのは、実データで 6 指標に共通因子が無かったため（目的間相関 ≤ 0.10、
  固有値 1.81 / 1.50 / 1.05）。分配金の 2 指標（dpu_stab, dpu_growth）は相関 0.06 で
  1 因子に束ねられないので、それぞれ別の目的にした
- 2 指標の因子は説明変数との共分散を通じてしか自由な負荷量が識別されない。
  説明変数が因子を説明しなければ識別不能になる（`weak`）ので、その目的は掲載しない
- 以下は 1 因子 + 符号判定（`fit_with_sign_branch`, 合成データ用）に関する注意:
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
