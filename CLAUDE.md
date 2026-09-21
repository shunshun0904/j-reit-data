# CLAUDE.md — J-REIT 統合指標 / 公開ダッシュボード

## 目的
J-REIT 58銘柄について、3つの目的（将来リターン・分配金の安定性/成長・金利上昇耐性）の
共通因子を潜在変数として推定し、財務指標から算出できる「統合スコア」を作る。
最終的に GitHub Pages で一般公開するダッシュボードにする。
**変更（2026-09-21）: 実データで 3 目的に共通する因子が確認できなかったため、統合スコアは作らず、
目的別の 3 スコアを並べる**（「決定済みの設計」の目的別 3 因子を参照）。

## 決定済みの設計
- モデル: MIMIC 型 SEM（semopy）。`quality =~ 6指標`, `quality ~ 財務指標`
  - 6指標 = ret_6m, ret_12m, dpu_stab(-CV), dpu_growth, rate_resil(-金利ベータ), dd_resil(-利上げ週DD)
  - 目的3つを各2本に分割しているのは、1因子モデルの自由度を確保して適合度検定を可能にするため
- 運用スコアは構造方程式のみ η̂ = β̂·X（将来情報を使わない）
- 検証: 時系列分割。目的変数が実現済みの期だけで学習。Spearman IC と 五分位スプレッド（Newey–West t）
- 判定ルール: 適合度が良くても負荷量の符号が割れたら 1因子で統合しない → 2因子へ
  （適合度指標だけでは符号の割れを検出できないことを合成データで確認済み）
  - 実装は `model.fit_with_sign_branch`。判定に適合度は使わない
  - 標準化負荷量 |λ|>=0.15 かつ p<0.05 の指標だけで判定する（弱い/非有意な符号反転での過剰分岐を防ぐ）
  - 2因子は負荷量の符号で指標を2群に分ける。各群2本以上必要で、1本しか無い場合は
    `two_factor_unavailable` として報告する（黙って1因子に統合しない）
  - 2因子仕様には `q1 ~~ q2` が必須。semopy は内生潜在変数間の残差共分散を自動追加しない
- 期ごとの横断面 z 化で期固定効果を除去（`model.cross_sectional_standardize`）
- データ: プロトタイプは JAPAN-REIT.COM（現在値を日次蓄積）+ J-Quants V2（価格/分配金）。
  公開段階で TDnet/EDINET の一次情報へ切替。
- 統合の重み付けは「潜在因子モデルでデータから推定」を選択済み（効用関数方式・利用者調整方式は不採用）
  → 実データで共通因子が無く成立しなかった。**統合自体をしない**（等ウェイト合成も不採用, 2026-09-21）
- **目的別 3 因子（決定 2026-09-21）**: `q_ret =~ ret_6m + ret_12m`, `q_dpu =~ dpu_stab + dpu_growth`,
  `q_rate =~ rate_resil + dd_resil`。各因子を 2 指標の MIMIC として**別々に**推定する
  （`model.fit_objective_factors`, 指標の対応は `features.OBJECTIVES`）
  - 別々に推定する理由: 因子間相関が 0.1 程度なので同時推定しても β̂ はほぼ同じで、
    識別不能な因子が他を巻き込まない。同時推定は適合度の参考にだけ使う（`fit_objective_model_joint`）
  - 2 指標の因子は説明変数との共分散を通じてしか自由な負荷量が識別されない
    （cov(y2,x)/cov(y1,x)=λ2）。因子ごとに判定し、`ok` 以外はスコアを出さない:
    `sign_split`（2 指標の向きが逆）, `weak`（|λ|<0.15 か p>=0.05 か SE が出ない = 識別不能）,
    `improper`（標準化負荷量 > 1）, `fit_failed`
  - 検証は `validation.rolling_validation(objectives=OBJECTIVES)`。因子ごとに担当指標で IC と
    五分位スプレッドを出し、期ごとの判定を `status_<因子>` に残す
  - 符号判定 + 2 因子分岐（`fit_with_sign_branch`）は合成データの検証用に残す。
    実データではこの判定は「割れていない」としか言えず、共通因子の有無は検出できない
    （`tests/test_model.py::test_one_factor_rule_does_not_catch_a_missing_common_factor`）

## 状態
- 動作確認済み: `run_local.py`（合成データで end-to-end）, `tests/test_ingest.py`（パーサのフィクスチャ検証）,
  `tests/test_model.py`（符号割れ判定と2因子分岐）。Python 3.11 / pandas 3.0 / semopy 2.3.11 で確認
- 動作確認済み: 2因子モデル分岐（`model.fit_with_sign_branch`）。`--opposite` で自動的に2因子へ落ちる
- 確認済み・要方針変更: `ingest/dpu_history.py` の取得元。Actions で 8985 を確認した結果、
  JAPAN-REIT.COM の銘柄ページは DPU を持つが「前期/当期/次期」の3期分しかない
  （table 2: cols=['Unnamed: 0','前期','当期','次期'],
    row_labels=['期首','期末','営業収益','当期利益','1口分配金']）。
  次期は予想なので実績は2期。`features.dpu_stability` は実績3期以上（既定 window=6）を
  要求するため、このページでは DPU 履歴を作れない。
  haitoukabu.com/reit/{code}.html も確認したが履歴は無い（表5件、最大でも (18,2) の
  プロフィール表。2列の表は行ラベルごとに値1つなので形状の時点で履歴を保持できない）。
  → DPU 履歴の取得元は J-Quants V2 に一本化する（`ingest/jquants.py`）
- 動作確認済み: `ingest/jgb.py`（財務省 国債金利情報）。Actions で実ファイルに対して検証した
  - 当月分 `jgbcm.csv`、過去分 `data/jgbcm_all.csv`（BASE 直下の jgbcm_all.csv は 404）
  - all は 13,290 行 / 1974-09-24〜2026-08-31、年限15本、10年の非欠損 9,929 件
  - all は当月分を含まないので `fetch_jgb10_full` で current と結合する
- 確認済み: `ingest/jquants.py`。公式クライアント jquants-api-client 2.7.0 のソースと
  実応答（discover 2回）で確定（2026-09-21）
  - ベースURL `https://api.jquants.com/v2`、認証 `x-api-key`、ページング `pagination_key`
  - `/equities/master`（4,450件。`ProdCat='013'` が 63 件で、うち名称に「インフラ」を含む
    インフラファンドが 5 件。除くと J-REIT 58 件で一致 → `select_reits`）
  - `/equities/bars/daily`（列 Date/Code/C/AdjC/Vo/MktCap[百万円]/ExRT）
  - `/fins/dividend` は**現プランで 403**。代わりに `/fins/summary` が通る:
    DocType `2Q…_REIT` / `FY…_REIT` が実績、`REITEarnForecastRevision` は予想修正。
    `CurPerEn` が期末日、`DivUnit` が1口当たり分配金の実績 → `to_dpu_from_summary`
  - 権利落ち日は取れないので総リターンの分配金計上は期末日で代用（誤差数日）
  - 決算期間の長さが銘柄で違う（大多数は6か月、8985 は12か月で年1回分配）。
    **決定（2026-09-21）: 年次決算の銘柄は母集団から除外**し、モデルは「1期あたり・6期の窓」
    のまま。判定は `to_dpu_from_summary` の期首→期末の中央値が 270 日超（`exclude_annual`）。
    census（2026-09-21）: 58 銘柄すべてで DPU 履歴あり、年次決算は `8985` の1銘柄のみ → 57 銘柄。
    期数は中央値 20（2016年8月以降）。`401A` は 2 期のみで `dpu_stability` が読み飛ばす
  - 期末が開示日より後の行（2027-01-31）が混入していたので、`period_end > DiscDate` を落とす
    （実績は期末より前に開示できない。今日の日付には依存させない）。
    再 census で確認済み: 未来期末 0 行、期末の範囲 2016-08-31〜2026-07-31
  - FY 行の `BPS`・`ShOutFY` と bars の `MktCap` で `nav_ratio` / `log_mcap` を J-Quants だけで作れる
  - 注意: `DivUnit`/`FDivUnit` は金額なので値集合をログに出さない（一度出してログを削除した）
- 注意: `Authorization: <生のキー>` は使わない。API Gateway が SigV4 として解釈し、
  ヘッダ値の SHA-256 を Base64 にしてエラーに含めて返す。候補から削除済みで、
  probe はエラーメッセージ中の長い Base64 塊を伏せ字にしてから出力する
- 動作確認済み: 公開ダッシュボードの配線（`site/index.html` + `jreit_score/site.py` +
  `.github/workflows/pages.yml`）。財務省の10年債利回りを月末値にして描画する。
  接続済みの系列だけを出し、未接続の指標は「未接続」と表示する。
  合成データの数値は公開しない
- 動作確認済み: 目的別 3 因子を実データで推定（fit-model run #3, 2026-09-21）。q_ret / q_rate は推定可、
  q_dpu は識別不能（結果はタスク 4 に記載）
- 未実装: 合併/上場廃止銘柄の復元（生存者バイアス対策）、日次スナップショット蓄積の Actions、
  スコアのダッシュボード掲載（q_ret / q_rate は検証済みで掲載可能な状態）

## 実行基盤
- 開発セッションのネットワークポリシーが japan-reit.com / api.jquants.com / mof.go.jp を
  遮断している（プロキシが CONNECT に 403）。通るのは PyPI 等と GitHub のみ。
  そのため取得は GitHub Actions 側で回す方針
- 確認済み: Actions ランナーからは mof.go.jp と japan-reit.com に到達できる（2026-09-20 実行）。
  api.jquants.com は未確認
- 取得先 URL は推測しない。`--list` のように一覧ページからリンクを列挙して確定する
  （`jgbcm_all.csv` を推測したところ 404 で、正しくは `data/jgbcm_all.csv` だった）
- ワークフローは手動実行のみ（workflow_dispatch）。スケジュール実行はしない
  - `inspect-sources.yml` JAPAN-REIT.COM の表構造確認。リポジトリが public なので
    出力は表のヘッダ名・行ラベル・行列数だけに限定する（転載・複製禁止のため）。数値セルは出さない
  - `inspect-jgb.yml` 財務省 CSV の構造確認
  - `probe-jquants.yml` J-Quants のエンドポイントと認証ヘッダの確認。
    `JQUANTS_API_KEY` / `JQUANTS_API` のどちらかが必要
  - `pages.yml` GitHub Pages へのデプロイ。財務省 CSV から `site/data.json` を生成して公開する。
    公開前に「jgb10 以外の数値系列が無いこと」を検証する
  - `fetch-jquants.yml` J-Quants の差分取得と cache 保存。出力は件数・期間・サイズのみ

## 次のタスク（優先順）
1. DPU 履歴の取得元を決め直す。JAPAN-REIT.COM の銘柄ページは3期分しか無く使えない
   （候補: J-Quants V2 の分配金、TDnet/EDINET、haitoukabu.com。いずれも未確認）
2. J-Quants の取得は確定・検証済み（58 銘柄で census 済み）。`fetch_all_dpu` と
   `fetch_prices` で全銘柄の価格・DPU を集め、`features.build_outcomes` に渡す。
   **決定（2026-09-21）: 生データは Actions cache に置く**（`ingest/jquants_store.py` +
   `.github/workflows/fetch-jquants.yml`）。価格は銘柄ごとに保存済み最終日の翌日から差分、
   DPU は毎回全件。cache のキーは run_id 込みで毎回新規保存、restore-keys で直近を復元。
   7 日未参照で消えるが全件取り直すだけ。次: 取得した store を `features.build_outcomes` に渡す
   - **現プランの価格は「今日から遡って10年」**（2026-09-21 に 400 の本文で確認:
     `covers the following dates: 2016-09-21 ~`）。窓は日々前へ動くので開始日を固定せず、
     400 の本文から開始日を読んで取り直す（`fetch_prices_clamped`）。差分取得は常に窓の内側。
     cache に残った古い日付はそのまま使える（再取得はできない）
   - 含意: rate_resilience の lookback 250 日と 12 か月の将来リターンを引くと、評価できる
     半期末は概ね 2017-12〜2025-06。rolling_validation の min_train_periods=8 だと
     out-of-sample は数期しか取れない
   - **本番取得済み（2026-09-21, fetch-jquants run #3）**: 57 銘柄、価格 132,397 行
     （2016-09-21〜2026-09-18）、DPU 1,074 行（2016-08-31〜2026-07-31）、失敗 0、所要 2分50秒、
     parquet 計 416 KB。cache キー `jquants-v1-<run_id>` に保存。次回以降は差分のみ
3. 完了（2026-09-20）。`ingest/jgb.py` で10年債利回りを取得できる（`fetch_jgb10_full`）
4. **初回実行済み（2026-09-21, fit-model run #1, 説明変数 = nav_ratio + log_mcap）**
   - 57 銘柄、2017-12〜2025-06 の 11 期、592 行（2017-06 は lookback 不足、2025-12 以降は
     12 か月先リターン未到達で 0 行）
   - 1因子の負荷量(std): ret_6m +0.73, ret_12m +0.95, dpu_stab +0.03(p=.43),
     dpu_growth +0.03(p=.55), rate_resil +0.12(p=.007), dd_resil +0.05(p=.27)
   - 構造: nav_ratio β=−0.44(p≈0), log_mcap β=+0.02(p=.70)
   - 適合度: chi2 p=0.000, CFI=0.61, TLI=0.46, RMSEA=0.16（悪い）
   - 符号判定は「1因子で統合する」（割れていない）。しかし**統合もされていない**:
     因子は将来リターン因子そのもので、分配金の安定性・成長と金利耐性はほぼ載らない。
     判定ルールが想定していなかったケース（符号は割れず、負荷量が無い）
   - rolling OOS (11 期, min_train=6): IC ret_6m +0.24 (NW-t 4.2), ret_12m +0.38 (4.7),
     dpu_stab +0.11 (1.6), dpu_growth −0.09 (−1.7), rate_resil −0.03, dd_resil −0.01,
     composite +0.21 (4.7), q5−q1 +0.34 (5.8)。予測力は nav_ratio → 将来リターンのみ
   - 未決: 「符号は割れないが統合されない」ケースの扱い（下の設計判断へ）。
     `ltv` / `noi_yield` / `unrealized_gain` は JAPAN-REIT.COM のスナップショット蓄積待ちだが、
     測定側（6 指標が共通因子を持つか）の結論は説明変数を足しても変わりにくい
   - **診断済み（2026-09-21, fit-model run #2, `run_real.indicator_structure`）**: 6 指標の Spearman 相関
     （横断面 z 化後をプール, 592 行）: ret_6m↔ret_12m 0.69、rate_resil↔dd_resil 0.60、
     dpu_stab↔dpu_growth 0.06、目的をまたぐ相関はすべて 0.10 以下。
     固有値 1.81, 1.50, 1.05, 0.93, 0.41, 0.30（1 超は 3 本）
     → **6 指標に共通因子は無い**。リターン 2 本と金利耐性 2 本はそれぞれ別の因子を成し、
       分配金の 2 本は互いにも相関しない。指標間の相関は説明変数に依存しないので、
       説明変数を足してもこの結論は変わらない
     指標 × 説明変数: nav_ratio は ret_6m −0.32 / ret_12m −0.42 / dpu_stab −0.14、他は |ρ|≤0.08。
     log_mcap は rate_resil +0.20 / dd_resil +0.15 / dpu_stab +0.18、リターンとは 0.07
     （1因子で log_mcap の β≈0 だったのは因子がリターンに寄っていたため。目的別なら効く可能性）
   - **決定（2026-09-21）: 目的別 3 因子に切替、統合しない**（実装済み。`run_real` の既定が目的別。
     `--one-factor` で参考として 1 因子 + 符号判定も出す）
   - **目的別 3 因子の初回実行（2026-09-21, fit-model run #3, causes = nav_ratio + log_mcap）**
     - q_ret（推定可）: λ ret_6m +0.72 / ret_12m +0.97、β nav_ratio −0.44 (p≈0) / log_mcap −0.02 (n.s.)、
       chi2 p=0.97, CFI 1.00
     - q_rate（推定可）: λ rate_resil +0.96 / dd_resil +0.63 (p=3e-7)、β log_mcap +0.24 (p=2e-10) /
       nav_ratio +0.01 (n.s.)、chi2 p=1.00。金利耐性は時価総額の大きさで説明される
     - q_dpu（**識別不能**）: dpu_stab λ=1.00（境界解、残差分散 0）、dpu_growth λ=+0.26 (p=0.17)。
       β nav_ratio −0.14 (p=8e-6) / log_mcap +0.08 (p=.014)。相関 0.06 の 2 指標を 1 因子に束ねられない。
       原因は説明変数ではなく指標側（安定性と成長は別のものを測っている）
     - 同時推定: DoF 15, chi2 14.5, p=0.49, CFI 1.00, RMSEA 0.00（1 因子の CFI 0.61 から改善）。
       因子間の残差相関 −0.05 / +0.09 / −0.06（説明変数で説明した後も互いに独立）
     - rolling OOS（11 期, min_train=6）:
       q_ret ok=11: IC ret_6m +0.24 (NW-t 4.2), ret_12m +0.39 (4.9), composite +0.33 (4.5), q5−q1 +1.01 (6.7)
       q_rate ok=11: IC rate_resil +0.17 (3.0), dd_resil +0.17 (8.4), composite +0.16 (4.0), q5−q1 +0.53 (6.9)
       q_dpu weak=7 / ok=4: ok の 4 期だけで IC dpu_stab +0.27, dpu_growth −0.01, q5−q1 +0.05 (0.6)。信頼できない
     - 未決: q_dpu の扱い（説明変数が揃うまで「識別不能・非掲載」のまま / 安定性と成長を別々の目的にする /
       指標定義の見直し）と、q_ret・q_rate のダッシュボード掲載
5. 配線は完了（`pages.yml`）。残りは日次で JAPAN-REIT.COM スナップショット蓄積
   （生データはコミットしない）と、スコア算出後のダッシュボード掲載

## 制約・注意
- JAPAN-REIT.COM は転載・複製禁止。個人利用に限定、アクセス間隔 2 秒以上、`data/` はコミットしない
- J-Quants データも再配布不可。公開するのはスコア等の派生値のみ
- 公開ダッシュボードは情報提供にとどめ、売買推奨の表現をしない。スコアは「モデル推定値」と明記
- Secrets: `JQUANTS_API_KEY` を GitHub Secrets に登録（V2 は API キー認証。ダッシュボード「設定 » APIキー」で取得）
- 説明変数のうち固定金利比率・稼働率は JAPAN-REIT.COM から取れないため一次情報切替まで除外（`panel.PROTO_CAUSES`）

## コードの場所
- `jreit_score/features.py` 目的変数の生成
- `jreit_score/model.py` MIMIC 推定・スコア算出・サマリ
- `jreit_score/validation.py` 時系列検証
- `jreit_score/ingest/` 取得（`japan_reit.py` ランキング, `dpu_history.py` DPU, `jgb.py` 国債利回り）
- `jreit_score/panel.py` 説明変数の整形
- `jreit_score/synthetic.py` 合成データ
- `jreit_score/run_real.py` 実データでの推定（目的別 3 因子）と診断（`indicator_structure`）
- `jreit_score/site.py` 公開ダッシュボード用 JSON の組み立て
- `site/index.html` 公開ダッシュボード（`site/data.json` は生成物なのでコミットしない）

## 実行
```
pip install -r requirements.txt
python run_local.py            # 合成データで全体を回す（1因子に収まる）
python run_local.py --opposite # 符号が割れるケース（2因子へ分岐する）
python run_local.py --objectives # 目的別3因子（合成データでは3因子とも使える）
PYTHONPATH=. python tests/test_ingest.py
PYTHONPATH=. python tests/test_model.py
PYTHONPATH=. python tests/test_jgb.py
PYTHONPATH=. python tests/test_jquants.py
PYTHONPATH=. python tests/test_jquants_store.py
PYTHONPATH=. python tests/test_site.py
PYTHONPATH=. python tests/test_jquants_panel.py
PYTHONPATH=. python tests/test_run_real.py
```

## 会話上の約束
- 比喩を使わない。結論を先に書く
- 方針の分岐がある作業は、着手前に選択肢を提示して選んでもらう
- 実在しない銘柄・サイト・数値を出さない。未検証のものは未検証と書く
