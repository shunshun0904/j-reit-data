# CLAUDE.md — J-REIT 統合指標 / 公開ダッシュボード

## 目的
J-REIT 58銘柄について、3つの目的（将来リターン・分配金の安定性/成長・金利上昇耐性）の
共通因子を潜在変数として推定し、財務指標から算出できる「統合スコア」を作る。
最終的に GitHub Pages で一般公開するダッシュボードにする。

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
  - `/equities/master`（4,450件。REIT は `ProdCat='013'` で 63 件。CLAUDE.md の 58 より
    多い理由は未確認。インフラファンド等を含む可能性）
  - `/equities/bars/daily`（列 Date/Code/C/AdjC/Vo/MktCap[百万円]/ExRT）
  - `/fins/dividend` は**現プランで 403**。代わりに `/fins/summary` が通る:
    DocType `2Q…_REIT` / `FY…_REIT` が実績、`REITEarnForecastRevision` は予想修正。
    `CurPerEn` が期末日、`DivUnit` が1口当たり分配金の実績 → `to_dpu_from_summary`
  - 権利落ち日は取れないので総リターンの分配金計上は期末日で代用（誤差数日）
  - 注意: `DivUnit`/`FDivUnit` は金額なので値集合をログに出さない（一度出してログを削除した）
- 注意: `Authorization: <生のキー>` は使わない。API Gateway が SigV4 として解釈し、
  ヘッダ値の SHA-256 を Base64 にしてエラーに含めて返す。候補から削除済みで、
  probe はエラーメッセージ中の長い Base64 塊を伏せ字にしてから出力する
- 動作確認済み: 公開ダッシュボードの配線（`site/index.html` + `jreit_score/site.py` +
  `.github/workflows/pages.yml`）。財務省の10年債利回りを月末値にして描画する。
  接続済みの系列だけを出し、未接続の指標は「未接続」と表示する。
  合成データの数値は公開しない
- 未実装: 合併/上場廃止銘柄の復元（生存者バイアス対策）、日次スナップショット蓄積の Actions、
  スコアのダッシュボード掲載（データ接続後）

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

## 次のタスク（優先順）
1. DPU 履歴の取得元を決め直す。JAPAN-REIT.COM の銘柄ページは3期分しか無く使えない
   （候補: J-Quants V2 の分配金、TDnet/EDINET、haitoukabu.com。いずれも未確認）
2. J-Quants の取得は確定済み。`reit_universe` → `fetch_prices` / `fetch_dpu` で
   全 REIT の価格・DPU を集めて `features.build_outcomes` に渡す（次は取得の一括実行と保存）
3. 完了（2026-09-20）。`ingest/jgb.py` で10年債利回りを取得できる（`fetch_jgb10_full`）
4. 実データで `fit_with_sign_branch` → 判定結果（1因子/2因子）と適合度を確認
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
- `jreit_score/site.py` 公開ダッシュボード用 JSON の組み立て
- `site/index.html` 公開ダッシュボード（`site/data.json` は生成物なのでコミットしない）

## 実行
```
pip install -r requirements.txt
python run_local.py            # 合成データで全体を回す（1因子に収まる）
python run_local.py --opposite # 符号が割れるケース（2因子へ分岐する）
PYTHONPATH=. python tests/test_ingest.py
PYTHONPATH=. python tests/test_model.py
PYTHONPATH=. python tests/test_jgb.py
PYTHONPATH=. python tests/test_jquants.py
PYTHONPATH=. python tests/test_site.py
```

## 会話上の約束
- 比喩を使わない。結論を先に書く
- 方針の分岐がある作業は、着手前に選択肢を提示して選んでもらう
- 実在しない銘柄・サイト・数値を出さない。未検証のものは未検証と書く
