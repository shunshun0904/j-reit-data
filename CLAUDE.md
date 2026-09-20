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
- 期ごとの横断面 z 化で期固定効果を除去（`model.cross_sectional_standardize`）
- データ: プロトタイプは JAPAN-REIT.COM（現在値を日次蓄積）+ J-Quants V2（価格/分配金）。
  公開段階で TDnet/EDINET の一次情報へ切替。
- 統合の重み付けは「潜在因子モデルでデータから推定」を選択済み（効用関数方式・利用者調整方式は不採用）

## 状態
- 動作確認済み: `run_local.py`（合成データで end-to-end）, `tests/test_ingest.py`（パーサのフィクスチャ検証）
- 未確認: `ingest/dpu_history.py` の取得元ページの表構造。`--inspect` で確認してからパーサを固定する
- 未実装: J-Quants 取得（価格・分配金・TRI）、10年国債利回り取得（財務省CSV）、2因子モデル分岐、
  合併/上場廃止銘柄の復元（生存者バイアス対策）、GitHub Actions、ダッシュボード

## 次のタスク（優先順）
1. `python -m jreit_score.ingest.dpu_history 8985 --inspect` の結果でパーサを実構造に合わせる
2. J-Quants V2 で REIT の日次四本値と分配金を取得し `features.build_outcomes` に渡す（APIキーは `JQUANTS_API_KEY`）
3. 財務省の国債金利情報 CSV から 10年債利回りを取得
4. 実データで `fit_mimic` → 負荷量の符号と適合度を確認 → 必要なら2因子
5. Actions: 日次で JAPAN-REIT.COM スナップショット蓄積（生データはコミットしない）。派生 JSON のみ Pages へ

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
- `jreit_score/ingest/` 取得
- `jreit_score/panel.py` 説明変数の整形
- `jreit_score/synthetic.py` 合成データ

## 実行
```
pip install -r requirements.txt
python run_local.py            # 合成データで全体を回す
PYTHONPATH=. python tests/test_ingest.py
```

## 会話上の約束
- 比喩を使わない。結論を先に書く
- 方針の分岐がある作業は、着手前に選択肢を提示して選んでもらう
- 実在しない銘柄・サイト・数値を出さない。未検証のものは未検証と書く
