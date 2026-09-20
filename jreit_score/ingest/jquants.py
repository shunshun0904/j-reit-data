"""J-Quants から REIT の価格と分配金を取得する.

DPU 履歴の取得元をここに一本化する。JAPAN-REIT.COM と haitoukabu.com の銘柄ページは
いずれも履歴を持たないことを確認済み（`ingest/dpu_history.py` の docstring 参照）。

未検証: API のベースURL・エンドポイント・認証ヘッダの形式は、この環境からも
JQUANTS_API_KEY 無しでも確認できていない。過去2回、URL を推測して2回とも外している
（財務省 `jgbcm_all.csv` は実際には `data/` 配下、JAPAN-REIT.COM の DPU 履歴表は存在せず）。
そのため本モジュールは「候補を決め打ちしない」構成にしてある:

  1. GitHub Secrets に `JQUANTS_API_KEY` を登録する
  2. `probe` ワークフローを実行し、どの組み合わせが 200 を返すかをログで確認する
  3. 確認できた組み合わせだけを `ENDPOINTS` / `AUTH_STYLES` に残し、パーサを固定する

probe はステータスコードと JSON のトップレベルキー・件数だけを出力する。
J-Quants のデータは再配布不可のため、レコードそのものはログに出さない。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd
import requests

API_BASE = os.environ.get("JQUANTS_API_BASE", "https://api.jquants.com")
KEY_ENV = "JQUANTS_API_KEY"
TIMEOUT = 30

# 候補（未検証）。probe で 200 を返したものだけを残す
AUTH_STYLES: dict[str, callable] = {
    "bearer": lambda k: {"Authorization": f"Bearer {k}"},
    "x-api-key": lambda k: {"x-api-key": k},
    "authorization-raw": lambda k: {"Authorization": k},
}
ENDPOINTS: dict[str, str] = {
    "listed_info_v2": "/v2/listed/info",
    "listed_info_v1": "/v1/listed/info",
    "daily_quotes_v2": "/v2/prices/daily_quotes",
    "daily_quotes_v1": "/v1/prices/daily_quotes",
    "dividend_v2": "/v2/fins/dividend",
    "dividend_v1": "/v1/fins/dividend",
}


def api_key() -> str:
    k = os.environ.get(KEY_ENV, "").strip()
    if not k:
        raise RuntimeError(
            f"{KEY_ENV} が未設定。GitHub Secrets に登録してから実行すること。"
            "（J-Quants のダッシュボード「設定 » APIキー」で取得）"
        )
    return k


@dataclass
class ProbeResult:
    endpoint: str
    auth: str
    status: int | str
    keys: list[str]
    n_records: int | None

    def line(self) -> str:
        ok = "OK " if self.status == 200 else "   "
        n = "" if self.n_records is None else f" records={self.n_records}"
        return f"  {ok}{self.endpoint:<16} auth={self.auth:<18} status={self.status}{n} keys={self.keys}"


def probe_one(path: str, auth: str, key: str, params: dict | None = None,
              session: requests.Session | None = None) -> ProbeResult:
    """1組み合わせを試す. 応答の中身は出さず、形だけを返す."""
    s = session or requests.Session()
    try:
        r = s.get(API_BASE + path, headers=AUTH_STYLES[auth](key),
                  params=params or {}, timeout=TIMEOUT)
    except requests.RequestException as e:
        return ProbeResult(path, auth, type(e).__name__, [], None)
    keys: list[str] = []
    n: int | None = None
    if r.headers.get("content-type", "").startswith("application/json"):
        try:
            body = r.json()
        except ValueError:
            body = None
        if isinstance(body, dict):
            keys = sorted(body)[:8]
            for v in body.values():
                if isinstance(v, list):
                    n = len(v)
                    break
    return ProbeResult(path, auth, r.status_code, keys, n)


def probe(session: requests.Session | None = None) -> list[ProbeResult]:
    """候補のエンドポイント × 認証ヘッダを総当たりし、どれが通るかを調べる."""
    key, s = api_key(), session or requests.Session()
    out = []
    for name, path in ENDPOINTS.items():
        for auth in AUTH_STYLES:
            out.append(probe_one(path, auth, key, session=s))
    return out


def to_prices(records: list[dict]) -> pd.DataFrame:
    """日次四本値のレコードを features.total_return_index の入力形式に整える.

    入力キー名は J-Quants の応答に合わせて調整する必要がある（未検証）。
    出力は columns = [code, date, close, dividend]。
    """
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["code", "date", "close", "dividend"])
    ren = {"Code": "code", "Date": "date", "Close": "close",
           "AdjustmentClose": "close", "LocalCode": "code"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    missing = {"code", "date", "close"} - set(df.columns)
    if missing:
        raise KeyError(f"想定した列が無い: {sorted(missing)}。実際の列: {sorted(df.columns)}")
    out = df[["code", "date", "close"]].copy()
    out["code"] = out["code"].astype(str).str.removesuffix("0").where(
        out["code"].astype(str).str.len() == 5, out["code"].astype(str))
    out["date"] = pd.to_datetime(out["date"])
    # 取得回によって int64 / float64 が混ざらないよう float に固定する
    out["close"] = pd.to_numeric(out["close"], errors="coerce").astype("float64")
    out["dividend"] = 0.0
    return out.dropna(subset=["close"]).sort_values(["code", "date"]).reset_index(drop=True)


def to_dpu(records: list[dict]) -> pd.DataFrame:
    """分配金のレコードを features.dpu_stability の入力形式に整える.

    出力は columns = [code, period_end, dpu]。予想値の行は除く。
    """
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["code", "period_end", "dpu"])
    ren = {"Code": "code", "LocalCode": "code",
           "RecordDate": "period_end", "CurrentPeriodEndDate": "period_end",
           "DistributionAmount": "dpu", "DividendAmount": "dpu"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})
    missing = {"code", "period_end", "dpu"} - set(df.columns)
    if missing:
        raise KeyError(f"想定した列が無い: {sorted(missing)}。実際の列: {sorted(df.columns)}")
    if "ForecastResultCode" in df.columns:  # 予想/実績の区分がある場合は実績のみ
        df = df[df["ForecastResultCode"].astype(str) != "1"]
    out = df[["code", "period_end", "dpu"]].copy()
    out["code"] = out["code"].astype(str).str.removesuffix("0").where(
        out["code"].astype(str).str.len() == 5, out["code"].astype(str))
    out["period_end"] = pd.to_datetime(out["period_end"], errors="coerce")
    out["dpu"] = pd.to_numeric(out["dpu"], errors="coerce").astype("float64")
    return (out.dropna()
               .sort_values(["code", "period_end"])
               .drop_duplicates(subset=["code", "period_end"], keep="last")
               .reset_index(drop=True))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true",
                    help="候補のエンドポイントと認証ヘッダを総当たりして通るものを調べる")
    a = ap.parse_args()
    if a.probe:
        print(f"base={API_BASE}")
        results = probe()
        for r in results:
            print(r.line())
        ok = [r for r in results if r.status == 200]
        print(f"\n200 を返した組み合わせ: {len(ok)} / {len(results)}")
        if not ok:
            raise SystemExit("通る組み合わせが無い。APIキー・ベースURL・候補一覧を見直すこと")
    else:
        ap.error("--probe を指定すること（取得の実装は probe の結果を見てから固定する）")
