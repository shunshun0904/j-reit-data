"""J-Quants から REIT の価格と分配金を取得する.

DPU 履歴の取得元をここに一本化する。JAPAN-REIT.COM と haitoukabu.com の銘柄ページは
いずれも履歴を持たないことを確認済み（`ingest/dpu_history.py` の docstring 参照）。

未検証: API のベースURL・エンドポイント・認証ヘッダの形式は、この環境からも
JQUANTS_API_KEY 無しでも確認できていない。過去2回、URL を推測して2回とも外している
（財務省 `jgbcm_all.csv` は実際には `data/` 配下、JAPAN-REIT.COM の DPU 履歴表は存在せず）。
そのため本モジュールは「候補を決め打ちしない」構成にしてある:

  1. GitHub Secrets に `JQUANTS_API_KEY`（または `JQUANTS_API`）を登録する
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
# Secrets の登録名の揺れを吸収する（先に見つかった方を使う）
KEY_ENVS = ("JQUANTS_API_KEY", "JQUANTS_API")
TIMEOUT = 30

# 候補（未検証）。probe で 200 を返したものだけを残す
AUTH_STYLES: dict[str, callable] = {
    "none": lambda k: {},          # 認証なし。403 がルート由来か認証由来かの判別に使う
    "bearer": lambda k: {"Authorization": f"Bearer {k}"},
    "x-api-key": lambda k: {"x-api-key": k},
    "authorization-raw": lambda k: {"Authorization": k},
}
ENDPOINTS: dict[str, str] = {
    "root": "/",
    "listed_info_v2": "/v2/listed/info",
    "listed_info_v1": "/v1/listed/info",
    "daily_quotes_v1": "/v1/prices/daily_quotes",
    "dividend_v1": "/v1/fins/dividend",
}
# ベースURLの候補。JQUANTS_API_BASE で上書きできる
BASE_CANDIDATES = [
    "https://api.jquants.com",
    "https://api.jquants.com/v1",
    "https://api.jpx-jquants.com",
]


def api_key() -> tuple[str, str]:
    """(キー, 採用した環境変数名) を返す. キーそのものはログに出さないこと."""
    for name in KEY_ENVS:
        k = os.environ.get(name, "").strip()
        if k:
            return k, name
    raise RuntimeError(
        f"APIキーが未設定。{' または '.join(KEY_ENVS)} を GitHub Secrets に登録すること。"
        "（J-Quants のダッシュボード「設定 » APIキー」で取得）"
    )


@dataclass
class ProbeResult:
    endpoint: str
    auth: str
    status: int | str
    keys: list[str]
    n_records: int | None
    message: str = ""     # エラー応答の message。ルート由来か認証由来かの判別に使う
    base: str = ""

    def line(self) -> str:
        ok = "OK " if self.status == 200 else "   "
        n = "" if self.n_records is None else f" records={self.n_records}"
        msg = f' msg="{self.message}"' if self.message else ""
        return (f"  {ok}{self.endpoint:<24} auth={self.auth:<18} "
                f"status={self.status}{n} keys={self.keys}{msg}")


def probe_one(path: str, auth: str, key: str, params: dict | None = None,
              session: requests.Session | None = None, base: str | None = None) -> ProbeResult:
    """1組み合わせを試す.

    データ本体は出さないが、エラー応答の `message` は出す。
    403 がルート不在（API Gateway の "Missing Authentication Token"）なのか
    認証失敗なのかは、これを見ないと区別できない。
    """
    s = session or requests.Session()
    b = base or API_BASE
    try:
        r = s.get(b + path, headers=AUTH_STYLES[auth](key),
                  params=params or {}, timeout=TIMEOUT)
    except requests.RequestException as e:
        return ProbeResult(path, auth, type(e).__name__, [], None, base=b)
    keys: list[str] = []
    n: int | None = None
    msg = ""
    if r.headers.get("content-type", "").startswith("application/json"):
        try:
            body = r.json()
        except ValueError:
            body = None
        if isinstance(body, dict):
            keys = sorted(body)[:8]
            for k2, v in body.items():
                if isinstance(v, list) and n is None:
                    n = len(v)
            if r.status_code != 200:
                raw = body.get("message") or body.get("error") or ""
                msg = str(raw)[:160]
    elif r.status_code != 200:
        msg = r.text.strip()[:160].replace("\n", " ")
    return ProbeResult(path, auth, r.status_code, keys, n, msg, b)


def probe(session: requests.Session | None = None) -> list[ProbeResult]:
    """候補のエンドポイント × 認証ヘッダを総当たりし、どれが通るかを調べる."""
    (key, env_name), s = api_key(), session or requests.Session()
    print(f"使用する環境変数: {env_name}（値は出力しない, 長さ={len(key)}）")
    bases = ([API_BASE] if os.environ.get("JQUANTS_API_BASE") else BASE_CANDIDATES)
    out = []
    for b in bases:
        print(f"\n--- base={b} ---")
        for path in ENDPOINTS.values():
            for auth in AUTH_STYLES:
                r = probe_one(path, auth, key, session=s, base=b)
                print(r.line())
                out.append(r)
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
        results = probe()
        ok = [r for r in results if r.status == 200]
        print(f"\n200 を返した組み合わせ: {len(ok)} / {len(results)}")
        for r in ok:
            print("  " + r.line().strip())
        if not ok:
            msgs = sorted({r.message for r in results if r.message})
            print("\n観測した message:")
            for m in msgs:
                print(f"  - {m}")
            raise SystemExit(
                "通る組み合わせが無い。message が 'Missing Authentication Token' なら"
                "ルートが存在しない（パスの候補が誤り）。認証エラー文言ならキーの形式を見直す"
            )
    else:
        ap.error("--probe を指定すること（取得の実装は probe の結果を見てから固定する）")
