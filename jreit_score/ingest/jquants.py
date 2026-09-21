"""J-Quants V2 から REIT の価格と分配金を取得する.

DPU 履歴の取得元をここに一本化する。JAPAN-REIT.COM と haitoukabu.com の銘柄ページは
いずれも履歴を持たないことを確認済み（`ingest/dpu_history.py` の docstring 参照）。

確定済み（公式クライアント jquants-api-client 2.7.0 のソースから, 2026-09-21）:
  ベースURL  : https://api.jquants.com/v2
  認証       : ヘッダ `x-api-key: <APIキー>`
  ページング : 応答の `pagination_key` をクエリ `pagination_key` に戻す。データ配列は `data`
  パス       : /equities/master  (上場銘柄一覧)  列 Date, Code, CoName, ..., Mkt, MktNm, ProdCat
               /equities/bars/daily (日次四本値)  列 Date, Code, C, AdjC, Vo, AdjFactor, ...
               /fins/dividend     (配当・分配金)  列 Code, DivRate, DistAmt, RecDate, ExDate,
                                                  FRCode, IFCode, PayDate, PubDate, ...
  銘柄コード : 5桁 (例 89850)。4桁指定も可とクライアントの docstring にある

未確定（仕様ページが 403 のため実応答で確認する。`--discover`）:
  - ProdCat のどの値が REIT か
  - FRCode のどの値が実績か（予想を除くため）
  - REIT の分配金は DivRate と DistAmt のどちらに入るか

過去の教訓: URL を推測して2回外している（財務省は data/ 配下、JAPAN-REIT.COM の
DPU 履歴表は不在）。J-Quants も `/v2/listed/info` と推測して外した（正しくは
/v2/equities/master）。推測せず、公式クライアントのソースと実応答で確定する。

discover はレコードを出さない。列名・列ごとの非null件数・小さな列挙の値集合・件数だけ。
J-Quants のデータは再配布不可。APIキーも出さない。
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field

import pandas as pd
import requests

API_BASE = os.environ.get("JQUANTS_API_BASE", "https://api.jquants.com/v2")
KEY_ENVS = ("JQUANTS_API_KEY", "JQUANTS_API")   # 登録名の揺れを吸収（先に見つかった方）
TIMEOUT = 30
SLEEP = 0.5                                       # ページ間・銘柄間の間隔（秒）

ENDPOINTS = {
    "master": "/equities/master",
    "bars_daily": "/equities/bars/daily",
    "dividend": "/fins/dividend",
}
# 値の集合をログに出してよい小さな列挙列（レコードそのものは出さない）
ENUM_COLS = ("ProdCat", "Mkt", "MktNm", "S17", "S33", "FRCode", "IFCode", "StatCode",
             "CommSpecCode", "IFTerm")
_B64 = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")


def redact(text: str) -> str:
    """長い Base64 らしき塊を伏せる（キーのハッシュがログに残らないように）."""
    return _B64.sub("<redacted>", text)


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


class Client:
    """最小のクライアント. 公式クライアントと同じ認証・ページング方式."""

    def __init__(self, key: str | None = None, session: requests.Session | None = None,
                 base: str = API_BASE):
        self.key = key or api_key()[0]
        self.s = session or requests.Session()
        self.base = base.rstrip("/")

    def get_all(self, path: str, params: dict | None = None,
                data_key: str = "data", sleep: float = SLEEP) -> list[dict]:
        """pagination_key を辿って全件返す."""
        url, q, out = self.base + path, dict(params or {}), []
        while True:
            r = self.s.get(url, params=q, headers={"x-api-key": self.key}, timeout=TIMEOUT)
            if r.status_code != 200:
                msg = ""
                try:
                    msg = redact(str(r.json().get("message", "")))[:200]
                except ValueError:
                    msg = redact(r.text[:200])
                raise RuntimeError(f"{path} -> HTTP {r.status_code}: {msg}")
            payload = r.json()
            batch = payload.get(data_key, [])
            if isinstance(batch, list):
                out.extend(batch)
            pk = payload.get("pagination_key")
            if not pk:
                return out
            q["pagination_key"] = pk
            time.sleep(sleep)


# ---------------------------------------------------------------------------
# discover: 実応答の「形」だけを見る
# ---------------------------------------------------------------------------

@dataclass
class Shape:
    name: str
    n: int
    columns: list[str] = field(default_factory=list)
    non_null: dict[str, int] = field(default_factory=dict)
    enums: dict[str, list[str]] = field(default_factory=dict)
    error: str = ""

    def lines(self) -> list[str]:
        if self.error:
            return [f"[{self.name}] ERROR {self.error}"]
        out = [f"[{self.name}] {self.n} 件, 列 {len(self.columns)}: {self.columns}"]
        if self.non_null:
            out.append("  非null件数: " + ", ".join(f"{k}={v}" for k, v in self.non_null.items()))
        for k, v in self.enums.items():
            out.append(f"  {k} の値集合 ({len(v)} 種): {v[:30]}")
        return out


def shape_of(name: str, records: list[dict], enum_cols=ENUM_COLS,
             count_cols: tuple[str, ...] = ()) -> Shape:
    if not records:
        return Shape(name, 0)
    df = pd.DataFrame.from_records(records)
    enums = {c: sorted(map(str, df[c].dropna().unique()))
             for c in enum_cols if c in df.columns}
    non_null = {c: int(df[c].notna().sum()) for c in count_cols if c in df.columns}
    return Shape(name, len(df), list(df.columns), non_null, enums)


def discover(codes: tuple[str, ...] = ("8985", "8951"), session=None) -> list[Shape]:
    """3エンドポイントの応答の形を調べる. 値の集合を出すのは小さな列挙列だけ."""
    c = Client(session=session)
    out: list[Shape] = []

    # (a) master 全体: ProdCat / Mkt の値集合と件数
    try:
        m = c.get_all(ENDPOINTS["master"])
        out.append(shape_of("master(全体)", m))
        df = pd.DataFrame.from_records(m)
        for code in codes:
            hit = df[df["Code"].astype(str).str.startswith(code)] if "Code" in df else df.iloc[0:0]
            out.append(shape_of(f"master(code={code})", hit.to_dict("records")))
    except Exception as e:
        out.append(Shape("master", 0, error=redact(str(e))[:200]))
    time.sleep(SLEEP)

    for code in codes:
        # (e) 4桁コードで通るか、(d) 四本値の列
        try:
            b = c.get_all(ENDPOINTS["bars_daily"], {"code": code, "from": "20240101", "to": "20241231"})
            out.append(shape_of(f"bars_daily(code={code}, 2024)", b, count_cols=("C", "AdjC", "Vo")))
        except Exception as e:
            out.append(Shape(f"bars_daily(code={code})", 0, error=redact(str(e))[:200]))
        time.sleep(SLEEP)
        # (b)(c) 配当: FRCode/IFCode の値集合と DivRate/DistAmt の非null件数
        try:
            d = c.get_all(ENDPOINTS["dividend"], {"code": code})
            out.append(shape_of(f"dividend(code={code})", d,
                                count_cols=("DivRate", "DistAmt", "RecDate", "ExDate", "PayDate")))
        except Exception as e:
            out.append(Shape(f"dividend(code={code})", 0, error=redact(str(e))[:200]))
        time.sleep(SLEEP)
    return out


# ---------------------------------------------------------------------------
# 整形: features 側の入力形式へ
# ---------------------------------------------------------------------------

def _code4(s: pd.Series) -> pd.Series:
    """5桁コード (89850) を4桁 (8985) にする. 4桁はそのまま."""
    s = s.astype(str).str.strip()
    return s.where(s.str.len() != 5, s.str[:4])


def to_prices(records: list[dict], price_col: str = "AdjC") -> pd.DataFrame:
    """日次四本値を [code, date, close, dividend] に整える（features.total_return_index の入力）.

    V2 の列名は略記: Date, Code, C (終値), AdjC (分割調整済み終値)。
    分配金は features 側で ExDate に計上するので、ここでは 0 で初期化する。
    """
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return pd.DataFrame(columns=["code", "date", "close", "dividend"])
    need = {"Code", "Date", price_col}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"想定した列が無い: {sorted(missing)}。実際の列: {sorted(df.columns)}")
    out = pd.DataFrame({
        "code": _code4(df["Code"]),
        "date": pd.to_datetime(df["Date"], errors="coerce"),
        "close": pd.to_numeric(df[price_col], errors="coerce").astype("float64"),
    })
    out["dividend"] = 0.0
    return (out.dropna(subset=["date", "close"])
               .sort_values(["code", "date"]).reset_index(drop=True))


def to_dpu(records: list[dict], amount_col: str = "DivRate",
           actual_codes: tuple[str, ...] | None = None) -> pd.DataFrame:
    """配当・分配金を [code, period_end, dpu, ex_date] に整える（features.dpu_stability の入力）.

    period_end は RecDate（基準日）。ex_date は総リターン計算で分配金を計上する日。
    actual_codes を渡すと FRCode がその集合に含まれる行だけ残す（予想を除く）。
    FRCode の実績値は --discover で確認してから固定する。
    """
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return pd.DataFrame(columns=["code", "period_end", "dpu", "ex_date"])
    need = {"Code", "RecDate", amount_col}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"想定した列が無い: {sorted(missing)}。実際の列: {sorted(df.columns)}")
    if actual_codes is not None and "FRCode" in df.columns:
        df = df[df["FRCode"].astype(str).isin(actual_codes)]
    out = pd.DataFrame({
        "code": _code4(df["Code"]),
        "period_end": pd.to_datetime(df["RecDate"], errors="coerce"),
        "dpu": pd.to_numeric(df[amount_col], errors="coerce").astype("float64"),
        "ex_date": pd.to_datetime(df["ExDate"], errors="coerce") if "ExDate" in df.columns
                   else pd.NaT,
    })
    return (out.dropna(subset=["period_end", "dpu"])
               .sort_values(["code", "period_end"])
               .drop_duplicates(subset=["code", "period_end"], keep="last")
               .reset_index(drop=True))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--discover", action="store_true",
                    help="3エンドポイントの応答の形（列・件数・列挙値）を調べる。レコードは出さない")
    ap.add_argument("--codes", nargs="*", default=["8985", "8951"])
    a = ap.parse_args()
    if a.discover:
        _, env = api_key()
        print(f"base={API_BASE}  key={env}（値は出さない）")
        for sh in discover(tuple(a.codes)):
            print("\n".join(sh.lines()))
        raise SystemExit(0)
    ap.error("--discover を指定すること")
