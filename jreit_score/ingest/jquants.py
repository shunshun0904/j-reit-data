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
discover 1〜2回目で確定（2026-09-21, 実応答）:
  - master 4,450 件。ProdCat='013' が 63 件で、8985/8951 はこれ（Mkt='0109', S33='9999'）
    → REIT は ProdCat='013'。CLAUDE.md の 58 より多いのはインフラファンド等を含むため
      と思われる（未確認。名称や別の列で絞る必要があるかは要確認）
  - bars/daily は C/AdjC/Vo が全件非null。MktCap（百万円）と ExRT も返る
  - /fins/dividend は現プランで HTTP 403（subscription）。DPU はここからは取れない
  - /fins/summary は現プランで通る。REIT の DocType は
      2QFinancialStatements_Consolidated_REIT / FYFinancialStatements_Consolidated_REIT（実績）
      REITEarnForecastRevision（予想修正）
    CurPerType は 2Q / FY。CurPerEn（当期末日）が全行にある。
    DivUnit が1口当たり分配金の実績、FDivUnit が予想（文字列。空は該当なし）
    8985 は 41 件、8951 は 27 件 → dpu_stability の窓（6期）に十分
  → DPU は /fins/summary から取る（to_dpu_from_summary）。プラン変更は不要

discover 3回目で確定（2026-09-21, 空でない件数で再集計）:
  - ProdCat='013' 63 件のうち名称に「インフラ」を含むもの 5 件 → J-REIT は 58 件
    （select_reits で除外）
  - 8985（12か月決算）の 2Q 行は Div1Q〜DivUnit が全て空で FDivUnit（予想）のみ。
    中間分配が無いので年1点が正しい。8951（6か月決算）は FY 行 20 件全てに DivUnit
    → 決算期間の長さが銘柄で違う。to_dpu_from_summary は period_start も返す
  - FY 行には BPS（1口当たり純資産）と ShOutFY（発行口数）があり、bars の MktCap と
    合わせて nav_ratio / log_mcap を J-Quants だけで作れる
  - REITEarnForecastRevision 行は Div* も FDiv* も空（別列にある可能性。使わない）

総リターンの分配金計上日: 権利落ち日は /fins/dividend でしか取れないため、
CurPerEn（期末日=基準日）で代用する。実際の権利落ち日は期末の2営業日前で、
6か月・12か月リターンに対する誤差は数日分。

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
    "dividend": "/fins/dividend",     # 2026-09-21: 現プランでは 403 (subscription)
    "summary": "/fins/summary",       # 決算短信の要約。DivFY/DivAnn 等の1口当たり分配金を含む
}
REIT_PRODCAT = "013"   # discover で 8985/8951 がこの値だった。件数 ≈58 を確認して固定する
# summary の中で非null件数を数える列（分配金・期間・開示日に関わるもの）
_SUMMARY_COUNT = re.compile(r"^(Div|FDiv|NxFDiv|.*Date|.*FY.*|CurPer|Disc|DocType|Sales|NP|EPS|BPS)$|^(Div|FDiv|NxFDiv|CurPer|CurFY|NxtFY)", re.I)
# 値の集合をログに出してよい小さな列挙列（レコードそのものは出さない）
# 値集合をログに出してよいのは区分コードだけ。DivUnit / FDivUnit は「1口当たり分配金の
# 金額」であり区分ではない（2回目の discover で値を出してしまい、ログを削除した）。
ENUM_COLS = ("ProdCat", "Mkt", "MktNm", "S17", "S33", "FRCode", "IFCode", "StatCode",
             "CommSpecCode", "IFTerm", "DocType", "CurPerType")
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
        head = f"[{self.name}] {self.n} 件"
        out = [head + (f", 列 {len(self.columns)}: {self.columns}" if self.columns else "")]
        if self.non_null:
            out.append("  空でない件数: " + ", ".join(f"{k}={v}" for k, v in self.non_null.items()))
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
    # 文字列 API では欠損が '' で来る。notna() は '' を数えてしまうので「空でない」で数える
    non_null = {c: int((df[c].astype(str).str.strip().replace({"nan": "", "None": ""}) != "").sum())
                for c in count_cols if c in df.columns}
    return Shape(name, len(df), list(df.columns), non_null, enums)


def discover(codes: tuple[str, ...] = ("8985", "8951"), session=None) -> list[Shape]:
    """3エンドポイントの応答の形を調べる. 値の集合を出すのは小さな列挙列だけ."""
    c = Client(session=session)
    out: list[Shape] = []

    # (a) master 全体: ProdCat / Mkt の値集合と件数、ProdCat ごとの銘柄数
    try:
        m = c.get_all(ENDPOINTS["master"])
        sh = shape_of("master(全体)", m)
        df = pd.DataFrame.from_records(m)
        if "ProdCat" in df.columns:
            vc = df["ProdCat"].astype(str).value_counts().sort_index()
            sh.non_null = {f"ProdCat={k}": int(v) for k, v in vc.items()}
            # 013 の内訳: 名称に含まれる語で数える（名称そのものは出さない）
            r = df[df["ProdCat"].astype(str) == REIT_PRODCAT]
            nm = r["CoName"].astype(str) if "CoName" in r.columns else pd.Series([], dtype=str)
            sh.non_null.update({
                "013_名称に「投資法人」": int(nm.str.contains("投資法人").sum()),
                "013_名称に「インフラ」": int(nm.str.contains("インフラ").sum()),
                "013_Mkt=0109": int((r["Mkt"].astype(str) == "0109").sum()) if "Mkt" in r else 0,
            })
        out.append(sh)
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
        # (f) 財務サマリ: 1口当たり分配金が埋まるか、何期分あるか（金額は出さない）
        try:
            f = c.get_all(ENDPOINTS["summary"], {"code": code})
            cols = tuple(k for k in (f[0].keys() if f else ()) if _SUMMARY_COUNT.match(k))
            out.append(shape_of(f"summary(code={code})", f, count_cols=cols))
            # (g') DocType×CurPerType ごとの分配金列の「空でない」件数。
            #      2Q 行で DivUnit が空なら Div2Q に中間分配があるか、をここで確かめる
            fdf = pd.DataFrame.from_records(f)
            if {"DocType", "CurPerType"} <= set(fdf.columns):
                for (dt, pt), g in fdf.groupby(["DocType", "CurPerType"]):
                    gs = shape_of(f"summary(code={code}) {dt}/{pt}", g.to_dict("records"),
                                  enum_cols=(), count_cols=("Div1Q", "Div2Q", "Div3Q", "DivFY",
                                                            "DivAnn", "DivUnit", "FDivUnit"))
                    gs.columns = []   # 列一覧は上で出しているので省く
                    out.append(gs)
            # (g) 整形後: 実績期の件数と期間だけ
            d = to_dpu_from_summary(f)
            sh = Shape(f"dpu_from_summary(code={code})", len(d))
            if len(d):
                sh.non_null = {"first_period_end": str(d["period_end"].min().date()),
                               "last_period_end": str(d["period_end"].max().date()),
                               "n_positive": int((d["dpu"] > 0).sum())}
            out.append(sh)
        except Exception as e:
            out.append(Shape(f"summary(code={code})", 0, error=redact(str(e))[:200]))
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


# /fins/summary の DocType のうち実績（決算短信本体）
SUMMARY_ACTUAL_DOCTYPES = ("2QFinancialStatements_Consolidated_REIT",
                           "FYFinancialStatements_Consolidated_REIT")


def to_dpu_from_summary(records: list[dict],
                        doctypes: tuple[str, ...] = SUMMARY_ACTUAL_DOCTYPES) -> pd.DataFrame:
    """財務サマリ (/fins/summary) から [code, period_end, dpu, ex_date] を作る.

    - DocType が実績の決算短信の行だけを使う（REITEarnForecastRevision は除く）
    - period_end = CurPerEn（当期末日）
    - dpu = DivUnit（1口当たり分配金の実績。文字列で空は該当なし）
    - ex_date は権利落ち日が取れないので period_end で代用（docstring 参照）
    同じ期の訂正開示が複数あれば DiscDate の新しい方を採る。
    """
    df = pd.DataFrame.from_records(records)
    if df.empty:
        return pd.DataFrame(columns=["code", "period_start", "period_end", "dpu", "ex_date"])
    need = {"Code", "CurPerEn", "DivUnit", "DocType"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"想定した列が無い: {sorted(missing)}。実際の列: {sorted(df.columns)}")
    df = df[df["DocType"].astype(str).isin(doctypes)].copy()
    if "DiscDate" in df.columns:
        df = df.sort_values("DiscDate")
    out = pd.DataFrame({
        "code": _code4(df["Code"]),
        "period_end": pd.to_datetime(df["CurPerEn"], errors="coerce"),
        # 決算期間の長さは銘柄で違う（大多数は6か月、8985 のように12か月もある）。
        # 下流で成長率を年率化できるよう期首も持たせる
        "period_start": pd.to_datetime(df["CurPerSt"], errors="coerce") if "CurPerSt" in df.columns
                        else pd.NaT,
        "dpu": pd.to_numeric(df["DivUnit"].astype(str).str.replace(",", ""), errors="coerce")
                 .astype("float64"),
    })
    out["ex_date"] = out["period_end"]
    return (out.dropna(subset=["period_end", "dpu"])
               .drop_duplicates(subset=["code", "period_end"], keep="last")
               .sort_values(["code", "period_end"]).reset_index(drop=True)
               [["code", "period_start", "period_end", "dpu", "ex_date"]])


INFRA_NAME_KEY = "インフラ"   # ProdCat='013' にはインフラファンド (5件) も含まれる


def select_reits(master: pd.DataFrame, exclude_infra: bool = True) -> pd.DataFrame:
    """master から J-REIT を選ぶ. ProdCat='013' から名称に「インフラ」を含むものを除く.

    2026-09-21 の実応答: ProdCat='013' は 63 件、うち「インフラ」を含む名称が 5 件。
    63 - 5 = 58 で CLAUDE.md の J-REIT 58 銘柄と一致する。
    """
    if master.empty or "ProdCat" not in master.columns:
        return pd.DataFrame(columns=["code", "code5", "name"])
    r = master[master["ProdCat"].astype(str) == REIT_PRODCAT]
    name = r["CoName"].astype(str) if "CoName" in r.columns else pd.Series("", index=r.index)
    if exclude_infra:
        r, name = r[~name.str.contains(INFRA_NAME_KEY)], name[~name.str.contains(INFRA_NAME_KEY)]
    return (pd.DataFrame({"code": _code4(r["Code"]), "code5": r["Code"].astype(str), "name": name})
              .drop_duplicates("code").sort_values("code").reset_index(drop=True))


# 決算期間がこの日数を超える銘柄は年次決算（年1回分配）とみなして母集団から除く。
# 大多数の J-REIT は6か月（≈182日）、8985 のような12か月決算（≈365日）は年1点しか
# DPU が無く、「1期あたり」の成長率と6期の窓が半期銘柄と揃わないため（設計判断 2026-09-21）
ANNUAL_SPAN_DAYS = 270


def period_span_days(dpu: pd.DataFrame) -> pd.Series:
    """銘柄ごとの決算期間の中央値（日）. period_start が無ければ連続する期末の差で代用."""
    if dpu.empty:
        return pd.Series(dtype="float64", name="span_days")
    d = dpu.sort_values(["code", "period_end"]).copy()
    span = (d["period_end"] - d["period_start"]).dt.days if "period_start" in d.columns else None
    fallback = d.groupby("code")["period_end"].diff().dt.days
    d["span"] = span.where(span.notna(), fallback) if span is not None else fallback
    return d.groupby("code")["span"].median().rename("span_days")


def exclude_annual(dpu: pd.DataFrame, max_days: int = ANNUAL_SPAN_DAYS) -> tuple[pd.DataFrame, list[str]]:
    """年次決算の銘柄を除いた DPU と、除いた銘柄コードを返す.

    期間が判定できない銘柄（1期しか無い等）は除かない。
    """
    span = period_span_days(dpu)
    annual = sorted(span[span > max_days].index.astype(str))
    return dpu[~dpu["code"].astype(str).isin(annual)].reset_index(drop=True), annual


def reit_universe(client: Client, exclude_infra: bool = True) -> pd.DataFrame:
    """上場 J-REIT の一覧. 列 code(4桁), code5, name."""
    return select_reits(pd.DataFrame.from_records(client.get_all(ENDPOINTS["master"])), exclude_infra)


def fetch_prices(client: Client, code: str, from_yyyymmdd: str, to_yyyymmdd: str) -> pd.DataFrame:
    """1銘柄の日次四本値を features 形式で返す."""
    return to_prices(client.get_all(ENDPOINTS["bars_daily"],
                                    {"code": code, "from": from_yyyymmdd, "to": to_yyyymmdd}))


def fetch_dpu(client: Client, code: str) -> pd.DataFrame:
    """1銘柄の DPU 履歴を /fins/summary から features 形式で返す."""
    return to_dpu_from_summary(client.get_all(ENDPOINTS["summary"], {"code": code}))


def census(client: Client, sleep: float = SLEEP) -> dict:
    """全 J-REIT のサマリを引き、決算期間の分布と年次決算で除外される銘柄を数える.

    出すのは件数・期間・銘柄コードだけ。分配金額は出さない。
    """
    uni = reit_universe(client)
    frames, failed = [], []
    for code in uni["code"]:
        try:
            frames.append(fetch_dpu(client, code))
        except Exception as e:
            failed.append((code, redact(str(e))[:80]))
        time.sleep(sleep)
    dpu = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    span = period_span_days(dpu)
    kept, annual = exclude_annual(dpu)
    per_code = dpu.groupby("code").size() if len(dpu) else pd.Series(dtype=int)
    return {
        "universe": int(len(uni)),
        "with_dpu": int(dpu["code"].nunique()) if len(dpu) else 0,
        "failed": failed,
        "span_days_distribution": span.round().value_counts().sort_index().to_dict(),
        "annual_excluded": annual,
        "kept": int(kept["code"].nunique()) if len(kept) else 0,
        "periods_per_code_min_median_max": (int(per_code.min()), float(per_code.median()), int(per_code.max()))
                                            if len(per_code) else None,
        "kept_codes_with_lt6_periods": sorted(per_code[per_code < 6].index.astype(str)) if len(per_code) else [],
        "period_end_range": (str(dpu["period_end"].min().date()), str(dpu["period_end"].max().date()))
                            if len(dpu) else None,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--discover", action="store_true",
                    help="3エンドポイントの応答の形（列・件数・列挙値）を調べる。レコードは出さない")
    ap.add_argument("--census", action="store_true",
                    help="全 J-REIT の決算期間の分布と、年次決算で除外される銘柄を数える")
    ap.add_argument("--codes", nargs="*", default=["8985", "8951"])
    a = ap.parse_args()
    if a.census:
        _, env = api_key()
        print(f"base={API_BASE}  key={env}（値は出さない）")
        for k, v in census(Client()).items():
            print(f"  {k}: {v}")
        raise SystemExit(0)
    if a.discover:
        _, env = api_key()
        print(f"base={API_BASE}  key={env}（値は出さない）")
        for sh in discover(tuple(a.codes)):
            print("\n".join(sh.lines()))
        raise SystemExit(0)
    ap.error("--discover を指定すること")
