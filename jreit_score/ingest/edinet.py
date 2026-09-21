"""EDINET API v2 の probe: 投資法人の有価証券報告書を探し、物件表の構造を確認する（タスク 6 段階 1）.

前提: GitHub Secrets の `EDINET_API_KEY`（EDINET でアカウント登録して発行する無料の API キー）。
API の仕様は金融庁「EDINET API 仕様書 (Version 2)」に従う想定だが、このセッションからは仕様書を
開けていないため、応答のキー名は決め打ちにせず `--inspect` で実物を表示して確定する。

- 書類一覧: GET {API}/documents.json?date=YYYY-MM-DD&type=2&Subscription-Key=KEY
  日付ごとにしか取れないので、期間をスキャンして銘柄（secCode = 4桁コード + "0"）で突合する
- 書類取得: GET {API}/documents/{docID}?type=1&Subscription-Key=KEY → zip（XBRL と HTML）
- 物件表: 有報（第7号様式）の「不動産等の概要」等は HTML 表。所在地を含む表を探して構造を出す

ログに出すのは件数・キー名・表の見出しと行数、物件名の例（公開開示の事実）だけ。キーは出さない。
"""
from __future__ import annotations

import argparse
import io
import os
import re
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

API = os.environ.get("EDINET_API_BASE", "https://api.edinet-fsa.go.jp/api/v2")
KEY_ENVS = ("EDINET_API_KEY",)
DOC_TYPE_ANNUAL = "120"          # 有価証券報告書（訂正は 130）
UA = "jreit-score-prototype/0.1 (personal research)"
LOCATION_KEYS = ("所在地", "所在")
NAME_KEYS = ("物件名", "物件の名称", "不動産等の名称", "名称")


def api_key() -> str:
    for k in KEY_ENVS:
        v = os.environ.get(k)
        if v:
            return v
    raise SystemExit(f"API キーが無い（環境変数 {' / '.join(KEY_ENVS)}）")


def redact(text: str, key: str) -> str:
    return text.replace(key, "***") if key else text


class Client:
    def __init__(self, key: str, sleep: float = 0.3):
        self.key, self.sleep = key, sleep
        self.s = requests.Session()
        self.s.headers["User-Agent"] = UA

    def _get(self, path: str, **params):
        params["Subscription-Key"] = self.key
        r = self.s.get(f"{API}{path}", params=params, timeout=60)
        time.sleep(self.sleep)
        if r.status_code != 200:
            raise RuntimeError(f"{path} -> HTTP {r.status_code}: {redact(r.text[:200], self.key)}")
        return r

    def documents(self, date: str) -> tuple[dict, list[dict]]:
        """書類一覧（type=2 はメタデータ + 一覧）. 返り値 (metadata, results).

        想定と違う形の応答（`results` が無い）は、先頭のキーと本文の抜粋（キーは伏せ字）を付けて失敗にする。
        """
        r = self._get("/documents.json", date=date, type=2)
        try:
            j = r.json()
        except ValueError:
            raise RuntimeError(f"documents.json: JSON でない応答 ({r.headers.get('content-type')}): "
                               f"{redact(r.text[:300], self.key)!r}")
        if not isinstance(j, dict) or "results" not in j:
            top = sorted(j.keys()) if isinstance(j, dict) else type(j).__name__
            raise RuntimeError(f"documents.json: 想定外の形。top-level={top}; "
                               f"本文抜粋={redact(r.text[:300], self.key)!r}")
        return j.get("metadata", {}) or {}, j.get("results", []) or []

    def download(self, doc_id: str, kind: int = 1) -> bytes:
        return self._get(f"/documents/{doc_id}", type=kind).content


def sec_code(code4: str) -> str:
    """J-Quants の 4 桁コード → EDINET の secCode（5 桁。末尾 0）."""
    return f"{code4}0"


def scan(client: Client, start: str, end: str, log_every: int = 30) -> pd.DataFrame:
    """期間内の書類一覧を日付ごとに集める."""
    frames, n_err = [], 0
    days = pd.date_range(start, end, freq="D")
    for i, d in enumerate(days):
        try:
            _, res = client.documents(d.strftime("%Y-%m-%d"))
        except Exception as e:  # 1 日分の失敗で止めない
            n_err += 1
            print(f"  [warn] {d.date()}: {type(e).__name__}: {str(e)[:120]}")
            continue
        if res:
            df = pd.DataFrame(res)
            df["_date"] = d.strftime("%Y-%m-%d")
            frames.append(df)
        if (i + 1) % log_every == 0:
            print(f"  ... {i + 1}/{len(days)} 日 ({sum(len(f) for f in frames)} 件)")
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    print(f"スキャン {start}〜{end}: {len(days)} 日, 書類 {len(out)} 件, 失敗 {n_err} 日")
    return out


def annual_reports(docs: pd.DataFrame, codes: list[str]) -> pd.DataFrame:
    """銘柄ごとの最新の有価証券報告書（docTypeCode == 120）."""
    if docs.empty or "secCode" not in docs.columns:
        return pd.DataFrame()
    want = {sec_code(c): c for c in codes}
    d = docs[docs["secCode"].astype(str).isin(want)].copy()
    if "docTypeCode" in d.columns:
        d = d[d["docTypeCode"].astype(str) == DOC_TYPE_ANNUAL]
    if "withdrawalStatus" in d.columns:
        d = d[d["withdrawalStatus"].astype(str) == "0"]
    d["code"] = d["secCode"].astype(str).map(want)
    sort_col = "submitDateTime" if "submitDateTime" in d.columns else "_date"
    d = d.sort_values(sort_col).groupby("code", as_index=False).tail(1)
    keep = [c for c in ["code", "secCode", "docID", "filerName", "docTypeCode", "ordinanceCode", "formCode",
                        "periodStart", "periodEnd", "submitDateTime", "xbrlFlag", "csvFlag"] if c in d.columns]
    return d[keep].sort_values("code").reset_index(drop=True)


def _cell(v) -> str:
    return re.sub(r"\s+", " ", str(v)).strip()


def find_location_tables(html: str) -> list[dict]:
    """所在地を含む表を探し、構造だけ返す（見出し・行数・物件名の例）."""
    try:
        tables = pd.read_html(io.StringIO(html), flavor="lxml")
    except ValueError:  # 表が無い
        return []
    out = []
    for i, t in enumerate(tables):
        if t.shape[0] < 2 or t.shape[1] < 2:
            continue
        head = [_cell(c) for c in (t.columns if not isinstance(t.columns, pd.RangeIndex) else t.iloc[0].tolist())]
        top = " ".join(head + [_cell(x) for x in t.iloc[:3].to_numpy().ravel()])
        if not any(k in top for k in LOCATION_KEYS):
            continue
        name_col = next((j for j, h in enumerate(head) if any(k in h for k in NAME_KEYS)), 0)
        samples = [_cell(x)[:30] for x in t.iloc[1:4, name_col].tolist()]
        out.append({"table": i, "shape": t.shape, "head": [h[:16] for h in head[:10]], "samples": samples})
    return out


def inspect_zip(blob: bytes, max_files: int = 12) -> None:
    z = zipfile.ZipFile(io.BytesIO(blob))
    names = z.namelist()
    print(f"zip: {len(names)} ファイル, {len(blob) / 1e6:.1f} MB")
    for n in names[:max_files]:
        print(f"  {n} ({z.getinfo(n).file_size} B)")
    if len(names) > max_files:
        print(f"  ... +{len(names) - max_files}")
    htmls = [n for n in names if n.lower().endswith((".htm", ".html")) and "PublicDoc" in n]
    print(f"PublicDoc の HTML: {len(htmls)} 件")
    hits = 0
    for n in htmls:
        html = z.read(n).decode("utf-8", errors="replace")
        found = find_location_tables(html)
        if found:
            print(f"  {n}: 所在地を含む表 {len(found)} 件")
            for f in found[:6]:
                print(f"    table {f['table']} shape={f['shape']} head={f['head']} 例={f['samples']}")
            hits += len(found)
    print(f"所在地を含む表: 合計 {hits} 件")


def load_codes(universe: str | None, codes: list[str]) -> list[str]:
    if universe and Path(universe).exists():
        u = pd.read_parquet(universe)
        return sorted(u["code"].astype(str).tolist())
    return codes


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=220, help="書類一覧を遡る日数")
    ap.add_argument("--end", default=None, help="スキャン終了日（既定: 今日）")
    ap.add_argument("--codes", nargs="*", default=["8972"])
    ap.add_argument("--universe", default="data/jquants/universe.parquet", help="J-Quants の universe（あれば全銘柄）")
    ap.add_argument("--inspect", default="8972", help="有報 zip を取って物件表の構造を出す銘柄（空なら取らない）")
    ap.add_argument("--sleep", type=float, default=0.3)
    a = ap.parse_args()

    key = api_key()
    client = Client(key, sleep=a.sleep)
    end = pd.Timestamp(a.end) if a.end else pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=a.days)

    # 直近の平日 1 日分を先に取って応答の形（キー名）を確認する。形が違えばここで止める
    probe_day = end - pd.offsets.BDay(1)
    meta, res = client.documents(probe_day.strftime("%Y-%m-%d"))
    print(f"probe {probe_day.date()}: metadata keys:", sorted(meta.keys()))
    print("  metadata:", {k: (str(v)[:60]) for k, v in meta.items() if k != "parameter"})
    print("  results:", len(res), "件; keys:", sorted(res[0].keys()) if res else "（当日は書類なし）")

    codes = load_codes(a.universe, a.codes)
    print(f"対象銘柄 {len(codes)} 件")
    docs = scan(client, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    if not docs.empty and "docTypeCode" in docs.columns:
        print("docTypeCode の内訳（上位）:", docs["docTypeCode"].astype(str).value_counts().head(8).to_dict())
    ar = annual_reports(docs, codes)
    print(f"\n有価証券報告書が見つかった銘柄: {len(ar)} / {len(codes)}")
    if len(ar):
        print(ar.to_string(index=False))
        missing = sorted(set(codes) - set(ar["code"]))
        print("見つからない銘柄:", missing if missing else "なし")
        for col in ("ordinanceCode", "formCode"):
            if col in ar.columns:
                print(f"{col} の内訳:", ar[col].astype(str).value_counts().to_dict())

    if a.inspect and len(ar) and a.inspect in set(ar["code"]):
        doc_id = ar.loc[ar["code"] == a.inspect, "docID"].iloc[0]
        print(f"\n=== {a.inspect} の有報 {doc_id} を取得して物件表を探す ===")
        inspect_zip(client.download(doc_id, 1))
