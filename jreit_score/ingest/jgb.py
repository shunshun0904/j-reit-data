"""財務省「国債金利情報」CSV から国債利回りを取得する.

`features.rate_resilience` が要求する `jgb10` (columns = [date, yield], yield は %)
を作るのが目的。

未検証: 取得元 URL と CSV の実構造はこの環境から到達できず確認できていない。
そのため構造を決め打ちせず、「基準日」を含む行をヘッダとして自動検出する汎用
パーサにし、`--inspect` で生の先頭行を表示して手元で当たりを付けられるようにしてある。
パーサを固定する前に必ず `--inspect` を通すこと。

候補URL（要確認）:
  当年分   : https://www.mof.go.jp/jgbs/reference/interest_rate/jgbcm.csv
  全期間分 : https://www.mof.go.jp/jgbs/reference/interest_rate/jgbcm_all.csv
  一覧ページ: https://www.mof.go.jp/jgbs/reference/interest_rate/

利用条件（要確認）: 財務省サイトは政府標準利用規約に基づき出典明記での利用を認めて
いると理解しているが、本プロジェクトでは確認するまで JAPAN-REIT.COM と同様に
生データをコミットしない扱いにする（`data/` は .gitignore）。
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

UA = "Mozilla/5.0 (compatible; jreit-score-prototype/0.1; personal research)"
BASE = "https://www.mof.go.jp/jgbs/reference/interest_rate/"
SOURCES = {"current": BASE + "jgbcm.csv", "all": BASE + "jgbcm_all.csv"}

HEADER_KEY = "基準日"
# 和暦の元号 → 元年の前年（元号 n 年 = base + n 年）
ERA_BASE = {"M": 1867, "T": 1911, "S": 1925, "H": 1988, "R": 2018}
ERA_KANJI = {"明治": "M", "大正": "T", "昭和": "S", "平成": "H", "令和": "R"}
MISSING = {"-", "－", "‐", "―", "", "nan", "None"}


def _to_halfwidth(s: str) -> str:
    """全角英数字・記号を半角にする（列名の '１０年' などを吸収する）."""
    return str(s).translate({c: c - 0xFEE0 for c in range(0xFF01, 0xFF5F)}).replace("　", " ")


def parse_jp_date(s) -> pd.Timestamp | None:
    """和暦 'S49.9.24' / '令和6年4月1日' と西暦 '2024-04-01' / '2024/4/1' を解釈する."""
    t = _to_halfwidth(s).strip()
    if not t or t in MISSING:
        return None
    for kanji, letter in ERA_KANJI.items():
        if t.startswith(kanji):
            t = letter + t[len(kanji):]
            break
    m = re.match(r"^([MTSHR])\s*(\d{1,2})\D+(\d{1,2})\D+(\d{1,2})", t)
    if m:
        era, y, mo, d = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        try:
            return pd.Timestamp(year=ERA_BASE[era] + y, month=mo, day=d)
        except ValueError:
            return None
    m = re.match(r"^(\d{4})\D+(\d{1,2})\D+(\d{1,2})", t)
    if m:
        try:
            return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=int(m.group(3)))
        except ValueError:
            return None
    return None


def find_header_row(text: str) -> int:
    """「基準日」を含む行の番号を返す. 見つからなければ -1."""
    for i, line in enumerate(text.splitlines()):
        if HEADER_KEY in _to_halfwidth(line):
            return i
    return -1


def parse_jgb_csv(text: str) -> pd.DataFrame:
    """国債金利情報 CSV を wide 形式 (date + 年限列) に整形する.

    値は % のまま。欠損記号（'-' など）は NaN にする。
    """
    h = find_header_row(text)
    if h < 0:
        raise ValueError(
            f"ヘッダ行（'{HEADER_KEY}' を含む行）が見つからない。"
            "--inspect で実構造を確認してからパーサを直すこと"
        )
    df = pd.read_csv(io.StringIO(text), skiprows=h, dtype=str)
    df.columns = [_to_halfwidth(c).strip() for c in df.columns]
    date_col = df.columns[0]
    out = pd.DataFrame({"date": df[date_col].map(parse_jp_date)})
    for c in df.columns[1:]:
        out[c] = pd.to_numeric(
            df[c].astype(str).str.strip().replace(list(MISSING), np.nan), errors="coerce"
        )
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if out.empty:
        raise ValueError("日付を1件も解釈できなかった。--inspect で日付表記を確認すること")
    return out


def tenor(df: pd.DataFrame, name: str = "10年") -> pd.DataFrame:
    """wide 形式から特定年限だけを [date, yield] で取り出す（features 側の入力形式）."""
    key = _to_halfwidth(name).strip()
    cols = [c for c in df.columns if c != "date" and _to_halfwidth(c).strip() == key]
    if not cols:
        raise KeyError(f"年限 '{name}' が無い。利用可能: {[c for c in df.columns if c != 'date']}")
    return df[["date", cols[0]]].rename(columns={cols[0]: "yield"}).dropna().reset_index(drop=True)


def fetch_jgb_csv(source: str = "all", session: requests.Session | None = None) -> str:
    """CSV を文字列で取得する. 財務省 CSV は Shift_JIS 系のため cp932 を優先して復号する."""
    s = session or requests.Session()
    r = s.get(SOURCES[source], headers={"User-Agent": UA}, timeout=60)
    r.raise_for_status()
    for enc in ("cp932", "utf-8-sig", "utf-8"):
        try:
            return r.content.decode(enc)
        except UnicodeDecodeError:
            continue
    return r.content.decode(r.apparent_encoding or "cp932", errors="replace")


def fetch_jgb10(source: str = "all", session: requests.Session | None = None) -> pd.DataFrame:
    """10年債利回りを [date, yield] で返す."""
    return tenor(parse_jgb_csv(fetch_jgb_csv(source, session)), "10年")


def save(df: pd.DataFrame, root: Path, name: str = "jgb.parquet") -> Path:
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return p


def inspect(text: str, n_lines: int = 12) -> str:
    """生の先頭行と検出したヘッダ・年限列を返す（実構造の確認用）."""
    lines = text.splitlines()
    h = find_header_row(text)
    out = [f"総行数: {len(lines)}", f"ヘッダ行: {h if h >= 0 else '検出できず'}", "--- 先頭行 ---"]
    out += [f"{i:>3}: {ln[:200]}" for i, ln in enumerate(lines[:n_lines])]
    if h >= 0:
        try:
            df = parse_jgb_csv(text)
        except ValueError as e:
            out.append(f"--- パース失敗: {e} ---")
        else:
            tenors = [c for c in df.columns if c != "date"]
            out += ["--- パース結果 ---",
                    f"行数: {len(df)}",
                    f"期間: {df['date'].min().date()} 〜 {df['date'].max().date()}",
                    f"年限列: {tenors}",
                    f"10年列の非欠損数: {int(df['10年'].notna().sum()) if '10年' in df else '10年列なし'}"]
    return "\n".join(out)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(SOURCES), default="all")
    ap.add_argument("--inspect", action="store_true", help="生の先頭行と検出結果を表示する")
    ap.add_argument("--tenor", default="10年")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    text = fetch_jgb_csv(a.source)
    if a.inspect:
        print(inspect(text))
    else:
        df = tenor(parse_jgb_csv(text), a.tenor)
        print(f"saved {len(df)} rows -> {save(df, Path(a.out))}")
        print(df.tail())
