"""投資法人の公式サイトのポートフォリオ一覧を個人利用で確認する（構造の表示と要約だけ）.

取得元（一次情報）: 各投資法人の IR サイト。KDX は https://www.kdx-reit.com/ja/portfolio/list.html
出力は表の構造（列名・行数）、用途別・地域別の件数と取得価格の合計、取得価格上位の物件名だけ。
物件一覧そのものは出力しない（サイトの転載条件は同時に取得する利用規約リンクで確認する）。
アクセス間隔は 2 秒以上。
"""
from __future__ import annotations

import argparse
import io
import re
import time
from urllib.parse import urljoin

import pandas as pd
import requests
from lxml import html as lxml_html

UA = "Mozilla/5.0 (compatible; jreit-score-prototype/0.1; personal research)"
SOURCES = {"kdx": "https://www.kdx-reit.com/ja/portfolio/list.html"}
NAME_KEYS = ("物件名", "物件名称", "名称")
PRICE_KEYS = ("取得価格", "取得価額")
TYPE_KEYS = ("用途", "アセットタイプ", "タイプ", "分類")
AREA_KEYS = ("地域", "エリア", "所在地", "所在")
POLICY_WORDS = ("利用規約", "サイトポリシー", "ご利用にあたって", "免責", "ご利用条件", "サイトのご利用")


def fetch(url: str, s: requests.Session) -> str:
    r = s.get(url, headers={"User-Agent": UA}, timeout=60)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or "utf-8"
    print(f"GET {url} -> {r.status_code}, {len(r.text)} 文字, content-type {r.headers.get('content-type')}")
    return r.text


def policy_links(page: str, base: str) -> list[tuple[str, str]]:
    doc = lxml_html.fromstring(page)
    out = []
    for a in doc.iter("a"):
        text = " ".join((a.text_content() or "").split())
        href = a.get("href")
        if href and any(w in text for w in POLICY_WORDS):
            out.append((text[:30], urljoin(base, href)))
    seen, uniq = set(), []
    for t, u in out:
        if u not in seen:
            seen.add(u); uniq.append((t, u))
    return uniq


def policy_sentences(page: str, words=("転載", "複製", "禁止", "無断", "著作権", "商用")) -> list[str]:
    text = " ".join(lxml_html.fromstring(page).text_content().split())
    sents = re.split(r"(?<=[。．])", text)
    return [s.strip()[:160] for s in sents if any(w in s for w in words)][:12]


def _col(t: pd.DataFrame, keys) -> str | None:
    for c in t.columns:
        name = " ".join(map(str, c)) if isinstance(c, tuple) else str(c)
        if any(k in name for k in keys):
            return c
    return None


def to_number(v) -> float:
    m = re.search(r"-?[\d,]+(?:\.\d+)?", str(v).replace("，", ","))
    return float(m.group(0).replace(",", "")) if m else float("nan")


def inspect_tables(page: str) -> list[pd.DataFrame]:
    try:
        tables = pd.read_html(io.StringIO(page), flavor="lxml")
    except ValueError:
        print("表なし（read_html）")
        return []
    print(f"表 {len(tables)} 件")
    for i, t in enumerate(tables):
        cols = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
        first = [str(v)[:16] for v in t.iloc[:2, 0].tolist()]
        print(f"  table {i}: shape={t.shape} cols={[c[:14] for c in cols[:10]]} 先頭列の例={first}")
    return tables


def summarize(tables: list[pd.DataFrame]) -> None:
    frames = []
    for i, t in enumerate(tables):
        name_c, price_c = _col(t, NAME_KEYS), _col(t, PRICE_KEYS)
        if name_c is None or price_c is None:
            continue
        df = pd.DataFrame({"name": t[name_c].astype(str), "price": t[price_c].map(to_number)})
        type_c, area_c = _col(t, TYPE_KEYS), _col(t, AREA_KEYS)
        df["type"] = t[type_c].astype(str) if type_c is not None else f"table{i}"
        df["area"] = t[area_c].astype(str).str[:6] if area_c is not None else ""
        frames.append(df.dropna(subset=["price"]))
    if not frames:
        print("物件名と取得価格を持つ表が無い（列名の候補を増やす必要あり）")
        return
    df = pd.concat(frames, ignore_index=True)
    # 合計行（「合計」「計」）は除く
    df = df[~df["name"].str.contains("合計|^計$|小計", regex=True)]
    print(f"\n物件 {len(df)} 件, 取得価格の合計 {df['price'].sum():,.0f}（表の単位のまま。百万円なら {df['price'].sum() / 1e6:,.2f} 兆円相当）")
    g = df.groupby("type").agg(n=("name", "size"), price=("price", "sum")).sort_values("price", ascending=False)
    g["share_%"] = (g["price"] / g["price"].sum() * 100).round(1)
    print("用途（または表）別:")
    print(g.to_string())
    if df["area"].str.len().gt(0).any():
        a = df.groupby("area").agg(n=("name", "size"), price=("price", "sum")).sort_values("price", ascending=False).head(12)
        a["share_%"] = (a["price"] / df["price"].sum() * 100).round(1)
        print("所在地（先頭6文字）別 上位:")
        print(a.to_string())
    print("取得価格 上位 10 件:")
    print(df.nlargest(10, "price")[["name", "type", "price"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(SOURCES), default="kdx")
    ap.add_argument("--policy", action="store_true", help="利用規約らしきページも取得して転載条件の文を出す")
    ap.add_argument("--sleep", type=float, default=2.0)
    a = ap.parse_args()
    s = requests.Session()
    url = SOURCES[a.source]
    page = fetch(url, s)
    tables = inspect_tables(page)
    summarize(tables)
    links = policy_links(page, url)
    print("\n利用規約らしきリンク:", links if links else "見つからず")
    if a.policy and links:
        time.sleep(a.sleep)
        pol = fetch(links[0][1], s)
        for sent in policy_sentences(pol):
            print("  -", sent)
