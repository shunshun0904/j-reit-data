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
TOP_PAGES = {"kdx": "https://www.kdx-reit.com/"}      # 分配金ページの URL は推測せず、トップからリンクを辿る
DIST_WORDS = ("分配金",)
PERIOD_KEYS = ("期", "決算期")
DPU_KEYS = ("分配金", "1口当たり分配金", "１口当たり分配金")
NAME_KEYS = ("物件名", "物件名称", "名称")
PRICE_KEYS = ("取得価格", "取得価額")
APPRAISAL_KEYS = ("鑑定評価額", "鑑定 評価額", "期末算定価額", "鑑定")
GFA_KEYS = ("延床面積",)
DATE_KEYS = ("取得日",)
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
        appr_c, gfa_c, date_c = _col(t, APPRAISAL_KEYS), _col(t, GFA_KEYS), _col(t, DATE_KEYS)
        df["type"] = t[type_c].astype(str) if type_c is not None else f"table{i}"
        df["area"] = t[area_c].astype(str).str[:6] if area_c is not None else ""
        df["appraisal"] = t[appr_c].map(to_number) if appr_c is not None else float("nan")
        df["gfa"] = t[gfa_c].map(to_number) if gfa_c is not None else float("nan")
        df["acq_year"] = (pd.to_datetime(t[date_c].astype(str).str.extract(r"(\d{4}[./年]\d{1,2}[./月]\d{1,2})")[0]
                                         .str.replace("年", "/").str.replace("月", "/").str.replace("日", ""),
                                         errors="coerce").dt.year if date_c is not None else float("nan"))
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
    if df["appraisal"].notna().any():
        a = df.dropna(subset=["appraisal"])
        g2 = a.groupby("type").agg(price=("price", "sum"), appraisal=("appraisal", "sum"))
        g2["appr/price"] = (g2["appraisal"] / g2["price"]).round(3)
        tot = a["appraisal"].sum() / a["price"].sum()
        print(f"鑑定評価額（表の単位）: 合計 {a['appraisal'].sum():,.0f}, 取得価格比 {tot:.3f}（{len(a)} 件）")
        print(g2.sort_values("price", ascending=False).to_string())
    if df["gfa"].notna().any():
        print(f"延床面積の合計 {df['gfa'].sum():,.0f} ㎡（{df['gfa'].notna().sum()} 件）")
    if df["acq_year"].notna().any():
        y = df.dropna(subset=["acq_year"]).groupby(df["acq_year"].astype("Int64")).agg(n=("name", "size"), price=("price", "sum"))
        print("取得年別（件数, 取得価格）:")
        print(y.to_string())
    print("取得価格 上位 10 件:")
    print(df.nlargest(10, "price")[["name", "type", "price", "appraisal"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(SOURCES), default="kdx")
    ap.add_argument("--mode", choices=["portfolio", "distribution"], default="portfolio")
    ap.add_argument("--policy", action="store_true", help="利用規約らしきページも取得して転載条件の文を出す")
    ap.add_argument("--jq-dpu", default="", help="J-Quants cache のディレクトリ（分割調整後の DPU 変化率を出す）")
    ap.add_argument("--code", default="8972")
    ap.add_argument("--splits", default="2022-11-01:2,2023-11-01:2", help="分割の効力日:比率 をカンマ区切り")
    ap.add_argument("--sleep", type=float, default=2.0)
    a = ap.parse_args()
    s = requests.Session()
    if a.mode == "portfolio":
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
    else:
        top = fetch(TOP_PAGES[a.source], s)
        links = find_links(top, TOP_PAGES[a.source])
        print("「分配金」を含むリンク:", links if links else "見つからず")
        if links:
            time.sleep(a.sleep)
            page = fetch(links[0][1], s)
            tables = inspect_tables(page)
            summarize_distribution(parse_distribution(tables))
        if a.jq_dpu:
            splits = [(pd.Timestamp(x.split(":")[0]), float(x.split(":")[1])) for x in a.splits.split(",") if x]
            print()
            jq_dpu_growth(a.jq_dpu, a.code, splits)


# ---------------------------------------------------------------------------
# 分配金の推移（公式 IR ページ）と、J-Quants の DPU 履歴からの分割調整後の変化率
# ---------------------------------------------------------------------------

def find_links(page: str, base: str, words=DIST_WORDS) -> list[tuple[str, str]]:
    doc = lxml_html.fromstring(page)
    out, seen = [], set()
    for a in doc.iter("a"):
        text = " ".join((a.text_content() or "").split())
        href = a.get("href")
        if href and any(w in text for w in words):
            u = urljoin(base, href)
            if u not in seen:
                seen.add(u); out.append((text[:30], u))
    return out


def parse_distribution(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """[period, label, dpu] を返す. 縦型（期が行）と横型（期が列）の両方を試す."""
    for t in tables:
        cols = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
        # 縦型: 列に「期」と「分配金」
        pc = next((c for c, n in zip(t.columns, cols) if any(k in n for k in PERIOD_KEYS)), None)
        dc = next((c for c, n in zip(t.columns, cols) if any(k in n for k in DPU_KEYS) and "利回り" not in n), None)
        if pc is not None and dc is not None and len(t) >= 3:
            df = pd.DataFrame({"label": t[pc].astype(str), "dpu": t[dc].map(to_number)}).dropna(subset=["dpu"])
            if len(df) >= 3:
                df["period"] = df["label"]
                return df.reset_index(drop=True)
        # 横型: 先頭列の行ラベルに「分配金」
        labels = t.iloc[:, 0].astype("string").fillna("")
        hit = labels[labels.str.contains("|".join(DPU_KEYS), regex=True) & ~labels.str.contains("利回り|予想")]
        if len(hit) and t.shape[1] >= 4:
            row = t.loc[hit.index[0]]
            df = pd.DataFrame({"label": [str(c) for c in cols[1:]], "dpu": [to_number(v) for v in row.iloc[1:]]}).dropna(subset=["dpu"])
            if len(df) >= 3:
                df["period"] = df["label"]
                return df.reset_index(drop=True)
    return pd.DataFrame(columns=["period", "label", "dpu"])


def summarize_distribution(df: pd.DataFrame) -> None:
    if df.empty:
        print("分配金の表を解釈できず（構造を見て列名の候補を増やす）")
        return
    actual = df[~df["label"].str.contains("予想")]
    print(f"分配金の系列 {len(df)} 期（うち実績 {len(actual)}）: {df['label'].iloc[0]} 〜 {df['label'].iloc[-1]}")
    tail = df.tail(3)
    print("  直近 3 期:", ", ".join(f"{r.label} {r.dpu:,.0f}円" for r in tail.itertuples()))
    if len(actual) >= 3:
        a = actual["dpu"].to_numpy()
        chg = pd.Series(a[1:] / a[:-1] - 1)
        print(f"  実績の前期比: 増配 {int((chg > 0).sum())} 回, 減配 {int((chg < 0).sum())} 回, 据え置き {int((chg == 0).sum())} 回")
        if len(a) >= 3:
            print(f"  実績の前年同期比（直近）: {a[-1] / a[-3] - 1:+.1%}")
        yrs = (len(a) - 1) / 2
        print(f"  実績の年率成長（半期 2 回/年として {yrs:.1f} 年）: {(a[-1] / a[0]) ** (1 / yrs) - 1:+.1%}  先頭 {a[0]:,.0f}円 → 直近 {a[-1]:,.0f}円")


def split_factor(period_end: pd.Timestamp, splits: list[tuple[pd.Timestamp, float]]) -> float:
    """分割効力日より前に終わる期の DPU を現在の 1 口に換算する係数（1/比率の積）."""
    f = 1.0
    for d, ratio in splits:
        if period_end < d:
            f /= ratio
    return f


def jq_dpu_growth(root: str, code: str, splits: list[tuple[pd.Timestamp, float]]) -> None:
    """J-Quants cache の DPU 履歴から、分割調整後の変化率だけを出す（金額は出さない）."""
    from pathlib import Path
    p = Path(root) / "dpu.parquet"
    if not p.exists():
        print("J-Quants の dpu.parquet が無い（cache 未復元）")
        return
    d = pd.read_parquet(p)
    d = d[d["code"].astype(str) == code].sort_values("period_end")
    if d.empty:
        print(f"J-Quants に {code} の DPU 無し")
        return
    d["adj"] = [r.dpu * split_factor(pd.Timestamp(r.period_end), splits) for r in d.itertuples()]
    d["pop"] = d["adj"].pct_change()
    d["yoy"] = d["adj"].pct_change(2)
    print(f"J-Quants の DPU 履歴（{code}, {len(d)} 期, {d['period_end'].min().date()} 〜 {d['period_end'].max().date()}, "
          f"分割 {len(splits)} 回を調整。変化率のみ）:")
    out = d[["period_end", "pop", "yoy"]].copy()
    out["period_end"] = out["period_end"].dt.date
    out["pop"] = out["pop"].map(lambda v: "" if pd.isna(v) else f"{v:+.1%}")
    out["yoy"] = out["yoy"].map(lambda v: "" if pd.isna(v) else f"{v:+.1%}")
    print(out.to_string(index=False))
    a = d["adj"].to_numpy()
    yrs = (len(a) - 1) / 2
    print(f"  調整後 DPU の先頭→直近: {a[-1] / a[0] - 1:+.1%}（{yrs:.1f} 年, 年率 {(a[-1] / a[0]) ** (1 / yrs) - 1:+.1%}）, "
          f"前期比で増配 {int((d['pop'] > 0).sum())} 回 / 減配 {int((d['pop'] < 0).sum())} 回")
