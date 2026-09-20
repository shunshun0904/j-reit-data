"""銘柄別の DPU（1口当たり分配金）履歴の取得.

取得元ページはこの環境から到達できず、表の構造を確認できていない。
そのため「決算期」「分配金」を含む表を自動検出する汎用パーサにし、
--inspect で全表のヘッダを表示して手元で当たりを付けられるようにしてある。

確認済み（Actions 実行 2026-09-20, workflow inspect-sources, 銘柄 8985）:
  https://www.japan-reit.com/meigara/{code}/ には表が6件ある。DPU は table 2
    cols       = ['Unnamed: 0', '前期', '当期', '次期']
    row_labels = ['期首', '期末', '営業収益', '当期利益', '1口分配金']
  つまり期が列・項目が行の転置レイアウトで、期は「前期/当期/次期」の3つしかない。
  うち次期は予想なので実績は2期分。`features.dpu_stability` は実績3期以上
  （既定 window=6）を要求するため、このページだけでは DPU 履歴を作れない。
  → 履歴の取得元は別途決める必要がある（未決）。

  なお table 0 / table 1 の行ラベルにある「分配金利回り」は DPU ではないので、
  行ラベル検出では `DPU_ROW_EXCLUDE` で除外している。

確認済み（同上, haitoukabu.com/reit/8985.html）: 表は5件で DPU 履歴は無い。
  3件はナビゲーション、table 3 は銘柄プロフィール (18, 2)、table 4 は株価上昇率 (3, 2)。
  2列の表は行ラベルごとに値が1つなので、形状の時点で履歴を保持できない。

結論: この2サイトの銘柄ページからは DPU 履歴を取れない。取得元は J-Quants V2 に一本化する
（`ingest/jquants.py`）。本モジュールは参考として残すが、DPU 履歴の取得には使わない。
"""
from __future__ import annotations

import io
import re
import time
from pathlib import Path

import pandas as pd
import requests

UA = "Mozilla/5.0 (compatible; jreit-score-prototype/0.1; personal research)"
SOURCES = {
    "japan_reit": "https://www.japan-reit.com/meigara/{code}/",
    "haitoukabu": "https://haitoukabu.com/reit/{code}.html",
}
PERIOD_KEYS = ("決算期", "期", "決算")
DPU_KEYS = ("分配金", "1口当たり", "１口当たり")
# 行ラベル検出で DPU と紛らわしいもの。「分配金利回り」は DPU ではなく利回り
DPU_ROW_EXCLUDE = ("利回り", "予想")
PERIOD_END_KEYS = ("期末", "決算期")


def _parse_period(s: str) -> pd.Timestamp | None:
    """'2025年12月期' / '2025/12' / '2025年12月' などを期末日に変換."""
    m = re.search(r"(\d{4})\D+(\d{1,2})", str(s))
    if not m:
        return None
    y, mo = int(m.group(1)), int(m.group(2))
    return pd.Timestamp(year=y, month=mo, day=1) + pd.offsets.MonthEnd(0)


def find_dpu_table(html: str) -> pd.DataFrame | None:
    """「決算期」と「分配金」を列名に含む表を探す.

    取得元により DPU が行ラベル側に来る場合があるため `find_dpu_rows` も併せて使う。
    """
    tables = pd.read_html(io.StringIO(html), flavor="lxml")
    for t in tables:
        cols = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
        has_period = any(any(k in c for k in PERIOD_KEYS) for c in cols)
        has_dpu = any(any(k in c for k in DPU_KEYS) for c in cols)
        if has_period and has_dpu and len(t) >= 3:
            t.columns = cols
            return t
    return None


def find_dpu_rows(html: str) -> tuple[pd.DataFrame, str] | None:
    """分配金が「行」側に来ている表を探す（列が期、行が項目の転置レイアウト）.

    JAPAN-REIT.COM の銘柄ページは `Unnamed: 0, 前期, 当期, 次期` のように
    期が列、項目が行ラベルになっている（2026-09-20 の inspect で確認）。
    返り値は (表, 分配金の行ラベル)。
    """
    include = "|".join(map(re.escape, DPU_KEYS))
    exclude = "|".join(map(re.escape, DPU_ROW_EXCLUDE))
    for t in pd.read_html(io.StringIO(html), flavor="lxml"):
        if t.shape[1] < 2:
            continue
        # pandas 3 の arrow 文字列型では NaN が float のまま渡るため、
        # fillna してからベクトル化した contains を使う
        labels = t.iloc[:, 0].astype("string").fillna("")
        hit = labels[labels.str.contains(include, regex=True)
                     & ~labels.str.contains(exclude, regex=True)]
        if len(hit):
            return t, str(hit.iloc[0])
    return None


def normalize(t: pd.DataFrame, code: str) -> pd.DataFrame:
    pcol = next(c for c in t.columns if any(k in c for k in PERIOD_KEYS))
    # 「予想」を含む列は除き、実績の分配金列を優先
    dcols = [c for c in t.columns if any(k in c for k in DPU_KEYS) and "予想" not in c] or \
            [c for c in t.columns if any(k in c for k in DPU_KEYS)]
    dcol = dcols[0]
    out = pd.DataFrame({
        "code": code,
        "period_end": t[pcol].map(_parse_period),
        "dpu": pd.to_numeric(t[dcol].astype(str).str.replace(",", "").str.replace("円", ""), errors="coerce"),
    }).dropna()
    return out.sort_values("period_end").reset_index(drop=True)


def fetch_dpu(code: str, source: str = "japan_reit", session: requests.Session | None = None,
              inspect: bool = False) -> pd.DataFrame | None:
    s = session or requests.Session()
    url = SOURCES[source].format(code=code)
    r = s.get(url, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    r.encoding = r.apparent_encoding
    if inspect:
        tables = pd.read_html(io.StringIO(r.text), flavor="lxml")
        print(f"[{code} {source}] {url}: 表 {len(tables)} 件")
        for i, t in enumerate(tables):
            cols = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
            # 行ラベル（先頭列）も出す。どの表が何かは列名だけでは分からないため。
            # 数値セルは出さない（JAPAN-REIT.COM は転載・複製禁止）
            shown = 24
            vals = t.iloc[:, 0].tolist()
            labels = [str(v)[:24] for v in vals[:shown]]
            more = f" (+{len(vals) - shown} 件省略)" if len(vals) > shown else ""
            print(f"  table {i}: shape={t.shape}")
            print(f"    cols      : {cols[:12]}")
            print(f"    row_labels: {labels}{more}")
        print(f"  find_dpu_table: {'該当あり' if find_dpu_table(r.text) is not None else '該当なし'}")
        return None
    t = find_dpu_table(r.text)
    return normalize(t, code) if t is not None else None


def fetch_all(codes: list[str], source: str, out_root: Path, sleep_sec: float = 2.0) -> pd.DataFrame:
    s = requests.Session()
    frames = []
    for c in codes:
        try:
            df = fetch_dpu(c, source, s)
            if df is not None:
                frames.append(df)
            else:
                print(f"[warn] no DPU table: {c}")
        except Exception as e:
            print(f"[warn] {c}: {e}")
        time.sleep(sleep_sec)  # サイト負荷配慮
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    p = out_root / "dpu_history.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("codes", nargs="*", default=["8985"])
    ap.add_argument("--source", choices=list(SOURCES), default="japan_reit")
    ap.add_argument("--inspect", action="store_true", help="表のヘッダ一覧だけ表示")
    ap.add_argument("--out", default="data")
    ap.add_argument("--sleep", type=float, default=2.0, help="連続取得の間隔（秒）")
    a = ap.parse_args()
    if a.inspect:
        # セッションを共有し, 銘柄間は必ず間隔を空ける（サイト負荷配慮）
        s = requests.Session()
        for i, c in enumerate(a.codes):
            if i:
                time.sleep(a.sleep)
            fetch_dpu(c, a.source, session=s, inspect=True)
    else:
        print(fetch_all(a.codes, a.source, Path(a.out), sleep_sec=a.sleep))
