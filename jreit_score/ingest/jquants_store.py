"""J-Quants から取得した生データの差分取得と保存（parquet）.

保存先は Actions の cache（設計判断 2026-09-21）。生データはリポジトリにコミットしない
（J-Quants のデータは再配布不可。`data/` は .gitignore）。
cache は同じキーでは上書きされないため、ワークフロー側でキーに run_id を含め、
restore-keys の前方一致で直近のものを復元し、毎回新しいエントリとして保存する。

  data/jquants/universe.parquet  code, code5, name
  data/jquants/prices.parquet    code, date, close(AdjC), dividend(=0), mktcap[百万円]
  data/jquants/dpu.parquet       code, period_start, period_end, dpu, ex_date, bps, disc_date
  （prices は features.total_return_index、dpu は features.dpu_stability の入力。
    mktcap / bps / disc_date は説明変数 log_mcap / nav_ratio の材料。cache キーは v2）

価格は銘柄ごとに保存済みの最終日の翌日から差分取得する。分配金は決算短信の要約で
小さい（銘柄あたり数十行）ので毎回全件を取り直し、訂正開示を取りこぼさない。
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .jquants import (SLEEP, Client, exclude_annual, fetch_dpu, fetch_prices, redact,
                      reit_universe)

# 希望する価格の取得開始日。実際にはプランの対象期間で切り詰められる（下記）。
PRICE_START = "20160801"

# プランの対象期間外を要求すると bars/daily が HTTP 400 を返し、本文に対象期間が書かれる:
#   "Your subscription covers the following dates: 2016-09-21 ~ . If you want more data, ..."
# 現プランは「今日から遡って 10 年」で、窓は日々前へ動く（2026-09-21 に確認）。
# 開始日を固定すると翌日には範囲外になるので、400 の本文から開始日を読み取って取り直す。
_COVERS = re.compile(r"covers the following dates:\s*(\d{4}-\d{2}-\d{2})\s*~\s*(\d{4}-\d{2}-\d{2})?")
FILES = {"universe": "universe.parquet", "prices": "prices.parquet", "dpu": "dpu.parquet"}


@dataclass
class Store:
    universe: pd.DataFrame
    prices: pd.DataFrame
    dpu: pd.DataFrame

    def summary(self) -> dict:
        """件数と期間だけ（値は出さない）."""
        out = {"universe": int(len(self.universe))}
        if len(self.prices):
            out.update(prices_rows=int(len(self.prices)), prices_codes=int(self.prices["code"].nunique()),
                       prices_range=(str(self.prices["date"].min().date()), str(self.prices["date"].max().date())))
        else:
            out.update(prices_rows=0)
        if len(self.dpu):
            out.update(dpu_rows=int(len(self.dpu)), dpu_codes=int(self.dpu["code"].nunique()),
                       dpu_range=(str(self.dpu["period_end"].min().date()), str(self.dpu["period_end"].max().date())))
        else:
            out.update(dpu_rows=0)
        return out


def _empty_prices() -> pd.DataFrame:
    return pd.DataFrame({"code": pd.Series(dtype=str), "date": pd.Series(dtype="datetime64[ns]"),
                         "close": pd.Series(dtype="float64"), "dividend": pd.Series(dtype="float64"),
                         "mktcap": pd.Series(dtype="float64")})


def load(root: Path) -> Store:
    """保存済みがあれば読む. 無ければ空."""
    def rd(name, empty):
        p = root / FILES[name]
        return pd.read_parquet(p) if p.exists() else empty
    return Store(
        universe=rd("universe", pd.DataFrame(columns=["code", "code5", "name"])),
        prices=rd("prices", _empty_prices()),
        dpu=rd("dpu", pd.DataFrame(columns=["code", "period_start", "period_end", "dpu", "ex_date",
                                            "bps", "disc_date"])),
    )


def save(store: Store, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    store.universe.to_parquet(root / FILES["universe"], index=False)
    store.prices.to_parquet(root / FILES["prices"], index=False)
    store.dpu.to_parquet(root / FILES["dpu"], index=False)


def merge_prices(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """差分を結合する. 同じ (code, date) は新しい方（分割調整で AdjC が変わりうる）."""
    df = pd.concat([old, new], ignore_index=True)
    if df.empty:
        return _empty_prices()
    return (df.sort_values(["code", "date"])
              .drop_duplicates(subset=["code", "date"], keep="last")
              .reset_index(drop=True))


def next_from(old: pd.DataFrame, code: str, default: str = PRICE_START) -> str:
    """銘柄の保存済み最終日の翌日 (YYYYMMDD). 無ければ default."""
    s = old.loc[old["code"].astype(str) == code, "date"] if len(old) else pd.Series([], dtype="datetime64[ns]")
    if s.empty:
        return default
    return (s.max() + pd.Timedelta(days=1)).strftime("%Y%m%d")


def covered_start(message: str) -> str | None:
    """400 の本文からプランの対象開始日 (YYYYMMDD) を読む. 無ければ None."""
    m = _COVERS.search(message or "")
    return m.group(1).replace("-", "") if m else None


def fetch_prices_clamped(client: Client, code: str, from_yyyymmdd: str, to_yyyymmdd: str
                         ) -> tuple[pd.DataFrame, str]:
    """対象期間外なら本文の開始日から取り直す. 返り値は (価格, 実際の開始日)."""
    try:
        return fetch_prices(client, code, from_yyyymmdd, to_yyyymmdd), from_yyyymmdd
    except RuntimeError as e:
        start = covered_start(str(e))
        if not start or start <= from_yyyymmdd:
            raise
        return fetch_prices(client, code, start, to_yyyymmdd), start


def update(client: Client, store: Store, to_yyyymmdd: str | None = None,
           sleep: float = SLEEP, codes: list[str] | None = None) -> tuple[Store, list[tuple[str, str]]]:
    """母集団を取り直し、価格は差分、DPU は全件を取得して store を更新する.

    返り値は (更新後の store, 取得に失敗した (code, 理由))。
    年次決算の銘柄は DPU 側で除外し、価格の母集団もそれに合わせる。
    """
    to = to_yyyymmdd or pd.Timestamp.today().strftime("%Y%m%d")
    uni = reit_universe(client)
    if codes:
        uni = uni[uni["code"].isin(codes)].reset_index(drop=True)
    failed: list[tuple[str, str]] = []

    dpu_frames = []
    for code in uni["code"]:
        try:
            dpu_frames.append(fetch_dpu(client, code))
        except Exception as e:
            failed.append((code, "dpu: " + redact(str(e))[:240]))
        time.sleep(sleep)
    dpu = pd.concat(dpu_frames, ignore_index=True) if dpu_frames else store.dpu.iloc[0:0]
    dpu, annual = exclude_annual(dpu)
    uni = uni[~uni["code"].isin(annual)].reset_index(drop=True)

    prices = store.prices[store.prices["code"].isin(uni["code"])] if len(store.prices) else store.prices
    new_frames, clamped = [], {}
    for code in uni["code"]:
        frm = next_from(prices, code)
        if frm > to:
            continue
        try:
            df, actual = fetch_prices_clamped(client, code, frm, to)
            new_frames.append(df)
            if actual != frm:
                clamped[code] = actual
        except Exception as e:
            failed.append((code, "prices: " + redact(str(e))[:240]))
        time.sleep(sleep)
    if new_frames:
        prices = merge_prices(prices, pd.concat(new_frames, ignore_index=True))
    if clamped:
        starts = sorted(set(clamped.values()))
        print(f"プランの対象期間で開始日を切り詰めた銘柄: {len(clamped)} 件（開始日 {starts}）")
    return Store(universe=uni, prices=prices, dpu=dpu), failed


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/jquants", help="parquet の保存先（cache で復元・保存する）")
    ap.add_argument("--to", default=None, help="価格の取得終了日 YYYYMMDD（既定は今日）")
    ap.add_argument("--codes", nargs="*", default=None, help="銘柄を絞る（動作確認用）")
    a = ap.parse_args()
    root = Path(a.data)
    before = load(root)
    print("復元:", before.summary())
    after, failed = update(Client(), before, a.to, codes=a.codes)
    save(after, root)
    print("保存:", after.summary())
    print(f"失敗 {len(failed)} 件:")
    for code, why in failed[:10]:
        print(f"  {code}: {why}")
    for p in sorted(root.glob("*.parquet")):
        print(f"  {p.name}: {p.stat().st_size:,} bytes")
    if failed and len(failed) >= max(3, len(after.universe) // 4):
        raise SystemExit("取得失敗が多すぎる")
