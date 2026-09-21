"""J-Quants から取得した生データの差分取得と保存（parquet）.

保存先は Actions の cache（設計判断 2026-09-21）。生データはリポジトリにコミットしない
（J-Quants のデータは再配布不可。`data/` は .gitignore）。
cache は同じキーでは上書きされないため、ワークフロー側でキーに run_id を含め、
restore-keys の前方一致で直近のものを復元し、毎回新しいエントリとして保存する。

  data/jquants/universe.parquet  code, code5, name
  data/jquants/prices.parquet    code, date, close(AdjC), dividend(=0)   ← features.total_return_index
  data/jquants/dpu.parquet       code, period_start, period_end, dpu, ex_date ← features.dpu_stability

価格は銘柄ごとに保存済みの最終日の翌日から差分取得する。分配金は決算短信の要約で
小さい（銘柄あたり数十行）ので毎回全件を取り直し、訂正開示を取りこぼさない。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .jquants import (SLEEP, Client, exclude_annual, fetch_dpu, fetch_prices, redact,
                      reit_universe)

PRICE_START = "20150601"   # DPU 履歴は 2016-08 から。rate_resilience の lookback 250 日ぶん手前から
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
                         "close": pd.Series(dtype="float64"), "dividend": pd.Series(dtype="float64")})


def load(root: Path) -> Store:
    """保存済みがあれば読む. 無ければ空."""
    def rd(name, empty):
        p = root / FILES[name]
        return pd.read_parquet(p) if p.exists() else empty
    return Store(
        universe=rd("universe", pd.DataFrame(columns=["code", "code5", "name"])),
        prices=rd("prices", _empty_prices()),
        dpu=rd("dpu", pd.DataFrame(columns=["code", "period_start", "period_end", "dpu", "ex_date"])),
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
            failed.append((code, "dpu: " + redact(str(e))[:80]))
        time.sleep(sleep)
    dpu = pd.concat(dpu_frames, ignore_index=True) if dpu_frames else store.dpu.iloc[0:0]
    dpu, annual = exclude_annual(dpu)
    uni = uni[~uni["code"].isin(annual)].reset_index(drop=True)

    prices = store.prices[store.prices["code"].isin(uni["code"])] if len(store.prices) else store.prices
    new_frames = []
    for code in uni["code"]:
        frm = next_from(prices, code)
        if frm > to:
            continue
        try:
            new_frames.append(fetch_prices(client, code, frm, to))
        except Exception as e:
            failed.append((code, "prices: " + redact(str(e))[:80]))
        time.sleep(sleep)
    if new_frames:
        prices = merge_prices(prices, pd.concat(new_frames, ignore_index=True))
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
    print(f"失敗 {len(failed)} 件:", failed[:10])
    for p in sorted(root.glob("*.parquet")):
        print(f"  {p.name}: {p.stat().st_size:,} bytes")
    if failed and len(failed) >= max(3, len(after.universe) // 4):
        raise SystemExit("取得失敗が多すぎる")
