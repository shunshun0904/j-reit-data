"""ネットワーク無しでパーサを検証する（実ページ構造を模したフィクスチャ）."""
import pandas as pd
from jreit_score.ingest.japan_reit import parse_ranking_html
from jreit_score.ingest.dpu_history import find_dpu_table, normalize

RANK_HTML = """
<table><tr><th>順位</th><th>コード 投資法人名</th><th>価格騰落率</th><th>分配金利回り</th><th>NAV倍率</th>
<th>時価総額(百万円)</th><th>資産規模(億円)</th><th>棟数</th><th>平均築年数</th><th>NOI利回り</th>
<th>含み損益率</th><th>年額分配金(円)</th><th>自己資本利益率（ROE）</th><th>有利子負債比率</th></tr>
<tr><td></td><td>8951 <a href="/sp/meigara/8951/">日本ビルファンド</a></td><td>-14.04%</td><td>4.11%</td><td>0.99</td>
<td>1,073,667</td><td>15,590</td><td>69</td><td>23.80</td><td>4.16%</td><td>25.74%</td><td>50,060</td><td>6.50%</td><td>42.6%</td></tr>
<tr><td></td><td>401A <a href="/sp/meigara/401A/">霞ヶ関ホテルリート</a></td><td>-8.60%</td><td>6.68%</td><td>0.85</td>
<td>28,414</td><td>492</td><td>15</td><td>3.24</td><td>6.52%</td><td>11.36%</td><td>33,050</td><td>3.16%</td><td>46.4%</td></tr>
</table>"""

DPU_HTML = """
<table><tr><th>その他</th><th>x</th></tr><tr><td>1</td><td>2</td></tr><tr><td>1</td><td>2</td></tr><tr><td>1</td><td>2</td></tr></table>
<table><tr><th>決算期</th><th>1口当たり分配金（円）</th><th>予想分配金</th></tr>
<tr><td>2023年12月期</td><td>2,900</td><td>-</td></tr>
<tr><td>2024年12月期</td><td>3,937</td><td>-</td></tr>
<tr><td>2025年12月期</td><td>4,830</td><td>4,900</td></tr></table>"""


def test_ranking():
    df = parse_ranking_html(RANK_HTML)
    assert list(df["code"]) == ["8951", "401A"]
    assert abs(df.loc[0, "ltv"] - 0.426) < 1e-9
    assert df.loc[0, "mcap_mn"] == 1073667
    assert abs(df.loc[1, "dist_yield"] - 0.0668) < 1e-9
    assert df.loc[0, "name"] == "日本ビルファンド"


def test_dpu():
    t = find_dpu_table(DPU_HTML)
    assert t is not None
    d = normalize(t, "8985")
    assert len(d) == 3
    assert d["period_end"].iloc[-1] == pd.Timestamp("2025-12-31")
    assert d["dpu"].iloc[1] == 3937


if __name__ == "__main__":
    test_ranking(); test_dpu(); print("ok")
