"""EDINET probe の整形ロジック（ネットワーク不要）."""
import pandas as pd

from jreit_score.ingest.edinet import annual_reports, describe_key, find_location_tables, redact, sec_code


def test_sec_code_appends_zero():
    assert sec_code("8972") == "89720" and sec_code("401A") == "401A0"


def test_annual_reports_keeps_latest_per_code_and_filters_type():
    docs = pd.DataFrame([
        {"secCode": "89720", "docID": "A", "docTypeCode": "120", "submitDateTime": "2026-01-28 15:00", "withdrawalStatus": "0"},
        {"secCode": "89720", "docID": "B", "docTypeCode": "120", "submitDateTime": "2026-07-29 15:00", "withdrawalStatus": "0"},
        {"secCode": "89720", "docID": "C", "docTypeCode": "130", "submitDateTime": "2026-08-01 15:00", "withdrawalStatus": "0"},  # 訂正
        {"secCode": "89510", "docID": "D", "docTypeCode": "120", "submitDateTime": "2026-03-30 15:00", "withdrawalStatus": "1"},  # 取下げ
        {"secCode": "72030", "docID": "E", "docTypeCode": "120", "submitDateTime": "2026-06-20 15:00", "withdrawalStatus": "0"},  # 対象外
    ])
    ar = annual_reports(docs, ["8972", "8951"])
    assert ar["code"].tolist() == ["8972"] and ar["docID"].tolist() == ["B"]


def test_annual_reports_empty_input():
    assert annual_reports(pd.DataFrame(), ["8972"]).empty


def test_find_location_tables_detects_property_table_only():
    html = """
    <table><tr><th>物件名称</th><th>所在地</th><th>用途</th></tr>
           <tr><td>テストビル</td><td>東京都中央区日本橋一丁目1番1号</td><td>オフィス</td></tr>
           <tr><td>テストレジデンス</td><td>大阪府大阪市北区梅田一丁目1番1号</td><td>住宅</td></tr></table>
    <table><tr><th>科目</th><th>金額</th></tr><tr><td>営業収益</td><td>1</td></tr><tr><td>営業利益</td><td>2</td></tr></table>
    """
    found = find_location_tables(html)
    assert len(found) == 1
    assert found[0]["shape"] == (2, 3)
    assert "所在地" in " ".join(found[0]["head"])
    assert found[0]["samples"][0].startswith("テスト")


def test_describe_key_never_contains_the_key():
    d = describe_key(" abc123XYZ\n")
    assert "abc123" not in d and "長さ 9" in d and "英数字のみ" in d and "空白/改行あり" in d
    assert describe_key(None) == "未設定"


def test_redact_hides_key():
    assert "abc" not in redact("error key=abc", "abc")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  {name} ok")
    print("ok")
