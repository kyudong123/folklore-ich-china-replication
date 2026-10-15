from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "source" / "national_ich_all_categories_source_snapshot.json"
SOURCE_RECORDS = ROOT / "data" / "source" / "folklore_records_with_coordinates.csv"
AUDIT = ROOT / "data" / "intermediate" / "record_inclusion_geocoding_audit.csv"
POP = ROOT / "data" / "source" / "province_demographics_2020_derived_from_census.csv"
REFERENCE = ROOT / "data" / "processed" / "provincial_analysis_panel_2020.csv"
OUT = ROOT / "results" / "rebuilt_inputs"

PROVINCE_MAP = {'北京市': '北京', '天津市': '天津', '河北省': '河北', '山西省': '山西', '内蒙古自治区': '内蒙古', '辽宁省': '辽宁', '吉林省': '吉林', '黑龙江省': '黑龙江', '上海市': '上海', '江苏省': '江苏', '浙江省': '浙江', '安徽省': '安徽', '福建省': '福建', '江西省': '江西', '山东省': '山东', '河南省': '河南', '湖北省': '湖北', '湖南省': '湖南', '广东省': '广东', '广西壮族自治区': '广西', '海南省': '海南', '重庆市': '重庆', '四川省': '四川', '贵州省': '贵州', '云南省': '云南', '西藏自治区': '西藏', '陕西省': '陕西', '甘肃省': '甘肃', '青海省': '青海', '宁夏回族自治区': '宁夏', '新疆维吾尔自治区': '新疆'}

def main():
    with RAW.open(encoding="utf-8") as f: raw = json.load(f)
    full_records = raw["records"]
    full_ids = {str(r["id"]) for r in full_records}
    folk_ids = {str(r["id"]) for r in full_records if r.get("type") == "民俗"}
    audit = pd.read_csv(AUDIT, encoding="utf-8-sig", dtype={"source_id": str})
    source_records = pd.read_csv(SOURCE_RECORDS, encoding="utf-8-sig", dtype={"官网记录ID": str})
    audit_ids = set(audit["source_id"].astype(str))
    source_ids = set(source_records["官网记录ID"].astype(str))
    if len(full_records) != 3610 or len(full_ids) != 3610: raise ValueError("Expected 3,610 unique records in the five-batch source snapshot")
    if len(folk_ids) != 492 or folk_ids != audit_ids or folk_ids != source_ids: raise ValueError("The official folklore-category IDs do not match the 492-row audit/source table")
    pop = pd.read_csv(POP, encoding="utf-8-sig", dtype={"province": str})
    if len(audit) != 492 or audit["source_id"].nunique() != 492:
        raise ValueError("Expected 492 unique official folklore source IDs")
    if not audit["official_category"].astype(str).eq("民俗").all():
        raise ValueError("The adjudicated sample must contain only the official folklore category")
    if int(audit["include_regional_map"].sum()) != 481:
        raise ValueError("Expected 481 regionally assigned records")
    main = audit.loc[audit["include_population_spatial_analysis"].eq(1)].copy()
    if len(main) != 471:
        raise ValueError("Expected 471 mainland records for 31-province analyses")
    main["province"] = main["province_match"].map(PROVINCE_MAP)
    if main["province"].isna().any(): raise ValueError("Unmapped province names in the inclusion audit")
    counts = (main.groupby("province", sort=False).agg(
        folklore_count=("source_id", "size"), source_ids=("source_id", lambda s: ";".join(s.astype(str)))
    ).reset_index())
    popcols = ["province", "pop_2020", "minority_pop_2020", "minority_pct_2020",
               "minority_share_2020", "han_pop_2020", "han_pct_2020", "naturalized_pop_2020"]
    missing = set(popcols).difference(pop.columns)
    if missing: raise ValueError(f"Population source is missing columns: {sorted(missing)}")
    panel = pop[popcols].merge(counts, on="province", how="left", validate="one_to_one")
    panel["folklore_count"] = panel["folklore_count"].fillna(0).astype(int)
    panel["pop_million"] = panel["pop_2020"] / 1_000_000
    panel["folklore_per_million"] = panel["folklore_count"] / panel["pop_million"]
    panel["source_ids"] = panel["source_ids"].fillna("")
    if len(panel) != 31 or int(panel["folklore_count"].sum()) != 471:
        raise ValueError("Rebuilt analysis panel does not have 31 provinces / 471 records")
    OUT.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT / "provincial_analysis_panel_2020.csv", index=False, encoding="utf-8-sig")
    flow = pd.DataFrame([
      {"stage":"Official folklore records", "n":492, "unit":"unique source records", "rule":"Official category = folklore"},
      {"stage":"Unassigned to a regional/provincial location", "n":11, "unit":"source records", "rule":"National/central unit; no assignable province"},
      {"stage":"Regionally assigned records", "n":481, "unit":"source records", "rule":"Includes mainland, Hong Kong, and Macao"},
      {"stage":"Mainland records in population/spatial analysis", "n":471, "unit":"source records", "rule":"31 mainland provincial units; excludes Hong Kong and Macao"},
      {"stage":"Population/spatial analysis units", "n":31, "unit":"provinces", "rule":"One observation per mainland province"},
    ])
    flow.to_csv(OUT / "sample_flow_492_481_471.csv", index=False, encoding="utf-8-sig")
    ref = pd.read_csv(REFERENCE, encoding="utf-8-sig")
    shared = [c for c in panel.columns if c in ref.columns and c != "source_ids"]
    a = panel[shared].sort_values("province").reset_index(drop=True)
    b = ref[shared].sort_values("province").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
    left = panel[["province", "source_ids"]].sort_values("province").reset_index(drop=True)
    right = ref[["province", "source_ids"]].sort_values("province").reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_dtype=False)
    print("Verified source snapshot 3,610 unique -> 492 folklore IDs -> 481 mapped -> 471 mainland; rebuilt panel matches the staged panel and source-ID assignments.")

if __name__ == "__main__": main()
