"""Export a source-linked, record-level inclusion and geographic matching table."""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "intermediate" / "record_inclusion_geocoding_audit.csv"
OUTPUT = ROOT / "results" / "tables" / "Supplementary_Table_S3_Record_Level_Inclusion_Geographic_Matching.csv"
README = ROOT / "results" / "tables" / "Supplementary_Table_S3_README.md"

PROVINCES = {
    "北京市": "Beijing", "天津市": "Tianjin", "河北省": "Hebei", "山西省": "Shanxi",
    "内蒙古自治区": "Inner Mongolia", "辽宁省": "Liaoning", "吉林省": "Jilin",
    "黑龙江省": "Heilongjiang", "上海市": "Shanghai", "江苏省": "Jiangsu",
    "浙江省": "Zhejiang", "安徽省": "Anhui", "福建省": "Fujian", "江西省": "Jiangxi",
    "山东省": "Shandong", "河南省": "Henan", "湖北省": "Hubei", "湖南省": "Hunan",
    "广东省": "Guangdong", "广西壮族自治区": "Guangxi", "海南省": "Hainan",
    "重庆市": "Chongqing", "四川省": "Sichuan", "贵州省": "Guizhou", "云南省": "Yunnan",
    "西藏自治区": "Tibet", "陕西省": "Shaanxi", "甘肃省": "Gansu", "青海省": "Qinghai",
    "宁夏回族自治区": "Ningxia", "新疆维吾尔自治区": "Xinjiang", "香港": "Hong Kong",
    "澳门": "Macao", "中直单位": "National/central-level unit",
}
LEVELS = {
    "区县中心点": "County/district centroid", "市级中心点": "Prefecture-level city centroid",
    "省级中心点": "Provincial centroid", "特别行政区中心点": "Special administrative region centroid",
    "未匹配": "Unmatched",
}
BATCHES = {f"第{x}批": f"Batch {x}" for x in ("一", "二", "三", "四", "五")}


def main() -> None:
    data = pd.read_csv(SOURCE, encoding="utf-8-sig", dtype={"source_id": str, "item_code": str})
    regional = data["include_regional_map"].astype(int)
    analysis = data["include_population_spatial_analysis"].astype(int)
    assert len(data) == 492 and data["source_id"].nunique() == 492
    assert int(regional.sum()) == 481 and int(analysis.sum()) == 471

    def rationale(row: pd.Series) -> str:
        if int(row["include_regional_map"]) == 0:
            return "No province-level location assigned; retained in the national folklore-record count, excluded from regional mapping and provincial analyses"
        if int(row["include_population_spatial_analysis"]) == 0:
            return "Included in regional mapping/count; outside the 31 mainland provincial-level units used for population-standardized and spatial analyses"
        return "Assigned to one of the 31 mainland provincial-level units and included in the provincial analyses"

    output = pd.DataFrame({
        "official_record_id": data["source_id"].astype(str),
        "official_project_code": data["item_code"].fillna("").astype(str),
        "official_batch": data["batch"].map(BATCHES).fillna(data["batch"]),
        "official_category": "Folklore",
        "reported_region_or_nominating_unit_original": data["region_raw"].fillna(""),
        "assigned_provincial_level_unit": data["province_match"].map(PROVINCES).fillna(data["province_match"]),
        "geographic_match_level": data["geographic_match_level"].map(LEVELS).fillna(data["geographic_match_level"]),
        "included_in_regional_map_and_count": regional,
        "included_in_31_unit_mainland_analysis": analysis,
        "inclusion_decision": data.apply(rationale, axis=1),
        "official_source_url": "https://www.ihchina.cn/project_details/" + data["source_id"].astype(str) + ".html",
    }).sort_values("official_record_id", key=lambda col: col.astype(int), kind="stable")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    README.write_text(
        "# Supplementary Table S3. Record-level inclusion and geographic matching\n\n"
        "This table documents all 492 records classified by the official source as folklore and their treatment in regional mapping/counts (481 records) and the 31-unit mainland analysis (471 records). It includes identifiers, reported region/nominating unit, assigned provincial-level unit, match level, inclusion decisions, and a direct source link. Project names and descriptive text are omitted.\n\n"
        "Source: China Intangible Cultural Heritage website. The source website requires attribution to the relevant author and the website when content is reproduced. Its copyright notice restricts commercial reuse without prior written authorization and prohibits unauthorized modification of website content. See https://www.ihchina.cn/about/206.html. Records in the underlying source table were downloaded on 2026-07-28.\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT} ({len(output)} rows; regional={int(regional.sum())}; analysis={int(analysis.sum())})")


if __name__ == "__main__":
    main()
