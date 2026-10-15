# Data source register

This register identifies the source files and provenance used in the provincial folklore ICH analysis.

| File | Source and processing | Verification |
|---|---|---|
| `national_ich_all_categories_source_snapshot.json` | China Intangible Cultural Heritage website; source and API URLs are recorded in the snapshot; downloaded 2026-07-28; five batches, 3,610 records | Unique source IDs and the official folklore-category filter were checked against the selected-record file and item-level audit |
| `folklore_records_with_coordinates.csv` | National folklore records joined to administrative matching and coordinate fields; each record retains an official project-detail URL | 492 unique official IDs; IDs match the source snapshot folklore records |
| `province_population_2010.csv` | Provincial resident population from the National Bureau of Statistics Sixth National Population Census communiqué (2011); source page archived as `census2010_bulletin_no2.html` | Provincial total checked against the published communiqué |
| `census2010_bulletin_no2.html` | National Bureau of Statistics of China, Sixth National Population Census communiqué (2011); source URL is recorded in the manuscript references | Source page snapshot documents the 2010 population input |
| `province_demographics_2020_derived_from_census.csv` | Provincial demographic variables derived from Table 1-4 of the *China Population Census Yearbook 2020* | Checked against the yearbook-derived provincial cross-check table |
| `cn-atlas.json` | Provincial-boundary TopoJSON attributed to the National Platform for Common Geospatial Information Services. The platform catalog describes the **1:1,000,000 Public Edition Fundamental Geographic Information Data (2021)**, with data currency to 2019 and a boundary-and-administrative-region layer ([product description](https://www.webmap.cn/commres.do?method=result25W); [data-download page](https://www.webmap.cn/commres.do?method=dataDownload)). The download page states that downloads have moved to the Tianditu data-source portal. | The local TopoJSON contains `prefectures`, `provinces`, and `nation` objects and is used for spatial analysis and provincial maps. The local file does not retain the original downloaded file, exact product/version identifier, download date, or conversion log; the 2021 catalog edition is therefore documented as the relevant platform product, not as independently verified provenance for this exact derivative. |
| `geocoding_reference_county_above.csv` | Supporting administrative-location reference table used in county-first matching | Used alongside reported project locations and the documented match decisions |

The record-level inclusion and geographic-matching decisions are documented in `data/intermediate/record_inclusion_geocoding_audit.csv`. Variable definitions and units are listed in `data/intermediate/data_dictionary.csv`.
