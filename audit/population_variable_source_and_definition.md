# 2020 Census demographic data verification

The provincial demographic inputs were checked against Table 1-4 of the *China Population Census Yearbook 2020* and the National Bureau of Statistics communiqué on the Seventh National Population Census.

## Official sources

- Provincial resident population: [Communiqué of the Seventh National Population Census (No. 3)](https://www.stats.gov.cn/xxgk/sjfb/zxfb2020/202105/t20210511_1817198.html).
- Ethnic population by province: [China Population Census Yearbook 2020, Table 1-4](https://www.stats.gov.cn/sj/pcsj/rkpc/7rp/lefte.htm); [official workbook](https://www.stats.gov.cn/sj/pcsj/rkpc/7rp/zk/html/A0104.xls).

## Definitions and checks

The minority-population measure includes the 55 officially identified minority groups and residents whose ethnicity was unclassified in Table 1-4. Naturalized residents are retained as a separate category and are not included in either the Han or minority counts. For the 31 mainland provincial-level units, Han population (1,284,446,389) plus minority population, including unclassified ethnicity (125,315,740), plus naturalized population (16,595) equals resident population (1,409,778,724).

The derived input is `data/source/province_demographics_2020_derived_from_census.csv`; the analytical values are included in `data/processed/provincial_analysis_panel_2020.csv`. Province-level checks are recorded in `audit/province_ethnic_counts_official_crosscheck.csv` and `audit/population_scope_arithmetic_checks.csv`.
