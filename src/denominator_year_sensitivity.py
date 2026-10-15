"""Rebuild Supplementary Table S2 from the staged 2010 and 2020 census inputs."""
from pathlib import Path
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
panel = pd.read_csv(ROOT / "data/processed/provincial_analysis_panel_2020.csv", encoding="utf-8-sig")
pop2010 = pd.read_csv(ROOT / "data/source/province_population_2010.csv", encoding="utf-8-sig")
merged = panel.merge(pop2010, on="province", validate="one_to_one")
if len(merged) != 31 or int(merged.folklore_count.sum()) != 471:
    raise ValueError("Expected 31 mainland provinces and 471 folklore records")
if int(pop2010.pop_2010.sum()) != 1_332_774_867:
    raise ValueError("2010 province total differs from official census bulletin")
merged["density_per_million_2010"] = merged.folklore_count / merged.pop_2010 * 1_000_000
merged["density_per_million_2020"] = merged.folklore_count / merged.pop_2020 * 1_000_000
if not (merged.density_per_million_2020.sub(merged.folklore_per_million).abs() < 1e-9).all():
    raise ValueError("2020 rates disagree with the main analysis panel")
merged["rank_2010"] = merged.density_per_million_2010.rank(ascending=False, method="min").astype(int)
merged["rank_2020"] = merged.density_per_million_2020.rank(ascending=False, method="min").astype(int)
rho = spearmanr(merged.density_per_million_2010, merged.density_per_million_2020).statistic
if abs(rho - 0.9903225806451613) > 1e-12:
    raise ValueError(f"Unexpected rank correlation: {rho}")
columns = ["province", "folklore_count", "pop_2010", "pop_2020", "density_per_million_2010",
           "density_per_million_2020", "rank_2010", "rank_2020"]
target = ROOT / "results/tables/Supplementary_Table_S2_Denominator_2010_2020.csv"
merged[columns].sort_values("rank_2020").to_csv(target, index=False, encoding="utf-8-sig")
print(f"Denominator-year sensitivity: n=31, Spearman rho={rho:.6f}; wrote {target.relative_to(ROOT)}")
