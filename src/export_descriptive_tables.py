from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "processed" / "provincial_analysis_panel_2020.csv"
OUT = ROOT / "results" / "tables"

def main():
    df = pd.read_csv(PANEL, encoding="utf-8-sig")
    if len(df) != 31 or df["province"].nunique() != 31: raise ValueError("Expected 31 provincial rows")
    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["folklore_count", "folklore_per_million", "minority_share_2020", "pop_2020", "minority_pop_2020", "han_pct_2020"]
    df[cols].astype(float).corr(method="pearson").to_csv(OUT / "correlation_matrix_2020.csv", encoding="utf-8-sig")
    desc = df[["pop_2020", "minority_pop_2020", "minority_pct_2020", "minority_share_2020", "han_pop_2020", "han_pct_2020", "folklore_count", "pop_million", "folklore_per_million"]].describe().T
    desc.index.name = "variable"
    desc.to_csv(OUT / "descriptive_statistics_31_provinces.csv", encoding="utf-8-sig")
    print("Exported descriptive statistics and Pearson correlation table for 31 provincial units.")

if __name__ == "__main__": main()
