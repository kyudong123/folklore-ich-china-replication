"""Check the staged panel, spatial weights, reported statistics, and figure derivatives."""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
checks = []
def ok(name, condition, detail):
    if not condition:
        raise AssertionError(f"FAIL: {name}: {detail}")
    checks.append(f"PASS — {name}: {detail}")

panel = pd.read_csv(ROOT / "data/processed/provincial_analysis_panel_2020.csv", encoding="utf-8-sig")
ok("analysis panel", len(panel) == 31 and panel.province.nunique() == 31 and int(panel.folklore_count.sum()) == 471,
   f"31 unique units; total records={int(panel.folklore_count.sum())}")
queen = pd.read_csv(ROOT / "data/weights/queen_row_standardized.csv", index_col=0, encoding="utf-8-sig")
knn = pd.read_csv(ROOT / "data/weights/knn4_row_standardized.csv", index_col=0, encoding="utf-8-sig")
adj = pd.read_csv(ROOT / "data/weights/queen_neighbors.csv", encoding="utf-8-sig")
neighbor_map = {r.province: set(str(r.neighbors).split(";")) if pd.notna(r.neighbors) and str(r.neighbors) else set() for r in adj.itertuples()}
symmetric = all(p in neighbor_map.get(q, set()) for p, ns in neighbor_map.items() for q in ns)
ok("Queen matrix", queen.shape == (31,31) and symmetric and np.allclose(queen.sum(axis=1).drop("海南"), 1) and queen.loc["海南"].sum() == 0,
   "31×31; underlying Queen adjacency symmetric; non-island rows sum to 1; Hainan isolate row sums to 0")
ok("KNN-4 matrix", knn.shape == (31,31) and np.allclose(knn.sum(axis=1), 1) and np.allclose((knn > 0).sum(axis=1), 4),
   "31×31; four outgoing neighbors and row sum 1 per province")
moran = pd.read_csv(ROOT / "results/spatial/05_global_moran_sensitivity.csv", encoding="utf-8-sig")
q = moran[(moran.indicator == "crude rate") & (moran.weights == "Queen row-standardized")].iloc[0]
k = moran[(moran.indicator == "crude rate") & (moran.weights == "KNN-4 row-standardized")].iloc[0]
ok("Moran results", abs(q.moran_I - .2872000533) < 1e-8 and q.permutation_p_two_sided == .0103 and abs(k.moran_I - .2660039003) < 1e-8 and k.permutation_p_two_sided == .0056 and set(moran.permutations.astype(int)) == {9999},
   f"9,999 permutations; Queen I={q.moran_I:.6f}, p={q.permutation_p_two_sided:.4f}; KNN-4 I={k.moran_I:.6f}, p={k.permutation_p_two_sided:.4f}")
local = pd.read_csv(ROOT / "results/spatial/06_local_moran_province_results.csv", encoding="utf-8-sig")
primary = local[(local.indicator == "crude rate") & (local.weights == "Queen row-standardized")]
ok("LISA and FDR", len(primary) == 31 and int((primary.cluster_bh_fdr_0_05 != "NS").sum()) == 0 and int((primary.cluster_by_fdr_0_05 != "NS").sum()) == 0,
   f"primary Queen: {int((primary.cluster_p_lt_0_05_nominal != 'NS').sum())} nominal units; no BH/BY discoveries")
supp = pd.read_csv(ROOT / "results/spatial/Supplementary_Table_S1_Local_Moran_Queen_9999.csv", encoding="utf-8-sig")
ok("Supplementary Table S1", len(supp) == 31 and "BH-FDR adjusted p" in supp and "BY-FDR adjusted p" in supp,
   "31 provincial rows with unadjusted, BH-FDR, and BY-FDR p values")
s2 = pd.read_csv(ROOT / "results/tables/Supplementary_Table_S2_Denominator_2010_2020.csv", encoding="utf-8-sig")
rho = s2[["density_per_million_2010", "density_per_million_2020"]].corr(method="spearman").iloc[0, 1]
ok("Supplementary Table S2", len(s2) == 31 and int(s2.folklore_count.sum()) == 471 and abs(rho - .9903225806451613) < 1e-12,
   f"31 provinces, 471 records; 2010/2020 rate-rank Spearman rho={rho:.6f}")
for n in range(2,11):
    stem = f"fig{n:02d}_"
    names = [x for x in (ROOT / "results/figures").glob(stem + "*.png")]
    ok(f"Figure {n} files", len(names) == 1 and all(names[0].with_suffix(ext).exists() for ext in [".svg", ".jpg", ".tif"]),
       "PNG/SVG/JPEG/TIFF present")
density = panel.folklore_count / panel.pop_2020 * 1_000_000
ok("Figure 4 input", np.allclose(density, panel.folklore_per_million),
   "31 provincial rates equal listed records divided by 2020 resident population")
supp_figure = ROOT / "results/figures/supp_fig_s1_exploratory_lisa_map.png"
ok("Supplementary Figure S1 files", all(supp_figure.with_suffix(ext).exists() for ext in [".png", ".svg", ".jpg", ".tif"]),
   "exploratory LISA PNG/SVG/JPEG/TIFF present")
report = ROOT / "audit/python_verification.txt"
report.write_text("\n".join(checks) + "\n", encoding="utf-8")
print("\n".join(checks))
print(f"\nWrote {report.relative_to(ROOT)}")
