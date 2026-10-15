"""Rebuild the analysis outputs and figures from the staged source data."""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent
STEPS = [
    ROOT / "src" / "rebuild_analysis_inputs.py",
    ROOT / "src" / "denominator_year_sensitivity.py",
    ROOT / "src" / "export_supplementary_table_s3.py",
    ROOT / "src" / "export_descriptive_tables.py",
    ROOT / "src" / "regression_robustness.py",
    ROOT / "src" / "spatial_sensitivity.py",
    *[ROOT / "src" / "figures" / f"fig{i:02d}_{name}.py" for i, name in [
        (2, "folklore_point_map"), (3, "moran_scatter"), (4, "listing_density_map"),
        (5, "batch_trend"), (6, "indicator_comparison"), (7, "minority_share_map"),
        (8, "scatter_minority_vs_folklore"), (9, "ols_residual_map"), (10, "correlation_heatmap")]],
    ROOT / "src" / "figures" / "supp_fig_s1_exploratory_lisa_map.py",
]
for script in STEPS:
    print(f"\n=== {script.relative_to(ROOT)} ===", flush=True)
    runpy.run_path(str(script), run_name="__main__")
runpy.run_path(str(ROOT / "src" / "build_manifest.py"), run_name="__main__")
runpy.run_path(str(ROOT / "src" / "verify_package.py"), run_name="__main__")
runpy.run_path(str(ROOT / "src" / "build_manifest.py"), run_name="__main__")
print("\nAnalysis run completed. Derived figures, tables, and verification records are available under results/ and audit/.")
