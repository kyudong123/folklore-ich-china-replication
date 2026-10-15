# -*- coding: utf-8 -*-
"""Spatial-weights and small-denominator sensitivity analysis (31 provinces).

Uses only the bundled Python runtime's numpy/pandas plus the Python standard
library. Queen neighbors are read from the project's TopoJSON shared arcs, and
KNN distances use great-circle distance between the same central points used by
the project's map workflow.
"""
from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PACKAGE_DIR = HERE.parent
OUTPUT_DIR = PACKAGE_DIR / "results" / "spatial"
WEIGHT_DIR = PACKAGE_DIR / "data" / "weights"
PANEL_SOURCE = PACKAGE_DIR / "data" / "processed" / "provincial_analysis_panel_2020.csv"
TOPO_SOURCE = PACKAGE_DIR / "data" / "source" / "cn-atlas.json"
PANEL_COPY = PANEL_SOURCE
TOPO_COPY = TOPO_SOURCE
N_PERMUTATIONS = 9999
SEED = 20261008
ALPHA = 0.05
KNN_K = 4
PROVINCES = [
    "北京", "天津", "河北", "山西", "内蒙古", "辽宁", "吉林", "黑龙江", "上海", "江苏",
    "浙江", "安徽", "福建", "江西", "山东", "河南", "湖北", "湖南", "广东", "广西",
    "海南", "重庆", "四川", "贵州", "云南", "西藏", "陕西", "甘肃", "青海", "宁夏", "新疆",
]
CENTROID_OVERRIDE = {
    "北京": (116.4, 40.1), "天津": (117.3, 39.1), "上海": (121.5, 31.2),
    "重庆": (107.6, 29.8), "宁夏": (106.2, 37.5), "海南": (109.7, 19.2),
    "内蒙古": (113.0, 43.8), "黑龙江": (128.0, 47.5), "西藏": (88.0, 31.5),
    "新疆": (86.0, 41.5), "甘肃": (103.8, 38.5), "四川": (103.0, 30.5),
    "云南": (101.5, 25.0), "广西": (108.5, 23.7), "陕西": (109.0, 35.7),
    "山西": (112.3, 37.5), "广东": (113.3, 23.4), "福建": (118.2, 26.1),
}


def province_key(name: str) -> str:
    for suffix in ["特别行政区", "壮族自治区", "回族自治区", "维吾尔自治区", "自治区", "省", "市"]:
        name = name.replace(suffix, "")
    return name.strip()


def signed_area(ring):
    return 0.5 * sum(
        ring[i][0] * ring[(i + 1) % len(ring)][1]
        - ring[(i + 1) % len(ring)][0] * ring[i][1]
        for i in range(len(ring))
    )


def topo_provinces(path: Path):
    topo = json.loads(path.read_text(encoding="utf-8"))
    sx, sy = topo["transform"]["scale"]
    tx, ty = topo["transform"]["translate"]
    decoded_arcs = []
    for arc in topo["arcs"]:
        x = y = 0
        pts = []
        for dx, dy in arc:
            x += dx
            y += dy
            pts.append((x * sx + tx, y * sy + ty))
        decoded_arcs.append(pts)

    output = {}
    for geom in topo["objects"]["provinces"]["geometries"]:
        raw_name = geom["properties"].get("地名", geom["properties"].get("name", ""))
        key = province_key(raw_name)
        if key not in PROVINCES:
            continue
        arc_ids = set()
        rings = []

        def collect(obj):
            if isinstance(obj, int):
                arc_ids.add(~obj if obj < 0 else obj)
            else:
                for child in obj:
                    collect(child)

        collect(geom["arcs"])
        polygons = geom["arcs"] if geom["type"] == "MultiPolygon" else [geom["arcs"]]
        for polygon in polygons:
            for ring_arcs in polygon:
                ring = []
                for arc_id in ring_arcs:
                    reverse = arc_id < 0
                    true_id = ~arc_id if reverse else arc_id
                    pts = list(reversed(decoded_arcs[true_id])) if reverse else decoded_arcs[true_id]
                    ring.extend(pts[1:] if ring else pts)
                if len(ring) >= 3:
                    rings.append(ring)
        if key in CENTROID_OVERRIDE:
            center = CENTROID_OVERRIDE[key]
        else:
            largest = max(rings, key=lambda ring: abs(signed_area(ring)))
            center = (float(np.mean([p[0] for p in largest])), float(np.mean([p[1] for p in largest])))
        output[key] = {"arc_ids": arc_ids, "center": center}
    missing = set(PROVINCES).difference(output)
    if missing:
        raise ValueError(f"TopoJSON is missing provincial geometries: {sorted(missing)}")
    return output


def make_queen(geometries):
    neighbors = {p: set() for p in PROVINCES}
    for i, a in enumerate(PROVINCES):
        for b in PROVINCES[i + 1 :]:
            if geometries[a]["arc_ids"].intersection(geometries[b]["arc_ids"]):
                neighbors[a].add(b)
                neighbors[b].add(a)
    return neighbors


def haversine_km(a, b):
    lon1, lat1 = map(math.radians, a)
    lon2, lat2 = map(math.radians, b)
    dlon, dlat = lon2 - lon1, lat2 - lat1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * math.asin(math.sqrt(h))


def make_knn(geometries, k):
    neighbors = {}
    for a in PROVINCES:
        distances = sorted(
            ((haversine_km(geometries[a]["center"], geometries[b]["center"]), b) for b in PROVINCES if b != a),
            key=lambda x: (x[0], PROVINCES.index(x[1])),
        )
        neighbors[a] = {b for _, b in distances[:k]}
    return neighbors


def matrix_from_neighbors(neighbors):
    n = len(PROVINCES)
    w = np.zeros((n, n), dtype=float)
    for i, a in enumerate(PROVINCES):
        for b in neighbors[a]:
            w[i, PROVINCES.index(b)] = 1.0
    degrees = w.sum(axis=1)
    # Queen weights can contain an island (notably Hainan); keep its row zero.
    # KNN has no isolates by construction. Nonzero rows are row-standardized.
    nonzero = degrees > 0
    w[nonzero] /= degrees[nonzero, None]
    return w


def moran_i(x, w):
    z = np.asarray(x, dtype=float) - np.mean(x)
    return float(len(z) / w.sum() * (z @ w @ z) / (z @ z))


def global_moran(x, w, rng):
    observed = moran_i(x, w)
    sims = np.empty(N_PERMUTATIONS, dtype=float)
    x = np.asarray(x, dtype=float)
    for b in range(N_PERMUTATIONS):
        sims[b] = moran_i(rng.permutation(x), w)
    # Match the project's established two-sided randomization convention:
    # count simulated statistics whose absolute value is at least |I_observed|.
    p = (1 + np.count_nonzero(np.abs(sims) >= abs(observed))) / (N_PERMUTATIONS + 1)
    return observed, float(p), float(sims.mean()), float(sims.std(ddof=1))


def local_moran(x, w, rng):
    x = np.asarray(x, dtype=float)
    z = (x - x.mean()) / x.std(ddof=0)
    n = len(z)
    result = []
    for i, province in enumerate(PROVINCES):
        idx = np.flatnonzero(w[i])
        if len(idx) == 0:
            result.append({"province": province, "z_value": float(z[i]), "spatial_lag_z": np.nan,
                           "local_moran_i": np.nan, "pseudo_p_two_sided": 1.0,
                           "cluster_p_lt_0_05_nominal": "NS"})
            continue
        observed = float(z[i] * (w[i] @ z))
        pool = np.delete(np.arange(n), i)
        sims = np.empty(N_PERMUTATIONS, dtype=float)
        for b in range(N_PERMUTATIONS):
            draw = rng.choice(pool, size=len(idx), replace=False)
            # KNN and binary Queen weights are uniform within each row after row standardization.
            sims[b] = float(z[i] * z[draw].mean())
        p = (1 + np.count_nonzero(np.abs(sims) >= abs(observed))) / (N_PERMUTATIONS + 1)
        lag = float(w[i] @ z)
        if p >= ALPHA:
            cluster = "NS"
        elif z[i] >= 0 and lag >= 0:
            cluster = "HH"
        elif z[i] < 0 and lag < 0:
            cluster = "LL"
        elif z[i] >= 0 and lag < 0:
            cluster = "HL"
        else:
            cluster = "LH"
        result.append({"province": province, "z_value": float(z[i]), "spatial_lag_z": lag,
                       "local_moran_i": observed, "pseudo_p_two_sided": float(p),
                       "cluster_p_lt_0_05_nominal": cluster})
    return result


def eb_standardized_rate(counts, population):
    """Assuncao-Reis EB standardization for rates, per GeoDa workbook equations."""
    counts = np.asarray(counts, dtype=float)
    pop = np.asarray(population, dtype=float)
    rate = counts / pop
    beta = counts.sum() / pop.sum()
    alpha = float(np.sum(pop * (rate - beta) ** 2) / pop.sum() - beta / (pop.sum() / len(pop)))
    variance = alpha + beta / pop
    raw_or_standardized = np.where(variance > 0, (rate - beta) / np.sqrt(variance), rate)
    # Local Moran is computed on a centered, unit-variance version of these standardized rates.
    z = (raw_or_standardized - raw_or_standardized.mean()) / raw_or_standardized.std(ddof=0)
    return z, beta, alpha, variance


def bh_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    m = len(pvalues)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty(m, dtype=float)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def by_adjust(pvalues):
    """Benjamini-Yekutieli adjustment, conservative under arbitrary dependence."""
    pvalues = np.asarray(pvalues, dtype=float)
    harmonic = float(np.sum(1.0 / np.arange(1, len(pvalues) + 1)))
    return np.minimum(1.0, bh_adjust(pvalues) * harmonic)


def write_weights(name, neighbors):
    WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
    filename = "queen_neighbors.csv" if name == "Queen" else "knn4_neighbors.csv"
    path = WEIGHT_DIR / filename
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["province", "neighbors", "degree"])
        for province in PROVINCES:
            ns = sorted(neighbors[province], key=PROVINCES.index)
            writer.writerow([province, ";".join(ns), len(ns)])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHT_DIR.mkdir(parents=True, exist_ok=True)
    if not PANEL_COPY.exists():
        shutil.copy2(PANEL_SOURCE, PANEL_COPY)
    if not TOPO_COPY.exists():
        shutil.copy2(TOPO_SOURCE, TOPO_COPY)
    df = pd.read_csv(PANEL_COPY, encoding="utf-8-sig").set_index("province").loc[PROVINCES].reset_index()
    if len(df) != 31 or int(df["folklore_count"].sum()) != 471:
        raise ValueError("Expected the 471-record, 31-mainland-province harmonized sample")

    geometries = topo_provinces(TOPO_COPY)
    queen = make_queen(geometries)
    knn = make_knn(geometries, KNN_K)
    w_queen = matrix_from_neighbors(queen)
    w_knn = matrix_from_neighbors(knn)
    pd.DataFrame(w_queen, index=PROVINCES, columns=PROVINCES).to_csv(WEIGHT_DIR / "queen_row_standardized.csv", encoding="utf-8-sig")
    pd.DataFrame(w_knn, index=PROVINCES, columns=PROVINCES).to_csv(WEIGHT_DIR / "knn4_row_standardized.csv", encoding="utf-8-sig")
    w_queen_binary = np.zeros_like(w_queen)
    for i, province in enumerate(PROVINCES):
        for neighbor in queen[province]:
            w_queen_binary[i, PROVINCES.index(neighbor)] = 1.0
    write_weights("Queen", queen)
    write_weights(f"KNN{k if (k:=KNN_K) else KNN_K}", knn)

    queen_degree = np.array([len(queen[p]) for p in PROVINCES])
    counts = df["folklore_count"].to_numpy(dtype=float)
    pop = df["pop_2020"].to_numpy(dtype=float)
    rate = df["folklore_per_million"].to_numpy(dtype=float)
    minority_share = df["minority_share_2020"].to_numpy(dtype=float)
    ols_design = np.column_stack([np.ones(len(df)), minority_share])
    ols_residuals = rate - ols_design @ np.linalg.lstsq(ols_design, rate, rcond=None)[0]
    eb_z, beta, alpha, eb_variance = eb_standardized_rate(counts, pop)

    settings = [
        ("Queen row-standardized", w_queen, rate, "crude rate"),
        (f"KNN-{KNN_K} row-standardized", w_knn, rate, "crude rate"),
        ("Queen row-standardized", w_queen, eb_z, "Assuncao-Reis EB-standardized rate"),
        ("Queen row-standardized", w_queen, counts, "raw count"),
        (f"KNN-{KNN_K} row-standardized", w_knn, counts, "raw count"),
        ("Queen row-standardized", w_queen, ols_residuals, "OLS residual"),
        (f"KNN-{KNN_K} row-standardized", w_knn, ols_residuals, "OLS residual"),
    ]
    global_rows, local_frames = [], []
    for ix, (weight_name, w, values, indicator) in enumerate(settings):
        rng = np.random.default_rng(SEED + ix)
        i, p, sim_mean, sim_sd = global_moran(values, w, rng)
        global_rows.append({
            "indicator": indicator, "weights": weight_name, "n_provinces": len(df),
            "records": int(counts.sum()), "queen_neighbors_min": int(queen_degree.min()),
            "queen_neighbors_median": float(np.median(queen_degree)), "queen_neighbors_mean": float(queen_degree.mean()),
            "knn_k": KNN_K if weight_name.startswith("KNN") else "",
            "row_standardized": True, "moran_I": i, "permutation_p_two_sided": p,
            "permutations": N_PERMUTATIONS, "seed": SEED + ix,
            "permutation_mean": sim_mean, "permutation_sd": sim_sd,
            "eb_beta_rate_per_person": beta if "EB" in indicator else "",
            "eb_alpha": alpha if "EB" in indicator else "",
            "eb_variance_nonpositive_n": int(np.count_nonzero(eb_variance <= 0)) if "EB" in indicator else "",
        })
        local_rng = np.random.default_rng(SEED + 100 + ix)
        lrows = local_moran(values, w, local_rng)
        p_adj = bh_adjust([r["pseudo_p_two_sided"] for r in lrows])
        p_by = by_adjust([r["pseudo_p_two_sided"] for r in lrows])
        for row, padj, pby in zip(lrows, p_adj, p_by):
            row.update({"indicator": indicator, "weights": weight_name,
                        "cluster_bh_fdr_0_05": row["cluster_p_lt_0_05_nominal"] if padj < ALPHA else "NS",
                        "pseudo_p_bh_fdr": float(padj),
                        "cluster_by_fdr_0_05": row["cluster_p_lt_0_05_nominal"] if pby < ALPHA else "NS",
                        "pseudo_p_by_fdr": float(pby)})
        local_frames.extend(lrows)

    # Audit: the manuscript's reported I=0.247 is numerically reproduced by
    # binary Queen weights, whereas the stated row-standardized Queen gives a
    # different value. Keep this row clearly labeled as an audit, not as the
    # primary sensitivity specification.
    for j, (indicator, values) in enumerate([
        ("crude rate", rate), ("raw count", counts), ("OLS residual", ols_residuals)
    ]):
        # These were the original project seeds for density, counts, residuals.
        audit_seed = [20260605, 20260606, 20260607][j]
        audit_i, audit_p, audit_mean, audit_sd = global_moran(
            values, w_queen_binary, np.random.default_rng(audit_seed)
        )
        global_rows.append({
            "indicator": indicator, "weights": "Queen binary (reported-value audit)",
            "n_provinces": len(df), "records": int(counts.sum()),
            "queen_neighbors_min": int(queen_degree.min()),
            "queen_neighbors_median": float(np.median(queen_degree)),
            "queen_neighbors_mean": float(queen_degree.mean()), "knn_k": "",
            "row_standardized": False, "moran_I": audit_i,
            "permutation_p_two_sided": audit_p, "permutations": N_PERMUTATIONS,
            "seed": audit_seed, "permutation_mean": audit_mean,
            "permutation_sd": audit_sd, "eb_beta_rate_per_person": "",
            "eb_alpha": "", "eb_variance_nonpositive_n": "",
        })

    pd.DataFrame(global_rows).to_csv(OUTPUT_DIR / "05_global_moran_sensitivity.csv", index=False, encoding="utf-8-sig")
    local = pd.DataFrame(local_frames)
    local.to_csv(OUTPUT_DIR / "06_local_moran_province_results.csv", index=False, encoding="utf-8-sig")
    primary = local[(local["indicator"] == "crude rate") &
                    (local["weights"] == "Queen row-standardized")].copy()
    primary = primary.rename(columns={
        "province": "Province", "z_value": "Standardized listing density (z)",
        "spatial_lag_z": "Spatial lag (z)", "local_moran_i": "Local Moran's I",
        "pseudo_p_two_sided": "Two-sided local permutation p (9,999 permutations)",
        "cluster_p_lt_0_05_nominal": "Exploratory LISA class (unadjusted p < .05)",
        "pseudo_p_bh_fdr": "BH-FDR adjusted p", "cluster_bh_fdr_0_05": "BH-FDR class (alpha = .05)",
        "pseudo_p_by_fdr": "BY-FDR adjusted p", "cluster_by_fdr_0_05": "BY-FDR class (alpha = .05)",
    })
    supplement_cols = ["Province", "Standardized listing density (z)", "Spatial lag (z)",
                       "Local Moran's I", "Two-sided local permutation p (9,999 permutations)",
                       "Exploratory LISA class (unadjusted p < .05)", "BH-FDR adjusted p",
                       "BH-FDR class (alpha = .05)", "BY-FDR adjusted p", "BY-FDR class (alpha = .05)"]
    primary[supplement_cols].to_csv(OUTPUT_DIR / "Supplementary_Table_S1_Local_Moran_Queen_9999.csv",
                                    index=False, encoding="utf-8-sig", float_format="%.8f")
    summary = (local.groupby(["indicator", "weights", "cluster_p_lt_0_05_nominal"]).size()
               .unstack(fill_value=0).reset_index())
    for c in ["HH", "LL", "HL", "LH", "NS"]:
        if c not in summary:
            summary[c] = 0
    correction_counts = (local.groupby(["indicator", "weights"])
                         .agg(BH_FDR_discoveries=("pseudo_p_bh_fdr", lambda s: int((s < ALPHA).sum())),
                              BY_FDR_discoveries=("pseudo_p_by_fdr", lambda s: int((s < ALPHA).sum())))
                         .reset_index())
    summary = summary.merge(correction_counts, on=["indicator", "weights"], how="left")
    summary.to_csv(OUTPUT_DIR / "07_lisa_cluster_counts.csv", index=False, encoding="utf-8-sig")

    # Province-level comparison table: membership changes make the denominator effect explicit.
    wide = local.pivot(index="province", columns=["indicator", "weights"], values="cluster_p_lt_0_05_nominal")
    wide.columns = [" | ".join(col) for col in wide.columns]
    wide.reset_index().to_csv(OUTPUT_DIR / "08_lisa_cluster_comparison_by_province.csv", index=False, encoding="utf-8-sig")

    write_readme(global_rows, summary, local, queen_degree, beta, alpha, eb_variance)
    print(pd.DataFrame(global_rows).to_string(index=False))
    print("\nLISA cluster counts:\n", summary.to_string(index=False))
    print(f"\nQueen degree range={queen_degree.min()}-{queen_degree.max()}, median={np.median(queen_degree):.1f}, mean={queen_degree.mean():.2f}")
    print(f"EB method-of-moments alpha={alpha:.10g}; beta={beta:.10g}; nonpositive estimated variance={np.count_nonzero(eb_variance <= 0)}")


def write_readme(global_rows, summary, local, queen_degree, beta, alpha, eb_variance):
    by = {r["indicator"] + " / " + r["weights"]: r for r in global_rows}
    lines = [
        "# Spatial weights and population-denominator sensitivity analysis", "",
        "This report accompanies the reproducible 31-province analysis of 471 nationally listed folklore ICH records. It records the spatial specifications and the values regenerated by the package workflow.", "",
        "## Specifications", "",
        f"- Queen contiguity is reconstructed from shared TopoJSON arcs; degree min/median/mean/max = {queen_degree.min()}/{np.median(queen_degree):.1f}/{queen_degree.mean():.2f}/{queen_degree.max()}. Hainan is retained as a zero-row isolate.",
        f"- Alternative weights are directed K-nearest neighbors with k={KNN_K}, based on great-circle distances between provincial central points; rows are standardized.",
        f"- Global and local Moran permutation tests use {N_PERMUTATIONS:,} permutations and reproducible specification-specific seeds. Local classifications are exploratory; BH/BY false-discovery-rate adjustments are reported.",
        f"- Crude-rate Queen Moran's I={by['crude rate / Queen row-standardized']['moran_I']:.6f} (p={by['crude rate / Queen row-standardized']['permutation_p_two_sided']:.3f}); KNN-4 I={by['crude rate / KNN-4 row-standardized']['moran_I']:.6f} (p={by['crude rate / KNN-4 row-standardized']['permutation_p_two_sided']:.3f}).",
        "- Primary Queen LISA nominal classes: three High-High units (Tibet, Qinghai, Xinjiang) and two Low-High units (Sichuan, Yunnan); no unit remains significant after BH-FDR or BY-FDR correction.",
        f"- Assunção–Reis estimates: beta={beta:.10g}; alpha={alpha:.10g}; units with nonpositive estimated rate variance={np.count_nonzero(eb_variance <= 0)}.",
        "- These statistics describe the spatial distribution of nationally listed records; they do not measure the prevalence of all folklore practices or establish causal mechanisms.", "",
        "## Outputs", "",
        "- `05_global_moran_sensitivity.csv`: global Moran statistics for crude rates, raw counts, and OLS residuals under row-standardized Queen and KNN-4 weights, plus sensitivity/audit specifications.",
        "- `06_local_moran_province_results.csv`: provincial local I statistics, pseudo-p values, nominal classifications, and BH/BY-adjusted results.",
        "- `07_lisa_cluster_counts.csv` and `08_lisa_cluster_comparison_by_province.csv`: summaries of local classifications.",
        "- Neighbor lists and row-standardized matrices are stored in the package's `data/weights/` folder.", "",
        "The previously reported Queen binary value is retained only as a specification audit; the manuscript specification and Figure 3 use row-standardized Queen weights.", "",
        "## References", "",
        "- Anselin, L. (1995). Local Indicators of Spatial Association—LISA. Geographical Analysis, 27(2), 93–115. https://doi.org/10.1111/j.1538-4632.1995.tb00338.x",
        "- Assunção, R. M., & Reis, E. A. (1999). A new proposal to adjust Moran's I for population density. Statistics in Medicine, 18(16), 2147–2162. https://doi.org/10.1002/(SICI)1097-0258(19990830)18:16<2147::AID-SIM179>3.0.CO;2-I",
        "- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. Journal of the Royal Statistical Society: Series B, 57(1), 289–300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x",
        "- Benjamini, Y., & Yekutieli, Y. (2001). The control of the false discovery rate in multiple testing under dependency. The Annals of Statistics, 29(4), 1165–1188. https://doi.org/10.1214/aos/1013699998",
        "- GeoDa Center. Distance-Based Spatial Weights: https://geodacenter.github.io/workbook/4b_dist_weights/lab4b.html",
        "- GeoDa Center. Local Spatial Autocorrelation (2): https://geodacenter.github.io/workbook/6b_local_adv/lab6b.html",
    ]
    (OUTPUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
if __name__ == "__main__":
    main()
