# -*- coding: utf-8 -*-
import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from plot_env import setup_environment
setup_environment()

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.gridspec import GridSpec


PROVINCE_EN = {
    "北京": "Beijing", "天津": "Tianjin", "河北": "Hebei", "山西": "Shanxi",
    "内蒙古": "Inner Mongolia", "辽宁": "Liaoning", "吉林": "Jilin", "黑龙江": "Heilongjiang",
    "上海": "Shanghai", "江苏": "Jiangsu", "浙江": "Zhejiang", "安徽": "Anhui",
    "福建": "Fujian", "江西": "Jiangxi", "山东": "Shandong", "河南": "Henan",
    "湖北": "Hubei", "湖南": "Hunan", "广东": "Guangdong", "广西": "Guangxi",
    "海南": "Hainan", "重庆": "Chongqing", "四川": "Sichuan", "贵州": "Guizhou",
    "云南": "Yunnan", "西藏": "Tibet", "陕西": "Shaanxi", "甘肃": "Gansu",
    "青海": "Qinghai", "宁夏": "Ningxia", "新疆": "Xinjiang",
    "香港": "Hong Kong", "澳门": "Macao",
    "台湾": "Taiwan",
}

PROVINCE_ORDER = [
    "北京", "天津", "河北", "山西", "内蒙古", "辽宁", "吉林", "黑龙江", "上海", "江苏",
    "浙江", "安徽", "福建", "江西", "山东", "河南", "湖北", "湖南", "广东", "广西",
    "海南", "重庆", "四川", "贵州", "云南", "西藏", "陕西", "甘肃", "青海", "宁夏", "新疆",
]
REGION_ORDER = PROVINCE_ORDER + ["香港", "澳门"]
MAP_KEYS = set(REGION_ORDER).union({"台湾"})

BATCH_ORDER = ["ICH_1st_2006", "ICH_2nd_2008", "ICH_3rd_2011", "ICH_4th_2014", "ICH_5th_2021"]
BATCH_LABELS = {
    "ICH_1st_2006": "2006 Batch I",
    "ICH_2nd_2008": "2008 Batch II",
    "ICH_3rd_2011": "2011 Batch III",
    "ICH_4th_2014": "2014 Batch IV",
    "ICH_5th_2021": "2021 Batch V",
}

CENTROID_OVERRIDE = {
    "北京": (116.4, 40.1), "天津": (117.3, 39.1), "上海": (121.5, 31.2),
    "重庆": (107.6, 29.8), "宁夏": (106.2, 37.5), "海南": (109.7, 19.2),
    "内蒙古": (113.0, 43.8), "黑龙江": (128.0, 47.5), "西藏": (88.0, 31.5),
    "新疆": (86.0, 41.5), "甘肃": (103.8, 38.5), "四川": (103.0, 30.5),
    "云南": (101.5, 25.0), "广西": (108.5, 23.7), "陕西": (109.0, 35.7),
    "山西": (112.3, 37.5), "广东": (113.3, 23.4), "福建": (118.2, 26.1),
    "香港": (114.18, 22.30), "澳门": (113.56, 22.20),
    "台湾": (121.0, 23.7),
}

SPECIAL_REGION_LABELS = {
    "香港": (114.18, 22.30, "Hong Kong", 4, 4, "left"),
    "澳门": (113.56, 22.20, "Macao", 3, -5, "left"),
}

NINE_DASH = [
    [(109.52, 16.36), (109.72, 16.06), (109.88, 15.77), (109.97, 15.53), (109.99, 15.34)],
    [(110.48, 12.43), (110.48, 12.09), (110.45, 11.86), (110.26, 11.39)],
    [(108.34, 7.27), (108.31, 6.73), (108.36, 6.11)],
    [(111.94, 3.55), (112.40, 3.65), (112.92, 3.85)],
    [(115.69, 7.29), (116.41, 8.14)],
    [(118.64, 11.08), (118.86, 11.46), (119.10, 12.06), (119.12, 12.14)],
    [(119.61, 18.14), (119.91, 18.77), (120.12, 19.12)],
    [(121.41, 20.80), (122.12, 21.72)],
    [(122.80, 23.67), (123.00, 24.75)],
    [(119.17, 15.11), (119.17, 15.76), (119.18, 16.27)],
]

PALETTE = {
    "blue": "#2878B5", "cyan": "#4DBBD5", "green": "#00A087", "orange": "#E64B35",
    "yellow": "#F39B7F", "purple": "#7E6148", "grey": "#8491A3", "dark": "#273046",
    "line": "#2B2D42", "grid": "#E8EDF3", "soft": "#F7F9FC",
}

LAMBERT_LON0 = 105.0
LAMBERT_LAT0 = 0.0
LAMBERT_LAT1 = 25.0
LAMBERT_LAT2 = 47.0
EARTH_RADIUS_M = 6371008.8
MAIN_EXTENT = (73.0, 135.0, 17.0, 54.0)


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "processed"
RAW_DIR = PROJECT_DIR / "data" / "source"
ROOT_FIG_DIR = PROJECT_DIR / "results" / "figures"


def set_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "SimSun", "STSong", "DejaVu Serif"],
        "font.sans-serif": ["Arial", "SimSun", "DejaVu Sans"],
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "axes.unicode_minus": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


def china_lambert(lon, lat):
    """China-centered Albers equal-area conic projection, returned in kilometres."""
    lon_arr = np.asarray(lon, dtype=float)
    lat_arr = np.asarray(lat, dtype=float)
    lon0 = np.deg2rad(LAMBERT_LON0)
    lat0 = np.deg2rad(LAMBERT_LAT0)
    lat1 = np.deg2rad(LAMBERT_LAT1)
    lat2 = np.deg2rad(LAMBERT_LAT2)
    lam = np.deg2rad(lon_arr)
    phi = np.deg2rad(lat_arr)

    n = 0.5 * (np.sin(lat1) + np.sin(lat2))
    c = np.cos(lat1) ** 2 + 2 * n * np.sin(lat1)
    rho = EARTH_RADIUS_M * np.sqrt(np.maximum(c - 2 * n * np.sin(phi), 0)) / n
    rho0 = EARTH_RADIUS_M * np.sqrt(max(c - 2 * n * np.sin(lat0), 0)) / n
    theta = n * (lam - lon0)
    x = rho * np.sin(theta) / 1000.0
    y = (rho0 - rho * np.cos(theta)) / 1000.0
    return x, y


def project_ring(ring):
    arr = np.asarray(ring, dtype=float)
    x, y = china_lambert(arr[:, 0], arr[:, 1])
    return np.column_stack([x, y])


def projected_extent(extent=MAIN_EXTENT):
    lon_min, lon_max, lat_min, lat_max = extent
    lons = np.concatenate([
        np.linspace(lon_min, lon_max, 80),
        np.linspace(lon_min, lon_max, 80),
        np.full(80, lon_min),
        np.full(80, lon_max),
    ])
    lats = np.concatenate([
        np.full(80, lat_min),
        np.full(80, lat_max),
        np.linspace(lat_min, lat_max, 80),
        np.linspace(lat_min, lat_max, 80),
    ])
    xs, ys = china_lambert(lons, lats)
    return float(np.nanmin(xs)), float(np.nanmax(xs)), float(np.nanmin(ys)), float(np.nanmax(ys))


def save_figure(fig, basename, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    folders = [out_dir]
    if ROOT_FIG_DIR.resolve() != out_dir.resolve():
        ROOT_FIG_DIR.mkdir(parents=True, exist_ok=True)
        folders.append(ROOT_FIG_DIR)
    for folder in folders:
        fig.savefig(folder / f"{basename}.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
        fig.savefig(folder / f"{basename}.svg", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved {basename}.png and {basename}.svg")


def _read_data_file(name):
    path = DATA_DIR / name
    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig")
    mid = DATA_DIR / "中间过程" / name
    if mid.exists():
        return pd.read_csv(mid, encoding="utf-8-sig")
    raise FileNotFoundError(f"Cannot find data file: {name}")


def _batch_code_from_year(year):
    y = str(year)
    return {
        "2006": "ICH_1st_2006",
        "2008": "ICH_2nd_2008",
        "2011": "ICH_3rd_2011",
        "2014": "ICH_4th_2014",
        "2021": "ICH_5th_2021",
    }.get(y, "")


def load_items():
    df = _read_data_file("folklore_records_with_xy_coordinates.csv")
    out = pd.DataFrame({
        "source_id": df["官网记录ID"].astype(str),
        "item_no": df["项目序号"].astype(str),
        "item_code": df["项目编号"].astype(str),
        "project_num": df["项目编码"].astype(str),
        "item_name": df["项目名称"].astype(str),
        "category": df["官方类别"].astype(str),
        "project_type": df["项目类型"].astype(str),
        "region_raw": df["申报地区或单位"].astype(str),
        "protect_unit": df["保护单位"].astype(str),
        "year": df["公布年份"].astype(str),
        "batch": df["公布年份"].apply(_batch_code_from_year),
        "province": df["匹配省份"].apply(province_key),
        "x": pd.to_numeric(df["经度_X"], errors="coerce"),
        "y": pd.to_numeric(df["纬度_Y"], errors="coerce"),
        "match_level": df["坐标匹配层级"].astype(str),
    })
    return out


def load_panel():
    items = load_items()
    mapped = items[items["province"].isin(REGION_ORDER)].copy()
    counts = mapped.groupby("province").size().reindex(REGION_ORDER).fillna(0).astype(int)

    pop = _read_data_file("province_minority_share_2020.csv")
    pop = pop.rename(columns={
        "minority_pop_2020": "minority_pop",
        "minority_pct_2020": "minority_pct",
        "minority_share_2020": "minority_share",
        "han_pop_2020": "han_pop",
        "han_pct_2020": "han_pct",
    })
    pop["province"] = pop["province"].apply(province_key)
    pop = pop[pop["province"].isin(PROVINCE_ORDER)].copy()
    special_regions = pd.DataFrame({
        "province": ["香港", "澳门"],
        "pop_2020": [7_474_000, 683_000],
        "minority_pop": [np.nan, np.nan],
        "minority_pct": [np.nan, np.nan],
        "minority_share": [np.nan, np.nan],
        "han_pop": [np.nan, np.nan],
        "han_pct": [np.nan, np.nan],
    })
    pop = pd.concat([pop, special_regions], ignore_index=True, sort=False)
    pop["province_order"] = pop["province"].apply(lambda x: REGION_ORDER.index(x))
    pop = pop.sort_values("province_order").reset_index(drop=True)
    pop["pop_2020"] = pd.to_numeric(pop["pop_2020"], errors="coerce")
    pop["minority_pop"] = pd.to_numeric(pop["minority_pop"], errors="coerce")
    pop["minority_pct"] = pd.to_numeric(pop["minority_pct"], errors="coerce")
    pop["minority_share"] = pd.to_numeric(pop["minority_share"], errors="coerce")
    pop["han_pop"] = pd.to_numeric(pop["han_pop"], errors="coerce")
    pop["han_pct"] = pd.to_numeric(pop["han_pct"], errors="coerce")
    pop["folk_count"] = pop["province"].map(counts).fillna(0).astype(int)
    pop["pop_million"] = pop["pop_2020"] / 1_000_000.0
    pop["folk_per_million"] = pop["folk_count"] / pop["pop_million"]
    pop["province_en"] = pop["province"].map(PROVINCE_EN)

    # Backward-compatible aliases used by older figure scripts.
    pop["festival_count"] = pop["folk_count"]
    pop["festival_per_million"] = pop["folk_per_million"]
    return pop


def load_analysis_panel():
    panel = load_panel()
    cols = ["pop_2020", "minority_share", "folk_per_million"]
    return panel[panel["province"].isin(PROVINCE_ORDER)].dropna(subset=cols).copy()


def province_key(name):
    name = str(name)
    for suffix in ["特别行政区", "壮族自治区", "回族自治区", "维吾尔自治区", "自治区", "省", "市"]:
        name = name.replace(suffix, "")
    if name.startswith("内蒙古"):
        return "内蒙古"
    if name.startswith("黑龙江"):
        return "黑龙江"
    return name.strip()


def decode_topojson():
    with open(RAW_DIR / "cn-atlas.json", "r", encoding="utf-8") as f:
        topo = json.load(f)
    sx, sy = topo["transform"]["scale"]
    tx, ty = topo["transform"]["translate"]
    arcs = []
    for arc in topo["arcs"]:
        x = y = 0
        pts = []
        for dx, dy in arc:
            x += dx
            y += dy
            pts.append((x * sx + tx, y * sy + ty))
        arcs.append(pts)

    provinces = []
    for geom in topo["objects"]["provinces"]["geometries"]:
        rings = []
        polys = geom["arcs"] if geom["type"] == "MultiPolygon" else [geom["arcs"]]
        for poly in polys:
            for ring in poly:
                pts = []
                for arc_id in ring:
                    rev = arc_id < 0
                    real = ~arc_id if rev else arc_id
                    ap = list(reversed(arcs[real])) if rev else arcs[real]
                    pts.extend(ap[1:] if pts else ap)
                if len(pts) >= 3:
                    rings.append(pts)
        arc_ids = set()

        def collect(obj):
            if isinstance(obj, int):
                arc_ids.add(~obj if obj < 0 else obj)
            else:
                for item in obj:
                    collect(item)

        collect(geom["arcs"])
        raw = geom["properties"].get("地名", geom["properties"].get("name", ""))
        provinces.append({"name": raw, "key": province_key(raw), "rings": rings, "arc_ids": arc_ids})
    return provinces


def ring_area(ring):
    arr = np.asarray(ring, dtype=float)
    x, y = arr[:, 0], arr[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def province_centroid(province):
    key = province["key"]
    if key in CENTROID_OVERRIDE:
        return CENTROID_OVERRIDE[key]
    if not province["rings"]:
        return (0, 0)
    arr = np.asarray(max(province["rings"], key=ring_area), dtype=float)
    return float(arr[:, 0].mean()), float(arr[:, 1].mean())


def make_adjacency(provinces):
    mainland = [p for p in provinces if p["key"] in PROVINCE_ORDER]
    neighbors = {p["key"]: set() for p in mainland}
    for i, a in enumerate(mainland):
        for b in mainland[i + 1:]:
            if a["arc_ids"].intersection(b["arc_ids"]):
                neighbors[a["key"]].add(b["key"])
                neighbors[b["key"]].add(a["key"])
    return neighbors


def set_bounds_from_patches(ax, patches, pad=0.045, right_pad=0.110):
    pts = np.vstack([p.get_xy() for p in patches if len(p.get_xy())])
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    dx = x1 - x0
    dy = y1 - y0
    ax.set_xlim(x0 - dx * pad, x1 + dx * right_pad)
    ax.set_ylim(y0 - dy * pad, y1 + dy * pad)


def draw_base_map(ax, provinces, value_map=None, cmap="YlOrRd", norm=None, missing="#F4F6F8", edge="#475569", lw=0.45, extent=None):
    patches, colors = [], []
    cm = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    for prov in provinces:
        key = prov["key"]
        if key not in MAP_KEYS:
            continue
        val = None if value_map is None else value_map.get(key, np.nan)
        for ring in prov["rings"]:
            if extent is None and np.mean([pt[1] for pt in ring]) < 17.0:
                continue
            patches.append(mpatches.Polygon(project_ring(ring), closed=True))
            colors.append(missing if value_map is None or pd.isna(val) else cm(norm(val) if norm else val))
    pc = PatchCollection(patches, facecolor=colors, edgecolor=edge, linewidth=lw, zorder=2)
    ax.add_collection(pc)
    if extent is None:
        set_bounds_from_patches(ax, patches)
    else:
        x0, x1, y0, y1 = projected_extent(extent)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")
    return pc


def add_map_frame(ax):
    ax.axis("on")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.05)
        spine.set_edgecolor("#334155")
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    dx = xlim[1] - xlim[0]
    dy = ylim[1] - ylim[0]
    for lon in [80, 120]:
        xb, _ = china_lambert(lon, MAIN_EXTENT[2])
        xt, _ = china_lambert(lon, MAIN_EXTENT[3])
        xb = float(xb)
        xt = float(xt)
        if xlim[0] <= xb <= xlim[1]:
            ax.plot([xb, xb], [ylim[0], ylim[0] + dy * 0.012], color="#111827", lw=1.0, clip_on=False, zorder=20)
            ax.text(xb, ylim[0] - dy * 0.038, f"{lon}°E", ha="center", va="top", fontsize=11, fontweight="bold")
        if xlim[0] <= xt <= xlim[1]:
            ax.plot([xt, xt], [ylim[1], ylim[1] - dy * 0.012], color="#111827", lw=1.0, clip_on=False, zorder=20)
            ax.text(xt, ylim[1] + dy * 0.028, f"{lon}°E", ha="center", va="bottom", fontsize=11, fontweight="bold")
    for lat in [20, 40]:
        xl, yl = china_lambert(MAIN_EXTENT[0], lat)
        xr, yr = china_lambert(MAIN_EXTENT[1], lat)
        xl, yl, xr, yr = float(xl), float(yl), float(xr), float(yr)
        if ylim[0] <= yl <= ylim[1]:
            ax.plot([xlim[0], xlim[0] + dx * 0.008], [yl, yl], color="#111827", lw=1.0, clip_on=False, zorder=20)
            ax.text(xlim[0] - dx * 0.018, yl, f"{lat}°N", ha="right", va="center", rotation=90,
                    fontsize=11, fontweight="bold")
        if ylim[0] <= yr <= ylim[1]:
            ax.plot([xlim[1], xlim[1] - dx * 0.008], [yr, yr], color="#111827", lw=1.0, clip_on=False, zorder=20)
            ax.text(xlim[1] + dx * 0.018, yr, f"{lat}°N", ha="left", va="center", rotation=90,
                    fontsize=11, fontweight="bold")
    add_special_region_labels(ax)


def add_map_labels(ax, provinces, fontsize=6.0, skip_extra=None):
    skip = {"北京", "天津", "上海"}
    if skip_extra:
        skip = skip.union(set(skip_extra))
    for prov in provinces:
        key = prov["key"]
        if key in skip or key not in MAP_KEYS:
            continue
        x, y = province_centroid(prov)
        x, y = china_lambert(x, y)
        x, y = float(x), float(y)
        label = PROVINCE_EN.get(key, key)
        if key in {"香港", "澳门"}:
            continue
        if key == "内蒙古":
            label = "Inner\nMongolia"
        ax.text(x, y, label, ha="center", va="center", fontsize=fontsize, color="#1F2937", zorder=4)


def add_special_region_labels(ax, fontsize=5.4):
    for key, (lon, lat, label, dx_pts, dy_pts, ha) in SPECIAL_REGION_LABELS.items():
        x, y = china_lambert(lon, lat)
        ax.annotate(label, xy=(x, y), xytext=(dx_pts, dy_pts), textcoords="offset points",
                    ha=ha, va="top", fontsize=fontsize, color="#111827", zorder=8)


def add_north_arrow(ax, x=0.940, y=0.895, size=0.075):
    ax.annotate("N", xy=(x, y + size * 0.70), xycoords="axes fraction", ha="center", va="center",
                fontsize=12, fontweight="bold")
    ax.annotate("", xy=(x, y + size * 0.48), xytext=(x, y - size * 0.45), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=1.25, color="#111827", mutation_scale=16))


def add_scale_bar(ax, x=0.080, y=0.085, length_km=1000.0, label="1000 km"):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x0 = xlim[0] + x * (xlim[1] - xlim[0])
    y0 = ylim[0] + y * (ylim[1] - ylim[0])
    tick_h = (ylim[1] - ylim[0]) * 0.018
    ax.plot([x0, x0 + length_km], [y0, y0], color="#111827", lw=2.0, solid_capstyle="butt", zorder=5)
    ax.plot([x0, x0], [y0 - tick_h * 0.55, y0 + tick_h * 0.55], color="#111827", lw=1.0, zorder=5)
    ax.plot([x0 + length_km, x0 + length_km], [y0 - tick_h * 0.55, y0 + tick_h * 0.55], color="#111827", lw=1.0, zorder=5)
    ax.text(x0, y0 - tick_h * 1.4, "0", ha="center", va="top", fontsize=8)
    ax.text(x0 + length_km, y0 - tick_h * 1.4, label, ha="center", va="top", fontsize=8)


def draw_nine_dash(ax, color="#111827", lw=0.7):
    for seg in NINE_DASH:
        xs, ys = zip(*seg)
        xs, ys = china_lambert(xs, ys)
        ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round", zorder=5)


def add_south_china_inset(fig, ax_parent, provinces, value_map=None, cmap="YlOrRd", norm=None):
    box = ax_parent.get_position()
    w = box.width * 0.150
    h = box.height * 0.285
    left = box.x1 - w - 0.006
    bottom = box.y0 + 0.082
    axins = fig.add_axes([left, bottom, w, h])
    inset_extent = (106, 125.5, 3.0, 25.5)
    draw_base_map(axins, provinces, value_map=value_map, cmap=cmap, norm=norm, edge="#334155", lw=0.28, extent=inset_extent)
    x0, x1, y0, y1 = projected_extent(inset_extent)
    axins.set_xlim(x0, x1)
    axins.set_ylim(y0, y1)
    axins.set_facecolor("#F8FBFF")
    draw_nine_dash(axins, lw=0.65)
    axins.axis("on")
    for sp in axins.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.95)
        sp.set_edgecolor("#334155")
    axins.set_xticks([])
    axins.set_yticks([])
    axins.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    return axins


def add_horizontal_colorbar(fig, ax, cmap, norm, label):
    box = ax.get_position()
    cax = fig.add_axes([box.x0 + box.width * 0.230, box.y1 - box.height * 0.062, box.width * 0.420, 0.016])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, length=2)
    cb.ax.xaxis.set_label_position("top")
    cb.set_label(label, fontsize=9, labelpad=2)
    return cb


def gradient_barh(ax, labels, values, cmap_name="Blues", edge="#FFFFFF"):
    cm = plt.get_cmap(cmap_name)
    vmax = max(values) if len(values) else 1
    for i, (lab, val) in enumerate(zip(labels, values)):
        color = cm(0.35 + 0.55 * (val / vmax if vmax else 0))
        ax.barh(i, val, height=0.64, color=color, edgecolor=edge, linewidth=0.8, zorder=3)
        ax.text(val + vmax * 0.018, i, f"{val:.3f}", va="center", ha="left", fontsize=10, color="#263238")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()


def ols_fit(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xm, ym = x.mean(), y.mean()
    slope = np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2)
    intercept = ym - slope * xm
    fitted = intercept + slope * x
    residual = y - fitted
    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum((y - ym) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot else np.nan
    return intercept, slope, fitted, residual, r2


def spatial_lag(values, neighbors):
    out = {}
    for p in values:
        ns = [n for n in neighbors.get(p, []) if n in values]
        out[p] = float(np.mean([values[n] for n in ns])) if ns else np.nan
    return out


def moran_global(values, neighbors):
    keys = [k for k in PROVINCE_ORDER if k in values]
    x = np.array([values[k] for k in keys], dtype=float)
    z = x - x.mean()
    num = 0.0
    wsum = 0.0
    key_index = {k: i for i, k in enumerate(keys)}
    for i, a in enumerate(keys):
        for b in neighbors.get(a, []):
            if b in key_index:
                num += z[i] * z[key_index[b]]
                wsum += 1.0
    den = np.sum(z ** 2)
    return len(keys) / wsum * num / den if wsum and den else np.nan


def local_lisa(values, neighbors, permutations=499, seed=2026, alpha=0.05):
    rng = np.random.default_rng(seed)
    keys = [k for k in PROVINCE_ORDER if k in values]
    x = np.array([values[k] for k in keys], dtype=float)
    z = (x - x.mean()) / x.std(ddof=0)
    zmap = dict(zip(keys, z))
    clusters = {}
    for k in keys:
        ns = [n for n in neighbors.get(k, []) if n in zmap]
        if not ns:
            clusters[k] = "NS"
            continue
        lag = np.mean([zmap[n] for n in ns])
        observed = zmap[k] * lag
        pool = np.array([zmap[j] for j in keys if j != k])
        sims = []
        for _ in range(permutations):
            draw = rng.choice(pool, size=len(ns), replace=False if len(pool) >= len(ns) else True)
            sims.append(zmap[k] * np.mean(draw))
        p = (np.sum(np.abs(sims) >= abs(observed)) + 1) / (permutations + 1)
        if p >= alpha:
            clusters[k] = "NS"
        elif zmap[k] >= 0 and lag >= 0:
            clusters[k] = "HH"
        elif zmap[k] < 0 and lag < 0:
            clusters[k] = "LL"
        elif zmap[k] >= 0 and lag < 0:
            clusters[k] = "HL"
        else:
            clusters[k] = "LH"
    return clusters
