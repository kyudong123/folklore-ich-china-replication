"""Build deterministic raster derivatives and a SHA-256 file inventory."""
from pathlib import Path
from hashlib import sha256
from PIL import Image
import csv

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
for png in sorted([*FIG.glob("fig[0-9][0-9]_*.png"), *FIG.glob("supp_fig_s1_*.png")]):
    with Image.open(png) as im:
        rgb = im.convert("RGB")
        rgb.save(png.with_suffix(".jpg"), quality=95, optimize=True)
        rgb.save(png.with_suffix(".tif"), compression="tiff_lzw")
manifest = ROOT / "MANIFEST_sha256.csv"
rows = []
for path in sorted(ROOT.rglob("*")):
    if not path.is_file() or path == manifest or ".git" in path.parts or "__pycache__" in path.parts:
        continue
    rel = path.relative_to(ROOT).as_posix()
    rows.append({"relative_path": rel, "bytes": path.stat().st_size,
                 "sha256": sha256(path.read_bytes()).hexdigest()})
with manifest.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["relative_path", "bytes", "sha256"])
    writer.writeheader(); writer.writerows(rows)
print(f"Updated generated raster derivatives and SHA-256 inventory for {len(rows)} package files.")
