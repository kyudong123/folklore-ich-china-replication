# Replication package

This repository contains the source files, analysis data, spatial weights, and Python scripts for the provincial analysis of nationally listed folklore intangible cultural heritage in China.

The sample consists of 492 official folklore records. Of these, 481 have a regional location and 471 are assigned to the 31 mainland provincial-level units used in population-standardized and spatial analyses. The 11 national or central-level records without an assignable province remain documented in the item-level audit.

## Reproduce the analysis

Use Python 3.12 and install the pinned packages in `environment/requirements.txt`. From the repository root, run:

```powershell
python run_all.py
```

The scripts reconstruct the analysis inputs, calculate descriptive and regression results, conduct spatial analyses with Queen and KNN-4 weights, and generate derived figures and tables under `results/`. The generated files can be recreated from the staged data and scripts.

## Data and provenance

The source register is in `data/source/README_sources.md`. The item-level inclusion and geographic-matching audit is in `data/intermediate/record_inclusion_geocoding_audit.csv`. Variable names, definitions, and units are documented in `data/intermediate/data_dictionary.csv`. The provincial panel and spatial weights used for analysis are in `data/processed/` and `data/weights/`.

`MANIFEST_sha256.csv` lists file sizes and SHA-256 hashes for the current package. Rebuild it after package changes with `python src/build_manifest.py`.
