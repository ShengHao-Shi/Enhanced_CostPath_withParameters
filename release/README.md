# Release Packages

This directory contains scripts to build distributable packages for two
usage scenarios.  The zip files themselves are **not committed** to the
repository; they are built locally and attached to a GitHub Release.

---

## Package Contents

### `EnhancedCostPath_ArcGIS.zip`

Intended for **ArcGIS Pro users** who want to add the toolbox to the Catalog
pane and run the algorithm through the standard geoprocessing UI.

```
EnhancedCostPath_ArcGIS/
├── arcgis_toolbox_with_progress.pyt   ← ArcGIS Python Toolbox (add this)
├── pure_python/
│   ├── __init__.py
│   └── cost_aware_straighten_lcp.py   ← pure-Python algorithm (required)
├── numba_accelerated/
│   ├── __init__.py
│   └── cost_aware_straighten_lcp.py   ← Numba JIT version (optional, faster)
└── requirements.txt
```

**Usage**: In ArcGIS Pro → Catalog → Toolboxes → Add Toolbox → select
`arcgis_toolbox_with_progress.pyt`.

---

### `EnhancedCostPath_Standalone.zip`

Intended for **standalone Python users** (no ArcGIS licence required).

```
EnhancedCostPath_Standalone/
├── pure_python/
│   ├── __init__.py
│   └── cost_aware_straighten_lcp.py
├── numba_accelerated/
│   ├── __init__.py
│   └── cost_aware_straighten_lcp.py
└── requirements.txt
```

Install dependencies:

```bash
pip install numpy rasterio        # minimum
pip install numba                  # optional, enables 20-50x speedup
```

---

## Building the packages

Run the script from the **repository root**:

```bash
bash release/build_release.sh
```

The script creates the two zip files inside `release/`.

---

## Publishing a GitHub Release

1. Build the packages with `build_release.sh`.
2. Create a new tag, e.g. `v1.0.0`:

   ```bash
   git tag -a v1.0.0 -m "Release v1.0.0"
   git push origin v1.0.0
   ```

3. On GitHub → Releases → Draft a new release → attach both zip files.
