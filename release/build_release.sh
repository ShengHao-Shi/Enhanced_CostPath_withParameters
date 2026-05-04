#!/usr/bin/env bash
# Build distributable zip packages for a GitHub Release.
# Run from the repository root:
#   bash release/build_release.sh
#
# Output:
#   release/EnhancedCostPath_ArcGIS.zip
#   release/EnhancedCostPath_Standalone.zip

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RELEASE_DIR="${REPO_ROOT}/release"
TMP_DIR="$(mktemp -d)"

cleanup() { rm -rf "${TMP_DIR}"; }
trap cleanup EXIT

echo "Building release packages from ${REPO_ROOT} …"

# ---------------------------------------------------------------------------
# ArcGIS package
# ---------------------------------------------------------------------------
ARCGIS_DIR="${TMP_DIR}/EnhancedCostPath_ArcGIS"
mkdir -p "${ARCGIS_DIR}/pure_python" "${ARCGIS_DIR}/numba_accelerated"

cp "${REPO_ROOT}/arcgis_toolbox_with_progress.pyt" "${ARCGIS_DIR}/"
cp "${REPO_ROOT}/requirements.txt"                  "${ARCGIS_DIR}/"
cp "${REPO_ROOT}/pure_python/__init__.py"            "${ARCGIS_DIR}/pure_python/"
cp "${REPO_ROOT}/pure_python/cost_aware_straighten_lcp.py" \
   "${ARCGIS_DIR}/pure_python/"
cp "${REPO_ROOT}/numba_accelerated/__init__.py"      "${ARCGIS_DIR}/numba_accelerated/"
cp "${REPO_ROOT}/numba_accelerated/cost_aware_straighten_lcp.py" \
   "${ARCGIS_DIR}/numba_accelerated/"

(cd "${TMP_DIR}" && zip -r "${RELEASE_DIR}/EnhancedCostPath_ArcGIS.zip" \
   EnhancedCostPath_ArcGIS/)
echo "  Created release/EnhancedCostPath_ArcGIS.zip"

# ---------------------------------------------------------------------------
# Standalone package
# ---------------------------------------------------------------------------
STANDALONE_DIR="${TMP_DIR}/EnhancedCostPath_Standalone"
mkdir -p "${STANDALONE_DIR}/pure_python" "${STANDALONE_DIR}/numba_accelerated"

cp "${REPO_ROOT}/requirements.txt"                  "${STANDALONE_DIR}/"
cp "${REPO_ROOT}/pure_python/__init__.py"            "${STANDALONE_DIR}/pure_python/"
cp "${REPO_ROOT}/pure_python/cost_aware_straighten_lcp.py" \
   "${STANDALONE_DIR}/pure_python/"
cp "${REPO_ROOT}/numba_accelerated/__init__.py"      "${STANDALONE_DIR}/numba_accelerated/"
cp "${REPO_ROOT}/numba_accelerated/cost_aware_straighten_lcp.py" \
   "${STANDALONE_DIR}/numba_accelerated/"

(cd "${TMP_DIR}" && zip -r "${RELEASE_DIR}/EnhancedCostPath_Standalone.zip" \
   EnhancedCostPath_Standalone/)
echo "  Created release/EnhancedCostPath_Standalone.zip"

echo "Done."
