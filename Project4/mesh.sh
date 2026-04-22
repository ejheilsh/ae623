#!/bin/bash
# ============================================================================
# mesh.sh — Generate q=1 and q=3 cascade meshes of any target size
# ============================================================================
# Usage:
#   bash mesh.sh                  # default: 1000 elements -> grids/1k.gri + 1k_q3.gri
#   TARGET=500  bash mesh.sh      # 500 elements  -> grids/500.gri  + 500_q3.gri
#   TARGET=2000 bash mesh.sh      # 2000 elements -> grids/2k.gri   + 2k_q3.gri
#
# Outputs (in grids/):
#   <label>.gri      straight q=1 mesh
#   <label>_q3.gri   curved q=3 mesh (wall edges projected onto blade spline)
# ============================================================================
set -e

TARGET="${TARGET:-1000}"

# Auto-derive label: multiples of 1000 -> Nk, otherwise use the number
if (( TARGET % 1000 == 0 )); then
  LABEL="$(( TARGET / 1000 ))k"
else
  LABEL="${TARGET}"
fi

PYTHON="/c/Users/William Zhang/AppData/Local/Programs/Python/Python313/python.exe"
# Fallback: try py launcher if above path doesn't exist
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=$(which python || which python3)
fi

CURVE_MESH="../Project3/curve_mesh.py"
Q1_GRID="grids/${LABEL}.gri"
Q3_GRID="grids/${LABEL}_q3.gri"

echo "============================================================"
echo "  Generating mesh: target=${TARGET} elements, label=${LABEL}"
echo "  Python: ${PYTHON}"
echo "============================================================"

echo ""
echo "--- Step 1: Generate q=1 mesh (grids/${LABEL}.gri) ---"
"$PYTHON" v2.py --target "${TARGET}" --label "${LABEL}"

echo ""
echo "--- Step 2: Curve to q=3 (grids/${LABEL}_q3.gri) ---"
"$PYTHON" "${CURVE_MESH}" "${Q1_GRID}" --q 3 -o "${Q3_GRID}"

echo ""
echo "Done. Outputs:"
echo "  q=1 : ${Q1_GRID}"
echo "  q=3 : ${Q3_GRID}"