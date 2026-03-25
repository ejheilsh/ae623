#!/bin/bash
set -euo pipefail

REMOTE_ROOT="ejheilsh@gl:/home/ejheilsh/repos/ae623/Project3"
INCLUDE_FILE="includeFile.txt"

if [[ ! -f "$INCLUDE_FILE" ]]; then
  echo "Missing $INCLUDE_FILE"
  exit 1
fi

rsync -azvP --files-from="$INCLUDE_FILE" "$REMOTE_ROOT/" .

cp data_steady/steady_32k_q3_p1_temp_iter_latest.bin \
  data_steady/steady_32k_q3_p1_results.bin
cp data_steady/steady_32k_q3_p1_temp_iter_latest_dg.bin \
  data_steady/steady_32k_q3_p1_results_dg.bin
cp data_steady/steady_32k_q3_p1_temp_iter_residual.bin \
  data_steady/steady_32k_q3_p1_residual.bin
cp data_steady/steady_32k_q3_p1_temp_iter_cell_res.bin \
  data_steady/steady_32k_q3_p1_cell_res.bin

echo "Updated local 32k_q3_p1 files from latest remote temp checkpoint."
