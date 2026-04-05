#!/usr/bin/env bash
set -euo pipefail

# Benchmark steady solver runtime versus OpenMP thread count.
#
# Default case:
#   OMP_NUM_THREADS=<3..10> ./euler_solver grids/8k_q3.gri 3 1 hlle 10000 steady
#
# Outputs:
#   benchmark_threads.csv
#   benchmark_threads.png
#
# Usage:
#   chmod +x benchmark_threads.sh
#   ./benchmark_threads.sh
#
# Optional overrides:
#   MESH=grids/8k_q3.gri ORDER=3 CFL=1 FLUX=hlle ITERCAP=10000 MODE=steady ./benchmark_threads.sh

MESH="${MESH:-grids/8k_q3.gri}"
ORDER="${ORDER:-3}"
CFL="${CFL:-1}"
FLUX="${FLUX:-hlle}"
ITERCAP="${ITERCAP:-10000}"
MODE="${MODE:-steady}"
THREAD_MIN="${THREAD_MIN:-3}"
THREAD_MAX="${THREAD_MAX:-10}"
CSV_OUT="${CSV_OUT:-benchmark_threads.csv}"
PLOT_OUT="${PLOT_OUT:-benchmark_threads.png}"

if [[ ! -x ./euler_solver ]]; then
  echo "Error: ./euler_solver not found or not executable." >&2
  exit 1
fi

if [[ ! -f "$MESH" ]]; then
  echo "Error: mesh file '$MESH' not found." >&2
  exit 1
fi

echo "threads,seconds" > "$CSV_OUT"

echo "Benchmarking:"
echo "  mesh     = $MESH"
echo "  order    = $ORDER"
echo "  CFL      = $CFL"
echo "  flux     = $FLUX"
echo "  itercap  = $ITERCAP"
echo "  mode     = $MODE"
echo "  threads  = ${THREAD_MIN}..${THREAD_MAX}"
echo

for threads in $(seq "$THREAD_MIN" "$THREAD_MAX"); do
  echo "Running with OMP_NUM_THREADS=$threads ..."

  time_file="$(mktemp)"
  log_file="$(mktemp)"

  /usr/bin/time -p -o "$time_file" \
    env OMP_NUM_THREADS="$threads" \
    ./euler_solver "$MESH" "$ORDER" "$CFL" "$FLUX" "$ITERCAP" "$MODE" \
    > "$log_file" 2>&1

  elapsed="$(awk '/^real / {print $2}' "$time_file")"
  echo "$threads,$elapsed" >> "$CSV_OUT"
  echo "  elapsed = ${elapsed} s"

  rm -f "$time_file" "$log_file"
done

python3 - "$CSV_OUT" "$PLOT_OUT" <<'PY'
import csv
import sys

import matplotlib.pyplot as plt

csv_file = sys.argv[1]
plot_file = sys.argv[2]

threads = []
seconds = []
with open(csv_file, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        threads.append(int(row["threads"]))
        seconds.append(float(row["seconds"]))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(threads, seconds, marker="o", linewidth=2)
ax.set_xlabel("OMP Threads")
ax.set_ylabel("Wall Time [s]")
ax.set_title("Solver Runtime vs OpenMP Thread Count")
ax.grid(True, alpha=0.3)
ax.set_xticks(threads)
fig.tight_layout()
fig.savefig(plot_file, dpi=150)
print(f"Saved {plot_file}")
PY

echo
echo "Saved $CSV_OUT"
echo "Saved $PLOT_OUT"
