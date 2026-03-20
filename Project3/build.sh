#!/bin/bash

# Build script for Euler Solver
# Default:
#   ./build.sh
# OpenMP build on macOS/Homebrew libomp:
#   OPENMP=1 ./build.sh

COMPILER=clang++
FLAGS="-O3 -std=c++17 -Wall"
OUTPUT="euler_solver"
SRC="src/"

# OpenMP is opt-in so the default build stays identical to the original solver build.
if [ "${OPENMP:-0}" = "1" ]; then
    echo "Building with OpenMP support..."
    if command -v brew >/dev/null 2>&1; then
        LIBOMP_PREFIX="$(brew --prefix libomp 2>/dev/null || true)"
    else
        LIBOMP_PREFIX=""
    fi

    if [ -n "$LIBOMP_PREFIX" ]; then
        # Apple clang needs explicit include/library paths for Homebrew's libomp.
        FLAGS="$FLAGS -Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include"
        LDFLAGS="-L${LIBOMP_PREFIX}/lib -lomp"
    else
        echo "Warning: libomp not found via brew; trying default OpenMP flags."
        FLAGS="$FLAGS -Xpreprocessor -fopenmp"
        LDFLAGS="-lomp"
    fi
else
    LDFLAGS=""
fi

echo "Compiling..."
$COMPILER $FLAGS $SRC/main.cpp $SRC/State.cpp $SRC/Mesh.cpp $SRC/Fluxes.cpp $SRC/Solver.cpp -o $OUTPUT $LDFLAGS

if [ $? -eq 0 ]; then
    echo "Successfully built $OUTPUT"
else
    echo "Build failed"
    exit 1
fi
