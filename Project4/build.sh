#!/bin/bash
set -e

OUTPUT="euler_solver"
SRC="src"
CXXFLAGS="-O3 -std=c++17 -Wall"
LDFLAGS=""

SOURCES=(
    "$SRC/main.cpp"
    "$SRC/State.cpp"
    "$SRC/Mesh.cpp"
    "$SRC/Fluxes.cpp"
    "$SRC/Solver.cpp"
    "$SRC/Adjoint.cpp"
    "$SRC/MeshRefinement.cpp"
)

if [ -n "${CXX:-}" ]; then
    COMPILER="$CXX"
elif command -v g++ >/dev/null 2>&1; then
    COMPILER="g++"
elif command -v clang++ >/dev/null 2>&1; then
    COMPILER="clang++"
else
    echo "Error: no suitable C++ compiler found."
    exit 1
fi

echo "Using compiler: $COMPILER"

COMPILER_VERSION="$($COMPILER --version 2>/dev/null || true)"

IS_CLANG=0
IS_GCC=0

if echo "$COMPILER_VERSION" | grep -qi "clang"; then
    IS_CLANG=1
fi

if echo "$COMPILER_VERSION" | grep -qiE "gcc|g\+\+"; then
    IS_GCC=1
fi

if [ "${OPENMP:-0}" = "1" ]; then
    echo "Building with OpenMP support..."

    if [ "$IS_GCC" -eq 1 ]; then
        CXXFLAGS="$CXXFLAGS -fopenmp"
    elif [ "$IS_CLANG" -eq 1 ]; then
        if command -v brew >/dev/null 2>&1; then
            LIBOMP_PREFIX="$(brew --prefix libomp 2>/dev/null || true)"
        else
            LIBOMP_PREFIX=""
        fi

        if [ -n "$LIBOMP_PREFIX" ]; then
            CXXFLAGS="$CXXFLAGS -Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include"
            LDFLAGS="$LDFLAGS -L${LIBOMP_PREFIX}/lib -lomp"
        else
            echo "Warning: clang detected but Homebrew libomp was not found."
            CXXFLAGS="$CXXFLAGS -Xpreprocessor -fopenmp"
            LDFLAGS="$LDFLAGS -lomp"
        fi
    else
        CXXFLAGS="$CXXFLAGS -fopenmp"
    fi
fi

if [ "$IS_GCC" -eq 1 ]; then
    LDFLAGS="$LDFLAGS -lstdc++fs"
fi

echo "Compiling..."
echo "$COMPILER $CXXFLAGS ${SOURCES[*]} -o $OUTPUT $LDFLAGS"

"$COMPILER" $CXXFLAGS "${SOURCES[@]}" -o "$OUTPUT" $LDFLAGS

echo "Successfully built $OUTPUT"
