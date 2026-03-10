#!/bin/bash

# Build script for Euler Solver

COMPILER=clang++
FLAGS="-O3 -std=c++17 -Wall"
OUTPUT="euler_solver"
SRC="src/"

echo "Compiling..."
$COMPILER $FLAGS $SRC/main.cpp $SRC/State.cpp $SRC/Mesh.cpp $SRC/Fluxes.cpp $SRC/Solver.cpp -o $OUTPUT

if [ $? -eq 0 ]; then
    echo "Successfully built $OUTPUT"
else
    echo "Build failed"
    exit 1
fi
