#!/bin/bash

# Build script for Euler Solver

COMPILER=clang++
FLAGS="-O3 -std=c++17 -Wall"
OUTPUT="euler_solver"

echo "Compiling..."
$COMPILER $FLAGS main.cpp State.cpp Mesh.cpp Fluxes.cpp Solver.cpp -o $OUTPUT

if [ $? -eq 0 ]; then
    echo "Successfully built $OUTPUT"
else
    echo "Build failed"
    exit 1
fi
