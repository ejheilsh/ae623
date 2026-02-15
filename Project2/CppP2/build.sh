#!/bin/bash

set -e   # stop if compilation fails

echo "Compiling..."

clang++ main.cpp mesh.cpp -std=c++17 -Wall -Wextra -O0 -g -o mesh

echo "Build complete."
