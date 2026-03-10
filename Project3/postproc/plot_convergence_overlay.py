#!/usr/bin/env python3
"""
Overlay convergence histories from residual binary files.

Supported filename patterns:
  - conv_<grid>_<order>_<flux>_<case>.bin
  - steady_<grid>_residual.bin
  - steady_<grid>_o<order>_<flux>_residual.bin

Binary format:
  int32 N
  float64 residual[0..N-1]
"""

import argparse
import glob
import os
import re
import struct
import sys

import matplotlib.pyplot as plt
import numpy as np


def read_residual(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        raw_n = f.read(4)
        if len(raw_n) != 4:
            raise ValueError(f"{path}: file too short (missing length header)")
        n = struct.unpack("i", raw_n)[0]
        if n < 0:
            raise ValueError(f"{path}: invalid length {n}")
        raw = f.read(8 * n)
        if len(raw) != 8 * n:
            raise ValueError(f"{path}: truncated data (expected {n} doubles)")
        vals = struct.unpack("d" * n, raw)
    return np.asarray(vals, dtype=float)


def parse_label_from_name(path: str) -> str:
    name = os.path.basename(path)
    stem = name[:-4] if name.endswith(".bin") else name

    # conv_2k_1_hlle_1.bin
    m = re.match(r"^conv_([^_]+)_([^_]+)_([^_]+)_([^_]+)$", stem)
    if m:
        grid, order, flux, case = m.groups()
        ord_label = f"O{order}"
        return f"{grid} {ord_label} {flux.upper()} case{case}"

    # conv_gri128k_ord1_hlle_cfl1.bin
    m = re.match(r"^conv_gri([^_]+)_ord([^_]+)_([^_]+)_cfl([^_]+)$", stem)
    if m:
        grid, order, flux, cfl = m.groups()
        return f"{grid} O{order} {flux.upper()} CFL={cfl}"

    # steady_8k_o2_hlle_residual.bin
    m = re.match(r"^steady_([^_]+)_o([^_]+)_([^_]+)_residual$", stem)
    if m:
        grid, order, flux = m.groups()
        return f"{grid} O{order} {flux.upper()}"

    # steady_8k_residual.bin
    m = re.match(r"^steady_([^_]+)_residual$", stem)
    if m:
        return f"{m.group(1)} steady"

    # Generic fallback
    return stem


def resolve_inputs(inputs):
    files = []
    for item in inputs:
        matches = sorted(glob.glob(item))
        if matches:
            files.extend(matches)
        else:
            files.append(item)
    # Keep order but remove duplicates
    seen = set()
    unique = []
    for p in files:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def main():
    parser = argparse.ArgumentParser(
        description="Overlay convergence histories from residual .bin files."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Residual files or glob patterns, e.g. 'data_steady/conv_*.bin'",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="convergence_overlay.png",
        help="Output image path (default: convergence_overlay.png)",
    )
    parser.add_argument(
        "--title",
        default="Convergence History Overlay",
        help="Plot title",
    )
    parser.add_argument(
        "--xmax",
        type=int,
        default=None,
        help="Optional max iteration shown on x-axis",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive plot window",
    )
    args = parser.parse_args()

    files = resolve_inputs(args.inputs)
    if not files:
        print("No files provided.")
        sys.exit(1)

    plt.figure(figsize=(10, 6))
    plotted = 0

    for path in files:
        if not os.path.isfile(path):
            print(f"Skipping (not found): {path}")
            continue
        try:
            y = read_residual(path)
        except Exception as exc:
            print(f"Skipping (read failed): {path} ({exc})")
            continue

        if y.size == 0:
            print(f"Skipping (empty history): {path}")
            continue

        x = np.arange(y.size)
        if args.xmax is not None:
            mask = x <= args.xmax
            x = x[mask]
            y = y[mask]
            if y.size == 0:
                print(f"Skipping (outside xmax): {path}")
                continue

        label = parse_label_from_name(path)
        plt.plot(x, y, linewidth=1.7, label=label)
        plotted += 1

    if plotted == 0:
        print("No valid residual files were plotted.")
        sys.exit(1)

    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Residual")
    plt.title(args.title)
    plt.grid(True, which="both", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved: {args.output}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
