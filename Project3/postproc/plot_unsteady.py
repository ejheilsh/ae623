#!/usr/bin/env python3
"""
Plot entropy field at t ≈ 100, 200, 300 from unsteady snapshot bins.

Usage:
    python postproc/plot_unsteady.py <meshfile> <results_dir> [output_dir]

Example:
    python postproc/plot_unsteady.py grids/32k.gri unsteady_data_2_32k unsteady_plots/
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import struct
import sys
import glob
import os

TARGET_TIMES = [100.0, 200.0, 300.0]
TIME_TOL     = 5.0   # pick the snapshot whose time is closest, within this window

def read_results(filename):
    with open(filename, 'rb') as f:
        ne = struct.unpack('i', f.read(4))[0]
        data = struct.unpack('d' * 4 * ne, f.read(8 * 4 * ne))
        return np.array(data).reshape(ne, 4)

def read_mesh(filename):
    with open(filename, 'r') as f:
        nn, ne, dim = map(int, f.readline().split())
        v = np.array([[float(s) for s in f.readline().split()] for _ in range(nn)])
        return v

def get_elements_from_gri(filename):
    with open(filename, 'r') as f:
        nn, ne, dim = map(int, f.readline().split())
        for _ in range(nn):
            f.readline()
        nb = int(f.readline())
        for _ in range(nb):
            line = f.readline().split()
            nb_in = int(line[0])
            for _ in range(nb_in):
                f.readline()
        elements = []
        ne_total = 0
        while ne_total < ne:
            line = f.readline().split()
            if not line:
                break
            nei = int(line[0])
            for _ in range(nei):
                elements.append([int(s)-1 for s in f.readline().split()])
            ne_total += nei
        return np.array(elements)

def extract_time_from_filename(filename):
    basename = os.path.basename(filename)
    try:
        parts = basename.replace('results_', '').replace('.bin', '').split('_')
        return float(parts[0])
    except:
        return 0.0

def plot_entropy_snapshots(meshfile, results_dir, output_dir='unsteady_plots'):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading mesh: {meshfile}")
    v = read_mesh(meshfile)
    e = get_elements_from_gri(meshfile)
    verts = v[e]

    pattern = os.path.join(results_dir, "results_*.bin")
    result_files = sorted(glob.glob(pattern), key=extract_time_from_filename)

    if not result_files:
        print(f"Error: No files found matching {pattern}")
        return

    print(f"Found {len(result_files)} snapshot files")

    # Build (time, filepath) list
    all_times = [(extract_time_from_filename(f), f) for f in result_files]

    # Select the closest snapshot to each target time
    selected = {}
    for target in TARGET_TIMES:
        best_t, best_f = min(all_times, key=lambda tf: abs(tf[0] - target))
        if abs(best_t - target) <= TIME_TOL:
            selected[target] = (best_t, best_f)
        else:
            print(f"  Warning: no snapshot within {TIME_TOL} of t={target} "
                  f"(closest is t={best_t:.2f})")

    if not selected:
        print("No snapshots matched the target times.")
        return

    # Compute entropy range across only the selected snapshots for a consistent colorbar
    gamma = 1.4
    entropy_min, entropy_max = np.inf, -np.inf
    for target, (t, f) in selected.items():
        u = read_results(f)
        rho = u[:, 0]; rhou = u[:, 1]; rhov = u[:, 2]; rhoe = u[:, 3]
        vel_u = rhou / rho; vel_v = rhov / rho
        p = (gamma - 1) * (rhoe - 0.5 * rho * (vel_u**2 + vel_v**2))
        entropy = p / rho**gamma        # isentropic function p/ρ^γ  (~0.7 in freestream)
        entropy_min = min(entropy_min, entropy.min())
        entropy_max = max(entropy_max, entropy.max())
    print(f"p/rho^gamma range (selected snapshots): [{entropy_min:.4f}, {entropy_max:.4f}]")

    # One figure with three subplots stacked vertically
    n = len(selected)
    fig, axes = plt.subplots(n, 1, figsize=(12, 6 * n))
    if n == 1:
        axes = [axes]

    for ax, target in zip(axes, sorted(selected)):
        actual_t, filepath = selected[target]
        u = read_results(filepath)
        rho = u[:, 0]; rhou = u[:, 1]; rhov = u[:, 2]; rhoe = u[:, 3]
        vel_u = rhou / rho; vel_v = rhov / rho
        vel_mag = np.sqrt(vel_u**2 + vel_v**2)
        p = (gamma - 1) * (rhoe - 0.5 * rho * vel_mag**2)
        entropy = p / rho**gamma        # isentropic function p/ρ^γ

        pc = PolyCollection(verts, cmap='viridis', edgecolors='none')
        pc.set_array(entropy)
        pc.set_clim(entropy_min, entropy_max)
        ax.add_collection(pc)
        ax.autoscale()
        ax.set_aspect('equal')
        fig.colorbar(pc, ax=ax, label=r'$e^s = p/\rho^\gamma$')
        ax.set_title(f't = {actual_t:.1f}')
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')

    plt.suptitle(r'Entropy: $e^s = p/\rho^\gamma$', fontsize=14)
    plt.tight_layout()
    outfile = os.path.join(output_dir, 'entropy_t100_200_300.png')
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"Saved {outfile}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    meshfile   = sys.argv[1]
    results_dir = sys.argv[2]
    output_dir  = sys.argv[3] if len(sys.argv) > 3 else 'unsteady_plots'

    plot_entropy_snapshots(meshfile, results_dir, output_dir)
