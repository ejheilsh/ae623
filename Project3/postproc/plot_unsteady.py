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
import matplotlib.tri as mtri
from matplotlib.collections import PolyCollection
import struct
import sys
import glob
import os

from dg_utils import build_dg_triangulation, infer_dg_filename, maybe_read_dg_results, primitive_from_state, read_gri_mesh, element_polygon

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

def get_plot_polygons_from_gri(filename, vertices):
    """Read element connectivity and return polygon vertices for plotting.

    Supports mixed-order TriLagrange blocks. Quadratic triangles are drawn with
    midside nodes so the wall-adjacent curved cells appear in the plot.
    """
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
        polygons = []
        ne_total = 0
        while ne_total < ne:
            line = f.readline().split()
            if not line:
                break
            nei = int(line[0])
            deg = int(line[1])
            for _ in range(nei):
                row = [int(s)-1 for s in f.readline().split()]
                if len(row) < 3:
                    raise ValueError(f"Invalid element row in {filename}: {row}")
                if deg == 1:
                    polygons.append(vertices[row[:3]])
                elif deg == 2 and len(row) >= 6:
                    poly_ids = [row[0], row[1], row[2], row[4], row[5], row[3]]
                    polygons.append(vertices[poly_ids])
                else:
                    polygons.append(vertices[row[:3]])
            ne_total += nei
        return polygons

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
    mesh = read_gri_mesh(meshfile)
    verts = [element_polygon(elem, mesh["nodes"]) for elem in mesh["elements"]]

    pattern = os.path.join(results_dir, "results_*.bin")
    result_files = sorted(
        [f for f in glob.glob(pattern) if not f.endswith("_dg.bin")],
        key=extract_time_from_filename,
    )

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
    entropy_min, entropy_max = np.inf, -np.inf
    for target, (t, f) in selected.items():
        dg_filename, U_dg, p_order, _ = maybe_read_dg_results(f)
        if U_dg is not None:
            _, _, _, entropy = build_dg_triangulation(mesh, U_dg, p_order, "entropy")
        else:
            u = read_results(f)
            entropy = np.exp(primitive_from_state(u)["entropy"])
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
        dg_filename, U_dg, p_order, _ = maybe_read_dg_results(filepath)
        if U_dg is not None:
            x, y, tris, entropy = build_dg_triangulation(mesh, U_dg, p_order, "entropy")
            triang = mtri.Triangulation(x, y, tris)
            pc = ax.tripcolor(triang, np.exp(entropy), shading='gouraud', cmap='viridis')
            pc.set_clim(entropy_min, entropy_max)
            fig.colorbar(pc, ax=ax, label=r'$e^s = p/\rho^\gamma$')
        else:
            u = read_results(filepath)
            entropy = np.exp(primitive_from_state(u)["entropy"])
            pc = PolyCollection(verts, cmap='viridis', edgecolors='none')
            pc.set_array(entropy)
            pc.set_clim(entropy_min, entropy_max)
            ax.add_collection(pc)
            ax.autoscale()
            fig.colorbar(pc, ax=ax, label=r'$e^s = p/\rho^\gamma$')

        ax.set_aspect('equal')
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
