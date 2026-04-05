#!/usr/bin/env python3
"""
Plot entropy snapshots from unsteady results.

Usage:
    python postproc/plot_unsteady.py <meshfile> <results_dir> [output_dir]
    python postproc/plot_unsteady.py <meshfile> <results_dir> --times 0.4 0.8 1.0
    python postproc/plot_unsteady.py <meshfile> <results_dir> --latest 3
"""

import argparse
import glob
import os
import struct

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.collections import PolyCollection

from dg_utils import (
    build_dg_triangulation,
    maybe_read_dg_results,
    primitive_from_state,
    read_gri_mesh,
    element_polygon,
)


def read_results(filename):
    with open(filename, "rb") as f:
        ne = struct.unpack("i", f.read(4))[0]
        data = struct.unpack("d" * 4 * ne, f.read(8 * 4 * ne))
        return np.array(data).reshape(ne, 4)


def extract_time_from_filename(filename):
    basename = os.path.basename(filename)
    try:
        parts = basename.replace("results_", "").replace(".bin", "").split("_")
        return float(parts[0])
    except Exception:
        return 0.0


def select_snapshots(result_files, target_times=None, time_tol=0.05, latest=None):
    all_times = [(extract_time_from_filename(f), f) for f in result_files]
    if not all_times:
        return {}

    if target_times:
        selected = {}
        for target in target_times:
            best_t, best_f = min(all_times, key=lambda tf: abs(tf[0] - target))
            if abs(best_t - target) <= time_tol:
                selected[target] = (best_t, best_f)
            else:
                print(
                    f"  Warning: no snapshot within {time_tol} of t={target} "
                    f"(closest is t={best_t:.6f})"
                )
        return selected

    latest = 3 if latest is None else latest
    chosen = all_times[-latest:]
    return {t: (t, f) for t, f in chosen}


def plot_entropy_snapshots(meshfile, results_dir, output_dir="unsteady_plots",
                           target_times=None, time_tol=0.05, latest=None):
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

    selected = select_snapshots(
        result_files, target_times=target_times, time_tol=time_tol, latest=latest
    )

    if not selected:
        print("No snapshots matched the requested selection.")
        return

    entropy_min, entropy_max = np.inf, -np.inf
    for _, (t, f) in sorted(selected.items()):
        dg_filename, U_dg, p_order, _ = maybe_read_dg_results(f)
        if U_dg is not None:
            _, _, _, entropy = build_dg_triangulation(mesh, U_dg, p_order, "entropy")
            entropy = np.exp(entropy)
        else:
            u = read_results(f)
            entropy = np.exp(primitive_from_state(u)["entropy"])
        entropy_min = min(entropy_min, entropy.min())
        entropy_max = max(entropy_max, entropy.max())

    print(f"p/rho^gamma range (selected snapshots): [{entropy_min:.4f}, {entropy_max:.4f}]")

    n = len(selected)
    fig, axes = plt.subplots(n, 1, figsize=(12, 6 * n))
    if n == 1:
        axes = [axes]

    ordered = sorted(selected.items(), key=lambda kv: kv[1][0])
    for ax, (_, (actual_t, filepath)) in zip(axes, ordered):
        dg_filename, U_dg, p_order, _ = maybe_read_dg_results(filepath)
        if U_dg is not None:
            x, y, tris, entropy = build_dg_triangulation(mesh, U_dg, p_order, "entropy")
            triang = mtri.Triangulation(x, y, tris)
            pc = ax.tripcolor(triang, np.exp(entropy), shading="gouraud", cmap="viridis")
            pc.set_clim(entropy_min, entropy_max)
            fig.colorbar(pc, ax=ax, label=r"$e^s = p/\rho^\gamma$")
        else:
            u = read_results(filepath)
            entropy = np.exp(primitive_from_state(u)["entropy"])
            pc = PolyCollection(verts, cmap="viridis", edgecolors="none")
            pc.set_array(entropy)
            pc.set_clim(entropy_min, entropy_max)
            ax.add_collection(pc)
            ax.autoscale()
            fig.colorbar(pc, ax=ax, label=r"$e^s = p/\rho^\gamma$")

        ax.set_aspect("equal")
        ax.set_title(f"t = {actual_t:.6f}")
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")

    plt.suptitle(r"Entropy: $e^s = p/\rho^\gamma$", fontsize=14)
    plt.tight_layout()

    if target_times:
        tag = "_".join(f"{t:.3f}".replace(".", "p") for t in target_times)
    else:
        tag = f"latest_{len(ordered)}"
    outfile = os.path.join(output_dir, f"entropy_{tag}.png")
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"Saved {outfile}")


def main():
    parser = argparse.ArgumentParser(description="Plot entropy snapshots from unsteady runs.")
    parser.add_argument("meshfile")
    parser.add_argument("results_dir")
    parser.add_argument("output_dir", nargs="?", default="unsteady_plots")
    parser.add_argument("--times", nargs="*", type=float, default=None,
                        help="Target times to plot; picks nearest snapshot within tolerance.")
    parser.add_argument("--tol", type=float, default=0.05,
                        help="Time matching tolerance for --times [default: 0.05]")
    parser.add_argument("--latest", type=int, default=None,
                        help="If --times is omitted, plot the latest N snapshots [default: 3]")
    args = parser.parse_args()

    plot_entropy_snapshots(
        args.meshfile,
        args.results_dir,
        args.output_dir,
        target_times=args.times,
        time_tol=args.tol,
        latest=args.latest,
    )


if __name__ == "__main__":
    main()
