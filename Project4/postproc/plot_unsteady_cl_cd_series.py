#!/usr/bin/env python3
"""
Generate unsteady Cl/Cd time-history plots for q=3 cases on 2k and 8k meshes.

The script scans unsteady_data for:
  - 2k_q3_p0 .. 2k_q3_p3
  - 8k_q3_p0 .. 8k_q3_p3

It then plots four separate figures:
  - Cl on 2k
  - Cd on 2k
  - Cl on 8k
  - Cd on 8k

Only cases that actually exist are included, so 8k_q3_p3 will appear
automatically once its directory/snapshots are present.
"""

from __future__ import annotations

import argparse
import glob
import os
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dg_utils import integrate_wall_forces, maybe_read_dg_results, read_cell_results, read_gri_mesh


DEFAULT_TMIN = 240.0
DEFAULT_TMAX = 300.0
GRID_TAGS = ("2k_q3", "8k_q3")
P_ORDERS = (0, 1, 2, 3)


def time_from_filename(filepath: str) -> float:
    basename = os.path.basename(filepath)
    return float(basename.replace("results_", "").split("_")[0])


def primitive_force_from_cell_average(mesh, U):
    nodes = mesh["nodes"]
    edge_to_elem = {}
    for elem_idx, element in enumerate(mesh["elements"]):
        c = element["corners"]
        for edge in [(c[0], c[1]), (c[1], c[2]), (c[2], c[0])]:
            edge_to_elem[tuple(sorted(edge))] = elem_idx

    rho = U[:, 0]
    rhou = U[:, 1]
    rhov = U[:, 2]
    rhoe = U[:, 3]
    p = (1.4 - 1.0) * (rhoe - 0.5 * (rhou * rhou + rhov * rhov) / rho)

    p0 = 1.0 / 1.4
    p_out = 0.7 * p0
    ratio = p0 / p_out
    m2 = (2.0 / (1.4 - 1.0)) * (ratio ** ((1.4 - 1.0) / 1.4) - 1.0)
    q_out = 0.5 * 1.4 * p_out * m2
    chord = 18.804

    fx = 0.0
    fy = 0.0
    wall_edges = None
    for name, edges in mesh["boundary_groups"].items():
        if name.lower() == "wall":
            wall_edges = edges
            break
    if wall_edges is None:
        return 0.0, 0.0

    for edge in wall_edges:
        key = tuple(sorted(edge))
        elem_idx = edge_to_elem.get(key)
        if elem_idx is None:
            continue
        n1, n2 = edge
        edge_vec = nodes[n2] - nodes[n1]
        length = np.linalg.norm(edge_vec)
        normal = np.array([edge_vec[1], -edge_vec[0]]) / length
        centroid = nodes[mesh["elements"][elem_idx]["corners"]].mean(axis=0)
        edge_mid = 0.5 * (nodes[n1] + nodes[n2])
        if np.dot(normal, edge_mid - centroid) < 0:
            normal = -normal
        fx += p[elem_idx] * normal[0] * length
        fy += p[elem_idx] * normal[1] * length

    return fx / (q_out * chord), fy / (q_out * chord)


def process_case(
    results_dir: Path, mesh_file: Path, tmin: float, tmax: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    snapshots = []
    for filepath in glob.glob(str(results_dir / "results_*.bin")):
        if filepath.endswith("_dg.bin"):
            continue
        t = time_from_filename(filepath)
        if tmin <= t <= tmax:
            snapshots.append(filepath)
    snapshots.sort(key=time_from_filename)
    if not snapshots:
        return None

    mesh = read_gri_mesh(str(mesh_file))
    times, cx_list, cy_list = [], [], []
    for snap in snapshots:
        dg_filename, u_dg, p_order, _ = maybe_read_dg_results(snap)
        if u_dg is not None:
            cx, cy = integrate_wall_forces(mesh, u_dg, p_order)
        else:
            u = read_cell_results(snap)
            cx, cy = primitive_force_from_cell_average(mesh, u)
        times.append(time_from_filename(snap))
        cx_list.append(cx)
        cy_list.append(cy)

    times = np.asarray(times)
    cx = np.asarray(cx_list)
    cy = np.asarray(cy_list)
    cl = cy
    cd = cx
    return times, cl, cd


def discover_cases(base_dir: Path) -> dict[str, dict[int, Path]]:
    cases: dict[str, dict[int, Path]] = {grid: {} for grid in GRID_TAGS}
    for grid in GRID_TAGS:
        for p in P_ORDERS:
            case_dir = base_dir / f"{grid}_p{p}"
            if case_dir.is_dir():
                cases[grid][p] = case_dir
    return cases


def style_axes(ax, ylabel: str):
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.tick_params(direction="out")


def plot_quantity(grid: str, quantity_name: str, series_by_p, output_path: Path, y_limits=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["tab:blue", "tab:red", "tab:green", "black"]
    plotted_any = False

    for p in P_ORDERS:
        data = series_by_p.get(p)
        if data is None:
            continue
        t, q = data
        ax.plot(t, q, linewidth=2.0, color=colors[p], label=rf"$p={p}$")
        plotted_any = True

    if not plotted_any:
        plt.close(fig)
        return

    style_axes(ax, rf"$C_{quantity_name.lower()}$")
    if y_limits is not None:
        ax.set_ylim(y_limits[0], y_limits[1])
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot unsteady Cl/Cd histories for q=3, 2k/8k cases.")
    parser.add_argument("--base-dir", default="unsteady_data")
    parser.add_argument("--outdir", default="postproc_out/task5")
    parser.add_argument("--tmin", type=float, default=DEFAULT_TMIN)
    parser.add_argument("--tmax", type=float, default=DEFAULT_TMAX)
    parser.add_argument("--cl-ylim", nargs=2, type=float, metavar=("YMIN", "YMAX"))
    parser.add_argument("--cd-ylim", nargs=2, type=float, metavar=("YMIN", "YMAX"))
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "font.size": 15,
            "axes.labelsize": 19,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
        }
    )

    base_dir = Path(args.base_dir)
    outdir = Path(args.outdir)
    cases = discover_cases(base_dir)

    for grid in GRID_TAGS:
        mesh_file = Path("grids") / f"{grid}.gri"
        cl_series = {}
        cd_series = {}
        for p, case_dir in cases[grid].items():
            processed = process_case(case_dir, mesh_file, args.tmin, args.tmax)
            if processed is None:
                continue
            times, cl, cd = processed
            cl_series[p] = (times, cl)
            cd_series[p] = (times, cd)
            print(f"{grid} p={p}: loaded {len(times)} snapshots from {case_dir}")

        plot_quantity(grid, "l", cl_series, outdir / f"cl_{grid}.png", y_limits=args.cl_ylim)
        plot_quantity(grid, "d", cd_series, outdir / f"cd_{grid}.png", y_limits=args.cd_ylim)

        if cl_series:
            print(f"Saved {outdir / f'cl_{grid}.png'}")
        if cd_series:
            print(f"Saved {outdir / f'cd_{grid}.png'}")


if __name__ == "__main__":
    main()
