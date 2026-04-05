#!/usr/bin/env python3
"""
Overlay plot of force coefficient time histories for arbitrary unsteady cases.

Usage:
    python postproc/plot_force_overlay.py output.png \
      "p0-2k:unsteady_data/p0_2k:grids/2k_q2.gri" \
      "p1-2k:unsteady_data/p1_2k:grids/2k_q2.gri"

Each case spec is:
    label:results_dir:mesh_file
"""

import argparse
import glob
import os
import struct

import matplotlib.pyplot as plt
import numpy as np

from dg_utils import (
    integrate_wall_forces,
    maybe_read_dg_results,
    read_cell_results,
    read_gri_mesh,
)


def read_snapshot(filepath):
    with open(filepath, "rb") as f:
        ne = struct.unpack("i", f.read(4))[0]
        data = struct.unpack("d" * 4 * ne, f.read(8 * 4 * ne))
    return np.array(data).reshape(ne, 4)


def time_from_filename(filepath):
    basename = os.path.basename(filepath)
    try:
        return float(basename.replace("results_", "").split("_")[0])
    except Exception:
        return 0.0


def parse_case(spec):
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise ValueError(
            f"Invalid case spec '{spec}'. Expected format label:results_dir:mesh_file"
        )
    return parts[0], parts[1], parts[2]


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
    M2 = (2.0 / (1.4 - 1.0)) * (ratio ** ((1.4 - 1.0) / 1.4) - 1.0)
    q_out = 0.5 * 1.4 * p_out * M2
    chord = 18.804

    Fx = 0.0
    Fy = 0.0
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
        ev = nodes[n2] - nodes[n1]
        length = np.linalg.norm(ev)
        normal = np.array([ev[1], -ev[0]]) / length
        centroid = nodes[mesh["elements"][elem_idx]["corners"]].mean(axis=0)
        edge_mid = 0.5 * (nodes[n1] + nodes[n2])
        if np.dot(normal, edge_mid - centroid) < 0:
            normal = -normal
        Fx += p[elem_idx] * normal[0] * length
        Fy += p[elem_idx] * normal[1] * length

    return Fx / (q_out * chord), Fy / (q_out * chord)


def process_case(results_dir, mesh_file):
    snapshots = sorted(
        [f for f in glob.glob(os.path.join(results_dir, "results_*.bin")) if not f.endswith("_dg.bin")],
        key=time_from_filename,
    )
    if not snapshots:
        return None, None, None

    mesh = read_gri_mesh(mesh_file)
    times, cx_list, cy_list = [], [], []
    for snap in snapshots:
        dg_filename, U_dg, p_order, _ = maybe_read_dg_results(snap)
        if U_dg is not None:
            cx, cy = integrate_wall_forces(mesh, U_dg, p_order)
        else:
            U = read_cell_results(snap)
            cx, cy = primitive_force_from_cell_average(mesh, U)
        times.append(time_from_filename(snap))
        cx_list.append(cx)
        cy_list.append(cy)

    return np.array(times), np.array(cx_list), np.array(cy_list)


def main():
    parser = argparse.ArgumentParser(description="Overlay force coefficient histories from unsteady runs.")
    parser.add_argument("output_file")
    parser.add_argument("cases", nargs="+", help="Case specs: label:results_dir:mesh_file")
    args = parser.parse_args()

    palette = plt.get_cmap("tab10").colors
    fig, (ax_cx, ax_cy) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    for idx, case_spec in enumerate(args.cases):
        label, data_dir, mesh_file = parse_case(case_spec)
        if not os.path.isdir(data_dir):
            print(f"SKIP {label}: directory '{data_dir}' not found")
            continue

        print(f"Processing {label} ...")
        t, cx, cy = process_case(data_dir, mesh_file)
        if t is None:
            print(f"SKIP {label}: no snapshot files found in '{data_dir}'")
            continue

        color = palette[idx % len(palette)]
        ax_cx.plot(t, cx, linewidth=2.0, color=color, label=label)
        ax_cy.plot(t, cy, linewidth=2.0, color=color, label=label)
        print(
            f"  cx: mean={cx.mean():.4f} ± {cx.std():.4f}   "
            f"cy: mean={cy.mean():.4f} ± {cy.std():.4f}   n={len(t)}"
        )

    ax_cx.set_ylabel("$c_x$", fontsize=12)
    ax_cx.set_title("Force Coefficient Time History", fontsize=13)
    ax_cx.legend(loc="best", fontsize=10)
    ax_cx.grid(True, alpha=0.3)

    ax_cy.set_ylabel("$c_y$", fontsize=12)
    ax_cy.set_xlabel("Time", fontsize=12)
    ax_cy.legend(loc="best", fontsize=10)
    ax_cy.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output_file, dpi=150)
    print(f"\nSaved: {args.output_file}")


if __name__ == "__main__":
    main()
