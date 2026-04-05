#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./plot_forces_vs_cells.sh [input_glob] [output_prefix]
# Example:
#   ./plot_forces_vs_cells.sh "output_final/state_*.bin" "output_final/forces_vs_cells"
# Produces:
#   output_final/forces_vs_cells_cx.png
#   output_final/forces_vs_cells_cy.png

INPUT_GLOB="${1:-output_final/state_*.bin}"
OUTPUT_PREFIX="${2:-output_final/forces_vs_cells}"

export MPLBACKEND="${MPLBACKEND:-Agg}"

python3 - "$INPUT_GLOB" "$OUTPUT_PREFIX" <<'PY'
import glob
import os
import re
import struct
import sys

import matplotlib.pyplot as plt
import numpy as np


def read_results(path):
    with open(path, "rb") as f:
        ne = struct.unpack("i", f.read(4))[0]
        data = struct.unpack("d" * (4 * ne), f.read(8 * 4 * ne))
    return ne, np.array(data).reshape(ne, 4)


def read_mesh_vertices_and_elements(meshfile):
    with open(meshfile, "r") as f:
        nn, ne, dim = map(int, f.readline().split())
        v = np.array([[float(s) for s in f.readline().split()] for _ in range(nn)])
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
                elements.append([int(s) - 1 for s in f.readline().split()])
            ne_total += nei
    return v, np.array(elements)


def read_boundary_groups(meshfile):
    with open(meshfile, "r") as f:
        nn, ne, dim = map(int, f.readline().split())
        for _ in range(nn):
            f.readline()
        nb = int(f.readline())
        groups = {}
        for ib in range(nb):
            line = f.readline().split()
            n_edges = int(line[0])
            bname = line[2] if len(line) > 2 else f"boundary_{ib}"
            edges = []
            for _ in range(n_edges):
                n1, n2 = map(int, f.readline().split())
                edges.append([n1 - 1, n2 - 1])
            groups[bname] = np.array(edges)
    return groups


def compute_force_coefficients(v, elements, boundary_groups, u, gamma=1.4):
    wall_edges = None
    for bname, edges in boundary_groups.items():
        if bname.lower() == "wall":
            wall_edges = edges
            break
    if wall_edges is None:
        return 0.0, 0.0

    elem_to_edge = {}
    for elem_idx, elem in enumerate(elements):
        edge_list = [
            tuple(sorted((elem[0], elem[1]))),
            tuple(sorted((elem[1], elem[2]))),
            tuple(sorted((elem[2], elem[0]))),
        ]
        for edge in edge_list:
            elem_to_edge.setdefault(edge, []).append(elem_idx)

    rho = u[:, 0]
    rhou = u[:, 1]
    rhov = u[:, 2]
    rhoe = u[:, 3]
    vel_u = rhou / rho
    vel_v = rhov / rho
    qsq = vel_u ** 2 + vel_v ** 2
    p = (gamma - 1.0) * (rhoe - 0.5 * rho * qsq)

    rho0 = 1.0
    a0 = 1.0
    p0 = a0 * a0 * rho0 / gamma
    p_out = 0.7 * p0
    pressure_ratio = p0 / p_out
    m_out_sq = (2.0 / (gamma - 1.0)) * (pressure_ratio ** ((gamma - 1.0) / gamma) - 1.0)
    q_out = 0.5 * gamma * p_out * m_out_sq
    chord = 18.804

    fx = 0.0
    fy = 0.0
    for edge in wall_edges:
        n1, n2 = edge[0], edge[1]
        edge_key = tuple(sorted((n1, n2)))
        if edge_key not in elem_to_edge:
            continue
        elem_idx = elem_to_edge[edge_key][0]
        p_elem = p[elem_idx]

        edge_vec = v[n2] - v[n1]
        edge_length = np.linalg.norm(edge_vec)
        normal = np.array([edge_vec[1], -edge_vec[0]])
        normal /= np.linalg.norm(normal)

        centroid = v[elements[elem_idx]].mean(axis=0)
        edge_mid = 0.5 * (v[n1] + v[n2])
        to_edge = edge_mid - centroid
        if np.dot(normal, to_edge) < 0:
            normal = -normal

        fx += p_elem * normal[0] * edge_length
        fy += p_elem * normal[1] * edge_length

    cx = fx / (q_out * chord)
    cy = fy / (q_out * chord)
    return cx, cy


def grid_name_from_state_file(path):
    name = os.path.basename(path)
    m = re.match(r"^state_gri(.+?)_ord", name)
    if m:
        return m.group(1)
    return None


def label_from_state_file(path):
    name = os.path.basename(path)
    stem = name[:-4] if name.endswith(".bin") else name
    m = re.match(r"^state_gri(.+?)_ord([^_]+)_([^_]+)_cfl([^_]+)$", stem)
    if m:
        grid, order, flux, cfl = m.groups()
        return f"{grid} O{order} {flux.upper()} CFL={cfl}"
    return stem


def main():
    input_glob = sys.argv[1]
    output_prefix = sys.argv[2]

    files = sorted(glob.glob(input_glob))
    if not files:
        print(f"No files matched: {input_glob}")
        sys.exit(1)

    rows = []
    for fpath in files:
        grid = grid_name_from_state_file(fpath)
        if grid is None:
            print(f"Skipping (unrecognized name): {fpath}")
            continue
        meshfile = os.path.join("grids", f"{grid}.gri")
        if not os.path.isfile(meshfile):
            print(f"Skipping (mesh not found): {meshfile} for {fpath}")
            continue

        ne_file, u = read_results(fpath)
        v, elements = read_mesh_vertices_and_elements(meshfile)
        if ne_file != len(elements):
            print(f"Skipping (cell mismatch): {fpath} (state={ne_file}, mesh={len(elements)})")
            continue
        boundary_groups = read_boundary_groups(meshfile)
        cx, cy = compute_force_coefficients(v, elements, boundary_groups, u)
        label = label_from_state_file(fpath)
        rows.append((ne_file, cx, cy, label, os.path.basename(fpath)))

    if not rows:
        print("No valid datasets to plot.")
        sys.exit(1)

    rows.sort(key=lambda r: r[0])
    ne = np.array([r[0] for r in rows], dtype=float)
    cx = np.array([r[1] for r in rows], dtype=float)
    cy = np.array([r[2] for r in rows], dtype=float)
    labels = [r[3] for r in rows]

    cx_png = f"{output_prefix}_cx.png"
    cy_png = f"{output_prefix}_cy.png"

    def make_scatter(y, ylabel, title, marker, out_png):
        plt.figure(figsize=(9, 6))
        plt.scatter(ne, y, s=80, marker=marker)
        for x_i, y_i, lbl in zip(ne, y, labels):
            plt.annotate(lbl, (x_i, y_i), textcoords="offset points", xytext=(6, 6), fontsize=9)
        plt.xscale("log")
        plt.xlabel("Number of Cells")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_png, dpi=160)
        print(f"Saved {out_png}")

    make_scatter(cx, "c_x", "c_x vs Number of Cells", "o", cx_png)
    make_scatter(cy, "c_y", "c_y vs Number of Cells", "s", cy_png)

    print("\nData:")
    for n, cxi, cyi, label, name in rows:
        print(f"{name:40s}  {label:24s}  Ne={n:7d}  c_x={cxi:+.6e}  c_y={cyi:+.6e}")


if __name__ == "__main__":
    main()
PY
