#!/usr/bin/env python3
"""
Overlay plot of force coefficient time histories.

Reads snapshot .bin files from each unsteady_data_<order>_<grid>/ directory,
recomputes wall force coefficients, and overlays all cases on a single figure.

Usage:
    python postproc/plot_force_overlay.py [output_file]
    (default output: force_coefficient_overlay.png)
"""

import numpy as np
import matplotlib.pyplot as plt
import struct
import glob
import os
import sys

# ── Dataset definitions ───────────────────────────────────────────────────────
# (label, data_dir, mesh_file, linestyle, color)
# Use matplotlib's tab10 palette for consistent, visually appealing colors
palette = plt.get_cmap("tab10").colors
DATASETS = [
    ("Order 1, coarse (2k)", "unsteady_data_1_2k", "grids/2k.gri", "-", palette[0], 1.0),
    ("Order 2, coarse (2k)", "unsteady_data_2_2k", "grids/2k.gri", "--", palette[1], 1.0),
    ("Order 1, fine (8k)",   "unsteady_data_1_8k", "grids/8k.gri", "-", palette[2], 1.0),
    ("Order 2, fine (8k)",   "unsteady_data_2_8k", "grids/8k.gri", "--", palette[3], 1.0),
    ("Order 1, fine (32k)",  "unsteady_data_1_32k", "grids/32k.gri", "-", palette[4], 1.0),
    ("Order 2, fine (32k)",  "unsteady_data_2_32k", "grids/32k.gri", "--", palette[5], 1.0),
]

GAMMA = 1.4

# ── I/O helpers ───────────────────────────────────────────────────────────────

def read_snapshot(filepath):
    with open(filepath, 'rb') as f:
        ne = struct.unpack('i', f.read(4))[0]
        data = struct.unpack('d' * 4 * ne, f.read(8 * 4 * ne))
    return np.array(data).reshape(ne, 4)

def time_from_filename(filepath):
    """Extract physical time from filenames like results_123.456789_0042.bin"""
    basename = os.path.basename(filepath)
    try:
        return float(basename.replace('results_', '').split('_')[0])
    except Exception:
        return 0.0

def read_mesh_nodes(gri_file):
    with open(gri_file) as f:
        nn, ne, dim = map(int, f.readline().split())
        verts = np.array([[float(x) for x in f.readline().split()] for _ in range(nn)])
    return verts

def read_mesh_elements(gri_file):
    with open(gri_file) as f:
        nn, ne, dim = map(int, f.readline().split())
        for _ in range(nn):
            f.readline()
        nb = int(f.readline())
        for _ in range(nb):
            line = f.readline().split()
            nb_edges = int(line[0])
            for _ in range(nb_edges):
                f.readline()
        elements, ne_total = [], 0
        while ne_total < ne:
            line = f.readline().split()
            if not line:
                break
            nei = int(line[0])
            for _ in range(nei):
                elements.append([int(s) - 1 for s in f.readline().split()])
            ne_total += nei
    return np.array(elements)

def read_wall_edges(gri_file):
    with open(gri_file) as f:
        nn, ne, dim = map(int, f.readline().split())
        for _ in range(nn):
            f.readline()
        nb = int(f.readline())
        for ib in range(nb):
            line = f.readline().split()
            nb_edges = int(line[0])
            bname = line[2] if len(line) > 2 else f"boundary_{ib}"
            edges = [[int(s) - 1 for s in f.readline().split()] for _ in range(nb_edges)]
            if bname.lower() == 'wall':
                return np.array(edges)
    return None

def build_edge_to_elem(elements):
    m = {}
    for ei, elem in enumerate(elements):
        for a, b in [(0,1),(1,2),(2,0)]:
            key = tuple(sorted([elem[a], elem[b]]))
            m.setdefault(key, []).append(ei)
    return m

def compute_cx_cy(verts, elements, wall_edges, edge_to_elem, U):
    rho  = U[:, 0]
    rhou = U[:, 1]
    rhov = U[:, 2]
    rhoe = U[:, 3]
    p = (GAMMA - 1) * (rhoe - 0.5 * (rhou**2 + rhov**2) / rho)

    # Reference dynamic pressure
    p0    = 1.0 / GAMMA          # a0=1, rho0=1
    p_out = 0.7 * p0
    ratio = p0 / p_out
    M2    = (2.0 / (GAMMA - 1)) * (ratio**((GAMMA-1)/GAMMA) - 1.0)
    q_out = 0.5 * GAMMA * p_out * M2
    chord = 18.804               # mm, same units as mesh

    Fx = Fy = 0.0
    for edge in wall_edges:
        n1, n2 = edge[0], edge[1]
        key = tuple(sorted([n1, n2]))
        if key not in edge_to_elem:
            continue
        ei = edge_to_elem[key][0]
        pe = p[ei]
        ev = verts[n2] - verts[n1]
        length = np.linalg.norm(ev)
        normal = np.array([ev[1], -ev[0]]) / length
        # ensure outward
        centroid  = verts[elements[ei]].mean(axis=0)
        edge_mid  = 0.5 * (verts[n1] + verts[n2])
        if np.dot(normal, edge_mid - centroid) < 0:
            normal = -normal
        Fx += pe * normal[0] * length
        Fy += pe * normal[1] * length

    return Fx / (q_out * chord), Fy / (q_out * chord)

# ── Mesh cache so we don't re-parse the same .gri multiple times ──────────────
_mesh_cache = {}

def get_mesh(gri_file):
    if gri_file not in _mesh_cache:
        verts      = read_mesh_nodes(gri_file)
        elements   = read_mesh_elements(gri_file)
        wall_edges = read_wall_edges(gri_file)
        e2e        = build_edge_to_elem(elements)
        _mesh_cache[gri_file] = (verts, elements, wall_edges, e2e)
    return _mesh_cache[gri_file]

# ── Main ──────────────────────────────────────────────────────────────────────

def process_case(data_dir, gri_file):
    snapshots = sorted(glob.glob(os.path.join(data_dir, "results_*.bin")),
                       key=time_from_filename)
    if not snapshots:
        return None, None, None

    verts, elements, wall_edges, e2e = get_mesh(gri_file)
    if wall_edges is None:
        print(f"  WARNING: no 'wall' boundary found in {gri_file}")
        return None, None, None

    times, cx_list, cy_list = [], [], []
    for snap in snapshots:
        U  = read_snapshot(snap)
        cx, cy = compute_cx_cy(verts, elements, wall_edges, e2e, U)
        times.append(time_from_filename(snap))
        cx_list.append(cx)
        cy_list.append(cy)

    return np.array(times), np.array(cx_list), np.array(cy_list)


def main():
    output_file = sys.argv[1] if len(sys.argv) > 1 else "force_coefficient_overlay.png"

    fig, (ax_cx, ax_cy) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    for label, data_dir, gri_file, ls, color, alpha in DATASETS:
        if not os.path.isdir(data_dir):
            print(f"  SKIP {label}: directory '{data_dir}' not found")
            continue

        print(f"Processing {label} ...")
        t, cx, cy = process_case(data_dir, gri_file)
        if t is None:
            print(f"  SKIP {label}: no snapshot files found in '{data_dir}'")
            continue

        lw = 1.5 if "coarse" in label.lower() else 2.0
        ax_cx.plot(t, cx, linestyle=ls, color=color, linewidth=lw, label=label, alpha=alpha)
        ax_cy.plot(t, cy, linestyle=ls, color=color, linewidth=lw, label=label, alpha=alpha)
        print(f"  cx: mean={cx.mean():.4f} ± {cx.std():.4f}   "
              f"cy: mean={cy.mean():.4f} ± {cy.std():.4f}   n={len(t)}")

    ax_cx.set_ylabel("$c_x$  (axial force coeff.)", fontsize=12)
    ax_cx.set_title("Force Coefficient Time History — Order & Mesh Comparison", fontsize=13)
    ax_cx.legend(loc="best", fontsize=10)
    ax_cx.grid(True, alpha=0.3)

    ax_cy.set_ylabel("$c_y$  (normal force coeff.)", fontsize=12)
    ax_cy.set_xlabel("Time", fontsize=12)
    ax_cy.legend(loc="best", fontsize=10)
    ax_cy.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
