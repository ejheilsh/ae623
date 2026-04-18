#!/usr/bin/env python3
"""
Render Mach contours for adjoint-adaptation cycles using the saved companion
meshes and primal DG snapshots.

Usage:
    python postproc/plot_adaptation_mach.py <data_dir> [output]

Examples:
    python postproc/plot_adaptation_mach.py data_steady coarse_amr_mach.gif
    python postproc/plot_adaptation_mach.py data_steady cycle2_mach.png --cycle 2
"""

import argparse
import io
import re
import struct
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from PIL import Image


GAMMA = 1.4


def read_companion_mesh(filename):
    with open(filename, "rb") as f:
        nn = struct.unpack("i", f.read(4))[0]
        nodes = np.frombuffer(f.read(8 * 2 * nn), dtype=np.float64).reshape(nn, 2)
        ne = struct.unpack("i", f.read(4))[0]
        elements = []
        q_orders = []
        for _ in range(ne):
            q, v0, v1, v2 = struct.unpack("iiii", f.read(16))
            q_orders.append(q)
            elements.append([v0, v1, v2])
    return nodes, elements, np.asarray(q_orders, dtype=int)


def read_dg_results(filename):
    with open(filename, "rb") as f:
        ne = struct.unpack("i", f.read(4))[0]
        p_order = struct.unpack("i", f.read(4))[0]
        ndof = struct.unpack("i", f.read(4))[0]
        data = struct.unpack("d" * 4 * ne * ndof, f.read(8 * 4 * ne * ndof))
    return np.array(data).reshape(ne, ndof, 4), p_order, ndof


def evaluate_basis(xi, eta, p):
    if p == 0:
        return np.array([1.0])
    if p == 1:
        return np.array([1.0 - xi - eta, xi, eta])
    if p == 2:
        return np.array(
            [
                1.0 - 3.0 * xi - 3.0 * eta + 2.0 * xi * xi + 4.0 * xi * eta + 2.0 * eta * eta,
                -xi + 2.0 * xi * xi,
                -eta + 2.0 * eta * eta,
                4.0 * xi * eta,
                4.0 * eta - 4.0 * xi * eta - 4.0 * eta * eta,
                4.0 * xi - 4.0 * xi * xi - 4.0 * xi * eta,
            ]
        )
    if p == 3:
        return np.array(
            [
                1.0
                - 11.0 / 2.0 * xi
                - 11.0 / 2.0 * eta
                + 9.0 * xi * xi
                + 18.0 * xi * eta
                + 9.0 * eta * eta
                - 9.0 / 2.0 * xi * xi * xi
                - 27.0 / 2.0 * xi * xi * eta
                - 27.0 / 2.0 * xi * eta * eta
                - 9.0 / 2.0 * eta * eta * eta,
                xi - 9.0 / 2.0 * xi * xi + 9.0 / 2.0 * xi * xi * xi,
                eta - 9.0 / 2.0 * eta * eta + 9.0 / 2.0 * eta * eta * eta,
                -9.0 / 2.0 * xi * eta + 27.0 / 2.0 * xi * xi * eta,
                -9.0 / 2.0 * xi * eta + 27.0 / 2.0 * xi * eta * eta,
                -9.0 / 2.0 * eta
                + 9.0 / 2.0 * xi * eta
                + 18.0 * eta * eta
                - 27.0 / 2.0 * xi * eta * eta
                - 27.0 / 2.0 * eta * eta * eta,
                9.0 * eta
                - 45.0 / 2.0 * xi * eta
                - 45.0 / 2.0 * eta * eta
                + 27.0 / 2.0 * xi * xi * eta
                + 27.0 * xi * eta * eta
                + 27.0 / 2.0 * eta * eta * eta,
                9.0 * xi
                - 45.0 / 2.0 * xi * xi
                - 45.0 / 2.0 * xi * eta
                + 27.0 / 2.0 * xi * xi * xi
                + 27.0 * xi * xi * eta
                + 27.0 / 2.0 * xi * eta * eta,
                -9.0 / 2.0 * xi
                + 18.0 * xi * xi
                + 9.0 / 2.0 * xi * eta
                - 27.0 / 2.0 * xi * xi * xi
                - 27.0 / 2.0 * xi * xi * eta,
                27.0 * xi * eta - 27.0 * xi * xi * eta - 27.0 * xi * eta * eta,
            ]
        )
    raise ValueError(f"Basis not implemented for p={p}")


def reconstruct_state(U_elem, p_order, xi, eta):
    phi = evaluate_basis(xi, eta, p_order)
    return phi @ U_elem


def primitive_from_state(U):
    rho = U[..., 0]
    rhou = U[..., 1]
    rhov = U[..., 2]
    rhoe = U[..., 3]
    vel_u = rhou / rho
    vel_v = rhov / rho
    qsq = vel_u * vel_u + vel_v * vel_v
    p = (GAMMA - 1.0) * (rhoe - 0.5 * rho * qsq)
    a = np.sqrt(np.maximum(GAMMA * p / rho, 1e-16))
    return np.sqrt(np.maximum(qsq, 0.0)) / a


def build_reference_submesh(level):
    coords = []
    index = {}
    for i in range(level + 1):
      for j in range(level + 1 - i):
        index[(i, j)] = len(coords)
        coords.append((i / level, j / level))

    tris = []
    for i in range(level):
      for j in range(level - i):
        a = index[(i, j)]
        b = index[(i + 1, j)]
        c = index[(i, j + 1)]
        tris.append((a, b, c))
        if j < level - i - 1:
          d = index[(i + 1, j + 1)]
          tris.append((b, d, c))

    return np.array(coords), np.array(tris, dtype=int)


def build_mach_triangulation(nodes, elements, U_dg, p_order, refine_level=None):
    if refine_level is None:
        refine_level = max(4, 2 * p_order + 2)

    ref_xy, ref_tris = build_reference_submesh(refine_level)
    x_all = []
    y_all = []
    mach_all = []
    tri_all = []
    node_offset = 0

    for elem_idx, element in enumerate(elements):
        corners = nodes[element]
        U_elem = U_dg[elem_idx]

        elem_xy = []
        elem_mach = []
        for xi, eta in ref_xy:
            xy = (
                (1.0 - xi - eta) * corners[0]
                + xi * corners[1]
                + eta * corners[2]
            )
            state = reconstruct_state(U_elem, p_order, xi, eta)
            elem_xy.append(xy)
            elem_mach.append(primitive_from_state(state))

        elem_xy = np.array(elem_xy)
        elem_mach = np.array(elem_mach)

        x_all.extend(elem_xy[:, 0].tolist())
        y_all.extend(elem_xy[:, 1].tolist())
        mach_all.extend(elem_mach.tolist())
        tri_all.extend((ref_tris + node_offset).tolist())
        node_offset += len(ref_xy)

    return (
        np.array(x_all),
        np.array(y_all),
        np.array(tri_all, dtype=int),
        np.array(mach_all),
    )


def discover_cycles(data_dir):
    data_dir = Path(data_dir)
    mesh_files = {}
    primal_files = {}

    for path in data_dir.glob("*adjoint_mesh_cycle*.bin"):
        match = re.search(r"cycle(\d+)\.bin$", path.name)
        if match:
            mesh_files[int(match.group(1))] = path

    for path in data_dir.glob("*adjoint_primal_cycle*_dg.bin"):
        match = re.search(r"cycle(\d+)_dg\.bin$", path.name)
        if match:
            primal_files[int(match.group(1))] = path

    cycles = sorted(set(mesh_files) & set(primal_files))
    if not cycles:
        raise SystemExit(
            f"No matching AMR primal/mesh cycle files found in {data_dir}. "
            "Rerun AMR after rebuilding so adjoint_primal_cycle*_dg.bin files are written."
        )
    return cycles, mesh_files, primal_files


def render_cycle(cycle, mesh_file, primal_file, dpi, figscale, cmap, vmin, vmax):
    nodes, elements, _ = read_companion_mesh(mesh_file)
    U_dg, p_order, _ = read_dg_results(primal_file)
    x, y, tris, mach = build_mach_triangulation(nodes, elements, U_dg, p_order)

    fig, ax = plt.subplots(figsize=(10 * figscale, 5 * figscale))
    triang = mtri.Triangulation(x, y, tris)
    artist = ax.tripcolor(triang, mach, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.set_title(f"Mach contour: adaptation cycle {cycle} | p={p_order} | {len(elements)} elements")
    fig.colorbar(artist, ax=ax, label="Mach Number")
    fig.tight_layout()

    return fig


def fig_to_image(fig, dpi):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    parser = argparse.ArgumentParser(description="Render AMR-cycle Mach contours from saved primal DG snapshots.")
    parser.add_argument("data_dir", help="Directory containing adjoint_mesh_cycle and adjoint_primal_cycle files")
    parser.add_argument("output", nargs="?", default="adaptation_mach.gif")
    parser.add_argument("--cycle", type=int, default=None, help="Render only one cycle to a PNG")
    parser.add_argument("--duration", type=int, default=500, help="GIF frame duration in ms [default: 500]")
    parser.add_argument("--figscale", type=float, default=1.2, help="Figure size scale factor [default: 1.2]")
    parser.add_argument("--dpi", type=int, default=160, help="Rendered frame DPI [default: 160]")
    parser.add_argument("--cmap", default="turbo", help="Colormap [default: turbo]")
    parser.add_argument("--vmin", type=float, default=None, help="Fixed min Mach for all frames")
    parser.add_argument("--vmax", type=float, default=None, help="Fixed max Mach for all frames")
    args = parser.parse_args()

    cycles, mesh_files, primal_files = discover_cycles(args.data_dir)

    if args.cycle is not None:
        if args.cycle not in mesh_files or args.cycle not in primal_files:
            raise SystemExit(f"Cycle {args.cycle} not found in {args.data_dir}")
        fig = render_cycle(
            args.cycle,
            mesh_files[args.cycle],
            primal_files[args.cycle],
            args.dpi,
            args.figscale,
            args.cmap,
            args.vmin,
            args.vmax,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"Saved {output}")
        return

    frames = []
    for cycle in cycles:
        fig = render_cycle(
            cycle,
            mesh_files[cycle],
            primal_files[cycle],
            args.dpi,
            args.figscale,
            args.cmap,
            args.vmin,
            args.vmax,
        )
        frames.append(fig_to_image(fig, args.dpi))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration,
        loop=0,
    )
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
