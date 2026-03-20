#!/usr/bin/env python3
"""
postprocess_suite.py
====================
Batch postprocessing for the Project 3 Euler solver outputs.

Current capabilities:
- Scan a steady-results directory for steady_*_p*_results.bin cases
- Generate per-case figures: residual history, Mach field, entropy field,
  cell residual field, and mesh preview
- Generate summary CSV for all steady cases
- Generate residual-comparison overlays grouped by mesh stem
- Optionally process unsteady snapshot directories and emit selected Mach/entropy
  snapshot figures plus a snapshot index CSV

Important limitation:
The current solver writes one conservative state per element, not the full DG
coefficient field. For p>0, contour plots are therefore cell-average previews,
not true high-order within-element reconstructions. Additionally, not functional
for higher q meshes yet.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

GAMMA = 1.4

STEADY_RE = re.compile(r"^(steady_(?P<grid>.+?)_p(?P<p>\d+))_results\.bin$")
SNAPSHOT_RE = re.compile(r"^results_(?P<time>-?\d+\.\d+)_(?P<idx>\d+)\.bin$")


@dataclass
class SteadyCase:
    mesh_label: str
    grid_label: str
    p_order: int
    results_path: Path
    residual_path: Path | None
    cellres_path: Path | None

@dataclass
class RichSteadyCase:
    run_name: str
    grid_label: str
    p_order: int
    run_dir: Path
    state_avg_path: Path
    residual_path: Path
    cellres_path: Path
    wall_surface_path: Path | None
    summary_path: Path | None
    state_dg_path: Path | None

def process_single_rich_case(case: RichSteadyCase, verts: np.ndarray, tris: np.ndarray,
                             wall_edges: np.ndarray | None, outdir: Path, gamma: float) -> dict:
    case_out = outdir / case.run_name
    case_out.mkdir(parents=True, exist_ok=True)

    U = read_state_bin(str(case.state_avg_path))
    if U.shape[0] != tris.shape[0]:
        raise ValueError(
            f"Element count mismatch for {case.run_name}: mesh has {tris.shape[0]} but state has {U.shape[0]}"
        )

    fields = conservative_fields(U, gamma=gamma)
    R = read_series_bin(str(case.residual_path))
    Rc = read_series_bin(str(case.cellres_path))

    save_mesh_plot(verts, tris, wall_edges, case_out / "mesh.png", f"Mesh Preview: {case.grid_label}")
    save_face_field_plot(verts, tris, fields["mach"], case_out / "mach.png",
                         f"Mach: {case.run_name}", "Mach", wall_edges)
    save_face_field_plot(verts, tris, fields["entropy"], case_out / "entropy.png",
                         f"Entropy: {case.run_name}", "log(p/rho^gamma)", wall_edges)
    save_face_field_plot(verts, tris, np.log10(np.maximum(Rc, 1e-300)),
                         case_out / "cell_residual.png",
                         f"log10(Cell Residual): {case.run_name}",
                         r"$\log_{10}(R_{cell})$", wall_edges)
    save_residual_plot(R, case_out / "residual_history.png", f"Residual History: {case.run_name}")

    summary_dict = {}
    if case.summary_path is not None:
        summary_dict = read_summary_txt(str(case.summary_path))
        save_force_summary(summary_dict, case_out, case.run_name)

    if case.wall_surface_path is not None:
        wall = read_wall_surface_csv(str(case.wall_surface_path))
        save_cp_plots(wall, case_out, case.run_name)

    row = {
        "case": case.run_name,
        "grid": case.grid_label,
        "p": case.p_order,
        "state_avg_file": str(case.state_avg_path),
        "residual_file": str(case.residual_path),
        "cellres_file": str(case.cellres_path),
        "wall_surface_file": str(case.wall_surface_path) if case.wall_surface_path else "",
        "summary_file": str(case.summary_path) if case.summary_path else "",
        "figure_dir": str(case_out),
        "residual_0": float(R[0]) if R.size else math.nan,
        "residual_final": float(R[-1]) if R.size else math.nan,
        "rho_min": float(np.nanmin(fields["rho"])),
        "rho_max": float(np.nanmax(fields["rho"])),
        "p_min": float(np.nanmin(fields["p"])),
        "p_max": float(np.nanmax(fields["p"])),
        "mach_min": float(np.nanmin(fields["mach"])),
        "mach_max": float(np.nanmax(fields["mach"])),
        "Fx_raw": float(summary_dict.get("Fx_raw", math.nan)) if summary_dict else math.nan,
        "Fy_raw": float(summary_dict.get("Fy_raw", math.nan)) if summary_dict else math.nan,
        "Lift_raw": float(summary_dict.get("Lift_raw", math.nan)) if summary_dict else math.nan,
        "Drag_raw": float(summary_dict.get("Drag_raw", math.nan)) if summary_dict else math.nan,
    }
    return row

def save_force_summary(summary: dict[str, float | str], outdir: Path, title_stub: str):
    keys = ["Fx_raw", "Fy_raw", "Lift_raw", "Drag_raw"]
    vals = [float(summary.get(k, np.nan)) for k in keys]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(keys, vals)
    ax.set_ylabel("Raw integrated force")
    ax.set_title(f"Force summary: {title_stub}")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "force_summary.png", dpi=200)
    plt.close(fig)

    with open(outdir / "force_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["quantity", "value"])
        for k, v in zip(keys, vals):
            writer.writerow([k, v])

def save_cp_plots(wall: dict[str, np.ndarray], outdir: Path, title_stub: str):
    x = wall["x"]
    y = wall["y"]
    cp = wall["cp"]

    # cp vs x
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x, cp, s=10)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$c_p$")
    ax.set_title(f"cp vs x: {title_stub}")
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # common aerodynamic convention
    fig.tight_layout()
    fig.savefig(outdir / "cp_vs_x.png", dpi=200)
    plt.close(fig)

    # physical wall distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    sc = ax.scatter(x, y, c=cp, s=12)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Wall cp distribution: {title_stub}")
    ax.grid(True, alpha=0.2)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"$c_p$")
    fig.tight_layout()
    fig.savefig(outdir / "cp_wall_scatter.png", dpi=200)
    plt.close(fig)

def read_summary_txt(fname: str) -> dict[str, float | str]:
    out: dict[str, float | str] = {}
    with open(fname, "r") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            k, v = parts
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out

def read_wall_surface_csv(fname: str) -> dict[str, np.ndarray]:
    data = np.genfromtxt(fname, delimiter=",", names=True)
    return {name: data[name] for name in data.dtype.names}

def scan_rich_steady_cases(rich_root: Path) -> list[RichSteadyCase]:
    cases: list[RichSteadyCase] = []
    pat = re.compile(r"^steady_(?P<grid>.+)_p(?P<p>\d+)$")

    for d in sorted(rich_root.glob("steady_*")):
        if not d.is_dir():
            continue
        m = pat.match(d.name)
        if not m:
            continue

        grid = m.group("grid")
        p_order = int(m.group("p"))

        state_avg = d / "state_avg.bin"
        residual = d / "residual.bin"
        cellres = d / "cell_res.bin"

        if not (state_avg.exists() and residual.exists() and cellres.exists()):
            continue

        cases.append(
            RichSteadyCase(
                run_name=d.name,
                grid_label=grid,
                p_order=p_order,
                run_dir=d,
                state_avg_path=state_avg,
                residual_path=residual,
                cellres_path=cellres,
                wall_surface_path=(d / "wall_surface.csv") if (d / "wall_surface.csv").exists() else None,
                summary_path=(d / "summary.txt") if (d / "summary.txt").exists() else None,
                state_dg_path=(d / "state_dg.bin") if (d / "state_dg.bin").exists() else None,
            )
        )
    return cases


def read_gri_tokens(fname: str):
    with open(fname, "r") as f:
        tokens = f.read().split()

    pos = 0

    def tok() -> str:
        nonlocal pos
        t = tokens[pos]
        pos += 1
        return t

    def tint() -> int:
        return int(tok())

    def tfloat() -> float:
        return float(tok())

    nnodes, nelem_total, dim = tint(), tint(), tint()
    if dim != 2:
        raise ValueError(f"Expected 2D mesh; got dim={dim}")

    verts = np.zeros((nnodes, 2), dtype=float)
    for i in range(nnodes):
        verts[i, 0] = tfloat()
        verts[i, 1] = tfloat()

    nbgroups = tint()
    bgroups: list[tuple[str, np.ndarray]] = []
    for _ in range(nbgroups):
        nb, nn = tint(), tint()
        name = tok()
        edges = np.zeros((nb, nn), dtype=int)
        for j in range(nb):
            for k in range(nn):
                edges[j, k] = tint()
        bgroups.append((name, edges))

    elem_blocks: list[tuple[int, str, np.ndarray]] = []
    ne_count = 0
    while ne_count < nelem_total:
        ne_block, deg = tint(), tint()
        etype = tok()
        npe = (deg + 1) * (deg + 2) // 2
        elems = np.zeros((ne_block, npe), dtype=int)
        for j in range(ne_block):
            for k in range(npe):
                elems[j, k] = tint()
        elem_blocks.append((deg, etype, elems))
        ne_count += ne_block

    trailing = tokens[pos:]
    return verts, bgroups, elem_blocks, trailing


def extract_triangle_corners(elem_blocks: Iterable[tuple[int, str, np.ndarray]]) -> np.ndarray:
    tri_list: list[np.ndarray] = []
    for deg, etype, elems in elem_blocks:
        if "Tri" not in etype:
            raise ValueError(f"Only triangular blocks supported here; got {etype}")
        npe = elems.shape[1]
        if npe != (deg + 1) * (deg + 2) // 2:
            raise ValueError(f"Bad block: deg={deg}, npe={npe}")
        i0 = 0
        i1 = deg
        i2 = npe - 1
        tris = np.column_stack((elems[:, i0], elems[:, i1], elems[:, i2])) - 1
        tri_list.append(tris)
    if not tri_list:
        raise ValueError("No element blocks found")
    return np.vstack(tri_list)


def parse_wall_edges(bgroups: list[tuple[str, np.ndarray]]) -> np.ndarray | None:
    wall_edges = []
    for name, edges in bgroups:
        if name.lower() == "wall" and edges.shape[1] >= 2:
            wall_edges.append(edges[:, [0, 1]] - 1)
    if not wall_edges:
        return None
    return np.vstack(wall_edges)


def read_state_bin(fname: str) -> np.ndarray:
    with open(fname, "rb") as f:
        raw = f.read(4)
        if len(raw) != 4:
            raise ValueError(f"Could not read element count from {fname}")
        ne = struct.unpack("i", raw)[0]
        data = np.fromfile(f, dtype=np.float64)
    if data.size != 4 * ne:
        raise ValueError(f"State file size mismatch in {fname}: expected {4*ne}, got {data.size}")
    return data.reshape(ne, 4)


def read_series_bin(fname: str) -> np.ndarray:
    with open(fname, "rb") as f:
        raw = f.read(4)
        if len(raw) != 4:
            raise ValueError(f"Could not read entry count from {fname}")
        n = struct.unpack("i", raw)[0]
        data = np.fromfile(f, dtype=np.float64)
    if data.size != n:
        raise ValueError(f"Series file size mismatch in {fname}: expected {n}, got {data.size}")
    return data


def conservative_fields(U: np.ndarray, gamma: float = GAMMA):
    rho = U[:, 0]
    rhou = U[:, 1]
    rhov = U[:, 2]
    rhoE = U[:, 3]
    u = rhou / rho
    v = rhov / rho
    q2 = u * u + v * v
    p = (gamma - 1.0) * (rhoE - 0.5 * rho * q2)
    a2 = gamma * p / rho
    mach = np.full(rho.shape, np.nan)
    ok = (rho > 0.0) & (p > 0.0) & (a2 > 0.0) & np.isfinite(a2)
    mach[ok] = np.sqrt(q2[ok] / a2[ok])
    entropy = np.full(rho.shape, np.nan)
    entropy[ok] = np.log(p[ok] / (rho[ok] ** gamma))
    return {
        "rho": rho,
        "u": u,
        "v": v,
        "p": p,
        "mach": mach,
        "entropy": entropy,
    }


def make_triangulation(verts: np.ndarray, tris: np.ndarray) -> mtri.Triangulation:
    return mtri.Triangulation(verts[:, 0], verts[:, 1], tris)


def save_mesh_plot(verts: np.ndarray, tris: np.ndarray, wall_edges: np.ndarray | None,
                   outpath: Path, title: str):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.triplot(verts[:, 0], verts[:, 1], tris, color="0.65", linewidth=0.4)
    if wall_edges is not None:
        for a, b in wall_edges:
            ax.plot([verts[a, 0], verts[b, 0]], [verts[a, 1], verts[b, 1]], color="black", linewidth=1.2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def save_residual_plot(R: np.ndarray, outpath: Path, title: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(np.arange(R.size), R, linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual Norm")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def save_face_field_plot(verts: np.ndarray, tris: np.ndarray, values: np.ndarray,
                         outpath: Path, title: str, cbar_label: str,
                         wall_edges: np.ndarray | None = None):
    triang = make_triangulation(verts, tris)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    tpc = ax.tripcolor(triang, facecolors=values, shading="flat")
    if wall_edges is not None:
        for a, b in wall_edges:
            ax.plot([verts[a, 0], verts[b, 0]], [verts[a, 1], verts[b, 1]], color="black", linewidth=1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True, alpha=0.15)
    cbar = fig.colorbar(tpc, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


def scan_steady_cases(steady_dir: Path) -> list[SteadyCase]:
    cases: list[SteadyCase] = []
    for path in sorted(steady_dir.glob("steady_*_results.bin")):
        m = STEADY_RE.match(path.name)
        if not m:
            continue
        grid = m.group("grid")
        p_order = int(m.group("p"))
        base = path.with_name(f"steady_{grid}_p{p_order}")
        residual = base.with_name(base.name + "_residual.bin")
        cellres = base.with_name(base.name + "_cell_res.bin")
        cases.append(
            SteadyCase(
                mesh_label=grid,
                grid_label=grid,
                p_order=p_order,
                results_path=path,
                residual_path=residual if residual.exists() else None,
                cellres_path=cellres if cellres.exists() else None,
            )
        )
    return cases


def process_single_steady_case(case: SteadyCase, verts: np.ndarray, tris: np.ndarray,
                               wall_edges: np.ndarray | None, outdir: Path,
                               gamma: float) -> dict:
    case_out = outdir / f"steady_{case.grid_label}_p{case.p_order}"
    case_out.mkdir(parents=True, exist_ok=True)

    U = read_state_bin(str(case.results_path))
    if U.shape[0] != tris.shape[0]:
        raise ValueError(
            f"Element count mismatch for {case.results_path.name}: mesh has {tris.shape[0]} but state has {U.shape[0]}"
        )
    fields = conservative_fields(U, gamma=gamma)
    R = read_series_bin(str(case.residual_path)) if case.residual_path else None
    Rc = read_series_bin(str(case.cellres_path)) if case.cellres_path else None

    save_mesh_plot(verts, tris, wall_edges, case_out / "mesh.png", f"Mesh Preview: {case.grid_label}")
    save_face_field_plot(verts, tris, fields["mach"], case_out / "mach.png",
                         f"Cell-Centered Mach: {case.results_path.stem}", "Mach", wall_edges)
    save_face_field_plot(verts, tris, fields["entropy"], case_out / "entropy.png",
                         f"Cell-Centered Entropy: {case.results_path.stem}", "log(p/rho^gamma)", wall_edges)
    if Rc is not None:
        save_face_field_plot(verts, tris, np.log10(np.maximum(Rc, 1e-300)),
                             case_out / "cell_residual.png",
                             f"log10(Cell Residual): {case.results_path.stem}",
                             r"$\log_{10}(R_{cell})$", wall_edges)
    if R is not None:
        save_residual_plot(R, case_out / "residual_history.png",
                           f"Residual History: {case.results_path.stem}")

    finite_mach = fields["mach"][np.isfinite(fields["mach"])]
    finite_entropy = fields["entropy"][np.isfinite(fields["entropy"])]
    summary = {
        "case": case.results_path.stem,
        "grid": case.grid_label,
        "p": case.p_order,
        "state_file": str(case.results_path),
        "residual_file": str(case.residual_path) if case.residual_path else "",
        "cellres_file": str(case.cellres_path) if case.cellres_path else "",
        "rho_min": float(np.nanmin(fields["rho"])),
        "rho_max": float(np.nanmax(fields["rho"])),
        "p_min": float(np.nanmin(fields["p"])),
        "p_max": float(np.nanmax(fields["p"])),
        "mach_min": float(np.nanmin(finite_mach)) if finite_mach.size else math.nan,
        "mach_max": float(np.nanmax(finite_mach)) if finite_mach.size else math.nan,
        "entropy_min": float(np.nanmin(finite_entropy)) if finite_entropy.size else math.nan,
        "entropy_max": float(np.nanmax(finite_entropy)) if finite_entropy.size else math.nan,
        "residual_0": float(R[0]) if R is not None and R.size else math.nan,
        "residual_final": float(R[-1]) if R is not None and R.size else math.nan,
        "residual_points": int(R.size) if R is not None else 0,
        "figure_dir": str(case_out),
    }
    return summary


def save_residual_overlays(cases: list[SteadyCase], outdir: Path):
    by_grid: dict[str, list[SteadyCase]] = defaultdict(list)
    for case in cases:
        if case.residual_path and case.residual_path.exists():
            by_grid[case.grid_label].append(case)

    comp_dir = outdir / "comparisons"
    comp_dir.mkdir(parents=True, exist_ok=True)

    for grid, group in by_grid.items():
        fig, ax = plt.subplots(figsize=(7, 5))
        for case in sorted(group, key=lambda c: c.p_order):
            R = read_series_bin(str(case.residual_path))
            ax.semilogy(np.arange(R.size), R, linewidth=1.5, label=f"p={case.p_order}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Residual Norm")
        ax.set_title(f"Residual Comparison: {grid}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(comp_dir / f"{grid}_residual_comparison.png", dpi=200)
        plt.close(fig)


def write_summary_csv(rows: list[dict], outpath: Path):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(outpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_unsteady_dir(mesh_path: Path, snapshot_dir: Path, outdir: Path,
                         gamma: float, snapshot_count: int):
    verts, bgroups, elem_blocks, _ = read_gri_tokens(str(mesh_path))
    tris = extract_triangle_corners(elem_blocks)
    wall_edges = parse_wall_edges(bgroups)

    snap_files = []
    for path in sorted(snapshot_dir.glob("results_*.bin")):
        m = SNAPSHOT_RE.match(path.name)
        if m:
            snap_files.append((float(m.group("time")), int(m.group("idx")), path))
    if not snap_files:
        print(f"[WARN] No snapshot files found in {snapshot_dir}")
        return

    target_indices = np.linspace(0, len(snap_files) - 1, num=min(snapshot_count, len(snap_files)), dtype=int)
    selected = [snap_files[i] for i in target_indices]

    outdir.mkdir(parents=True, exist_ok=True)
    save_mesh_plot(verts, tris, wall_edges, outdir / "mesh.png", f"Mesh Preview: {mesh_path.name}")

    rows = []
    for t, idx, path in selected:
        U = read_state_bin(str(path))
        fields = conservative_fields(U, gamma=gamma)
        save_face_field_plot(verts, tris, fields["mach"], outdir / f"mach_t{t:.6f}_{idx:04d}.png",
                             f"Mach Snapshot t={t:.6f}", "Mach", wall_edges)
        save_face_field_plot(verts, tris, fields["entropy"], outdir / f"entropy_t{t:.6f}_{idx:04d}.png",
                             f"Entropy Snapshot t={t:.6f}", "log(p/rho^gamma)", wall_edges)
        rows.append({"time": t, "snapshot_index": idx, "state_file": str(path)})

    write_summary_csv(rows, outdir / "snapshot_index.csv")


def main():
    parser = argparse.ArgumentParser(description="Batch postprocessing for Project 3 outputs")
    parser.add_argument("--mesh", required=True, help="Mesh .gri used for the cases in this call")
    parser.add_argument("--steady-dir", default=None, help="Directory containing steady_*_results.bin files")
    parser.add_argument("--unsteady-dir", default=None, help="Directory containing results_*.bin snapshots")
    parser.add_argument("--outdir", default="figures_suite", help="Output root directory")
    parser.add_argument("--gamma", type=float, default=1.4)
    parser.add_argument("--snapshot-count", type=int, default=6,
                        help="How many unsteady snapshots to plot if --unsteady-dir is used")
    parser.add_argument("--rich-root", default=None,
                        help="Root directory containing rich steady run dirs like results/steady_*")
    args = parser.parse_args()

    if args.steady_dir is None and args.unsteady_dir is None and args.rich_root is None:
        raise SystemExit("Provide at least one of --steady-dir, --unsteady-dir, or --rich-root")

    mesh_path = Path(args.mesh)
    outroot = Path(args.outdir)
    outroot.mkdir(parents=True, exist_ok=True)

    if args.rich_root is not None:
        rich_root = Path(args.rich_root)
        verts, bgroups, elem_blocks, _ = read_gri_tokens(str(mesh_path))
        tris = extract_triangle_corners(elem_blocks)
        wall_edges = parse_wall_edges(bgroups)

        rich_cases = scan_rich_steady_cases(rich_root)
        if not rich_cases:
            print(f"[WARN] No rich steady cases found in {rich_root}")
        else:
            rows = []
            rich_out = outroot / "rich_steady"
            rich_out.mkdir(parents=True, exist_ok=True)
            for case in sorted(rich_cases, key=lambda c: (c.grid_label, c.p_order)):
                print(f"Processing rich steady case: {case.run_name}")
                rows.append(process_single_rich_case(case, verts, tris, wall_edges, rich_out, args.gamma))
            write_summary_csv(rows, rich_out / "rich_steady_summary.csv")
            print(f"Wrote rich steady summary to {rich_out / 'rich_steady_summary.csv'}")

    if args.steady_dir is not None:
        steady_dir = Path(args.steady_dir)
        verts, bgroups, elem_blocks, _ = read_gri_tokens(str(mesh_path))
        tris = extract_triangle_corners(elem_blocks)
        wall_edges = parse_wall_edges(bgroups)
        cases = scan_steady_cases(steady_dir)
        if not cases:
            print(f"[WARN] No steady cases found in {steady_dir}")
        else:
            rows = []
            steady_out = outroot / "steady"
            steady_out.mkdir(parents=True, exist_ok=True)
            for case in sorted(cases, key=lambda c: (c.grid_label, c.p_order)):
                print(f"Processing steady case: {case.results_path.name}")
                rows.append(process_single_steady_case(case, verts, tris, wall_edges, steady_out, args.gamma))
            save_residual_overlays(cases, steady_out)
            write_summary_csv(rows, steady_out / "steady_summary.csv")
            print(f"Wrote steady summary to {steady_out / 'steady_summary.csv'}")

    if args.unsteady_dir is not None:
        unsteady_dir = Path(args.unsteady_dir)
        unsteady_out = outroot / f"unsteady_{unsteady_dir.name}"
        print(f"Processing unsteady directory: {unsteady_dir}")
        process_unsteady_dir(mesh_path, unsteady_dir, unsteady_out, args.gamma, args.snapshot_count)
        print(f"Wrote unsteady figures to {unsteady_out}")

    print("Done.")

if __name__ == "__main__":
    main()
