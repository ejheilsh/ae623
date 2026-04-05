#!/usr/bin/env python3
"""
Generate six separate entropy contour figures for the highest available q=3
unsteady order on the 2k and 8k meshes at t ~= 100, 200, 300.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np

from dg_utils import (
    build_dg_triangulation,
    maybe_read_dg_results,
    primitive_from_state,
    read_cell_results,
    read_gri_mesh,
)
from plot_entropy_contour import plot_entropy_contour


GRID_TAGS = ("2k_q3", "8k_q3")
TARGET_TIMES = (60.0, 150.0, 240.0)
P_ORDERS = (3, 2, 1, 0)


def time_from_filename(filepath: str) -> float:
    basename = os.path.basename(filepath)
    return float(basename.replace("results_", "").split("_")[0])


def find_highest_case(base_dir: Path, grid_tag: str) -> tuple[int, Path] | None:
    for p in P_ORDERS:
        case_dir = base_dir / f"{grid_tag}_p{p}"
        if not case_dir.is_dir():
            continue
        snapshots = [
            f
            for f in glob.glob(str(case_dir / "results_*.bin"))
            if not f.endswith("_dg.bin")
        ]
        if snapshots:
            return p, case_dir
    return None


def nearest_snapshot(case_dir: Path, target_time: float) -> tuple[str, float] | None:
    snapshots = [
        f for f in glob.glob(str(case_dir / "results_*.bin")) if not f.endswith("_dg.bin")
    ]
    if not snapshots:
        return None
    best = min(snapshots, key=lambda f: abs(time_from_filename(f) - target_time))
    return best, time_from_filename(best)


def entropy_range(mesh_file: Path, snapshot_path: str) -> tuple[float, float]:
    mesh = read_gri_mesh(str(mesh_file))
    _, u_dg, p_order, _ = maybe_read_dg_results(snapshot_path)
    if u_dg is not None:
        _, _, _, entropy_log = build_dg_triangulation(mesh, u_dg, p_order, "entropy")
        entropy = np.exp(entropy_log)
    else:
        u = read_cell_results(snapshot_path)
        entropy = np.exp(primitive_from_state(u)["entropy"])
    return float(np.nanmin(entropy)), float(np.nanmax(entropy))


def main():
    parser = argparse.ArgumentParser(description="Plot unsteady entropy contours at milestone times.")
    parser.add_argument("--base-dir", default="unsteady_data")
    parser.add_argument("--outdir", default="postproc_out/task5")
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--xlim", nargs=2, type=float, default=None)
    parser.add_argument("--ylim", nargs=2, type=float, default=None)
    parser.add_argument("--colorbar-output", default=None)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    colorbar_output = args.colorbar_output
    if colorbar_output is None:
        colorbar_output = str(outdir / "entropy_colorbar.png")

    wrote_colorbar = False
    selected_plots = []

    for grid_tag in GRID_TAGS:
        highest = find_highest_case(base_dir, grid_tag)
        if highest is None:
            print(f"SKIP {grid_tag}: no unsteady snapshots found")
            continue

        p_order, case_dir = highest
        mesh_file = Path("grids") / f"{grid_tag}.gri"
        print(f"{grid_tag}: using highest available order p={p_order} from {case_dir}")

        for target_time in TARGET_TIMES:
            nearest = nearest_snapshot(case_dir, target_time)
            if nearest is None:
                print(f"SKIP {grid_tag} t={target_time:.0f}: no snapshots")
                continue

            snapshot_path, actual_time = nearest
            selected_plots.append(
                {
                    "grid_tag": grid_tag,
                    "p_order": p_order,
                    "mesh_file": mesh_file,
                    "snapshot_path": snapshot_path,
                    "target_time": target_time,
                    "actual_time": actual_time,
                    "output_file": outdir / f"entropy_{grid_tag}_p{p_order}_t{int(target_time)}.png",
                    "label_text": rf"$t={int(target_time)}$s",
                }
            )

    if not selected_plots:
        print("No milestone snapshots found.")
        return

    if args.vmin is None or args.vmax is None:
        range_pairs = [entropy_range(item["mesh_file"], item["snapshot_path"]) for item in selected_plots]
        auto_vmin = min(r[0] for r in range_pairs)
        auto_vmax = max(r[1] for r in range_pairs)
        used_vmin = args.vmin if args.vmin is not None else auto_vmin
        used_vmax = args.vmax if args.vmax is not None else auto_vmax
    else:
        used_vmin = args.vmin
        used_vmax = args.vmax

    print(
        f"Using entropy color scale: vmin={used_vmin:.6f}, vmax={used_vmax:.6f}, cmap={args.cmap}"
    )

    for item in selected_plots:
        plot_entropy_contour(
            str(item["mesh_file"]),
            item["snapshot_path"],
            output_file=str(item["output_file"]),
            show_plot=not args.no_show,
            cmap=args.cmap,
            vmin=used_vmin,
            vmax=used_vmax,
            xlim=args.xlim,
            ylim=args.ylim,
            dpi=args.dpi,
            colorbar_output_file=colorbar_output if not wrote_colorbar else None,
            label_text=item["label_text"],
        )
        wrote_colorbar = True
        print(
            f"  target t={item['target_time']:.0f}, using snapshot at t={item['actual_time']:.6f} -> {item['output_file']}"
        )


if __name__ == "__main__":
    main()
