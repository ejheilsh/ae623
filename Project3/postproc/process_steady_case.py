#!/usr/bin/env python3
"""
Process one steady case into a self-contained output directory.

Outputs:
- summary.json
- summary.csv
- solution_summary.png
- cp_distribution.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from dg_utils import (
    integrate_wall_forces,
    maybe_read_dg_results,
    primitive_from_state,
    read_cell_results,
)
from plot_cp import plot_cp_distribution
from plot_results import plot_results, read_cell_residual, read_residual


def detect_case_name(results_path: Path) -> str:
    name = results_path.name
    if name.endswith("_results.bin"):
        return name[:-12]
    return results_path.stem


def compute_summary(meshfile: str, resultsfile: str, residualfile: str, cellresfile: str) -> dict:
    dg_filename, U_dg, p_order, ndof = maybe_read_dg_results(resultsfile)

    if U_dg is not None:
        U_avg = np.array([np.mean(elem, axis=0) for elem in U_dg])
    else:
        U_avg = read_cell_results(resultsfile)
        p_order = None
        ndof = None

    prim = primitive_from_state(U_avg)
    res_hist = read_residual(residualfile)
    cell_res = read_cell_residual(cellresfile)

    summary = {
        "meshfile": meshfile,
        "resultsfile": resultsfile,
        "residualfile": residualfile,
        "cellresfile": cellresfile,
        "dg_resultsfile": dg_filename or "",
        "dg_p_order": int(p_order) if p_order is not None else "",
        "dg_ndof": int(ndof) if ndof is not None else "",
        "rho_min": float(np.nanmin(prim["rho"])),
        "rho_max": float(np.nanmax(prim["rho"])),
        "p_min": float(np.nanmin(prim["p"])),
        "p_max": float(np.nanmax(prim["p"])),
        "mach_min": float(np.nanmin(prim["mach"])),
        "mach_max": float(np.nanmax(prim["mach"])),
        "entropy_min": float(np.nanmin(prim["entropy"])),
        "entropy_max": float(np.nanmax(prim["entropy"])),
        "residual_initial": float(res_hist[0]) if res_hist is not None and res_hist.size else "",
        "residual_final": float(res_hist[-1]) if res_hist is not None and res_hist.size else "",
        "residual_points": int(res_hist.size) if res_hist is not None else 0,
        "cell_residual_min": float(np.nanmin(cell_res)) if cell_res is not None and cell_res.size else "",
        "cell_residual_max": float(np.nanmax(cell_res)) if cell_res is not None and cell_res.size else "",
    }

    if U_dg is not None and p_order is not None:
        from dg_utils import read_gri_mesh

        mesh = read_gri_mesh(meshfile)
        c_x, c_y = integrate_wall_forces(mesh, U_dg, p_order)
        summary["c_x"] = float(c_x)
        summary["c_y"] = float(c_y)
    else:
        summary["c_x"] = ""
        summary["c_y"] = ""

    return summary


def write_summary(summary: dict, outdir: Path) -> None:
    with open(outdir / "summary.json", "w") as f:
      json.dump(summary, f, indent=2)

    with open(outdir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def main():
    parser = argparse.ArgumentParser(description="Process one steady solver case")
    parser.add_argument("meshfile")
    parser.add_argument("resultsfile")
    parser.add_argument("residualfile")
    parser.add_argument("cellresfile")
    parser.add_argument("--outdir", default=None,
                        help="Output directory (default: postproc_out/steady/<case>)")
    args = parser.parse_args()

    results_path = Path(args.resultsfile)
    case_name = detect_case_name(results_path)
    outdir = Path(args.outdir) if args.outdir else Path("postproc_out/steady") / case_name
    outdir.mkdir(parents=True, exist_ok=True)

    plot_results(
        args.meshfile,
        args.resultsfile,
        args.residualfile,
        args.cellresfile,
        show_plot=False,
        output_file=str(outdir / "solution_summary.png"),
    )
    plot_cp_distribution(
        args.meshfile,
        args.resultsfile,
        output_file=str(outdir / "cp_distribution.png"),
        show_plot=False,
    )

    summary = compute_summary(args.meshfile, args.resultsfile, args.residualfile, args.cellresfile)
    summary["case"] = case_name
    summary["output_dir"] = str(outdir)
    write_summary(summary, outdir)

    print(f"Wrote steady case outputs to {outdir}")


if __name__ == "__main__":
    main()
