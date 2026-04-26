#!/usr/bin/env python3
"""
Postprocess final adapted-grid cases and store lift values.

By default this script scans:
  final_solutions/<ncells>/<curvature>/<solver_order>/iter<N>_soln.bin

and matches each solution with:
  final_grids/<ncells>/<curvature>/<solver_order>/iter<N>.gri

It writes per-iteration summaries, per-case CL tables, and aggregate CL tables
under final_cases/.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from dg_utils import integrate_wall_forces, maybe_read_dg_results, read_gri_mesh


DEFAULT_REFERENCE_MESH = Path(
    "/Users/eheilshorn/Courses/AE623/Code/ae623/Project3/grids/128k_q3.gri"
)
DEFAULT_REFERENCE_RESULTS = Path(
    "/Users/eheilshorn/Courses/AE623/Code/ae623/Project3/data_steady/steady_128k_q3_p0_results.bin"
)
DEFAULT_REFERENCE_LABEL = "Project3/128k/q3/p0"
DEFAULT_PROJECT3_DATA = Path("/Users/eheilshorn/Courses/AE623/Code/ae623/Project3/data_steady")
DEFAULT_PROJECT3_GRIDS = Path("/Users/eheilshorn/Courses/AE623/Code/ae623/Project3/grids")
DEFAULT_UNIFORM_GRIDS = ["2k", "8k", "32k", "128k"]


ROW_FIELDS = [
    "case",
    "ncells",
    "curvature",
    "solver_order",
    "iteration",
    "n_nodes",
    "n_elements",
    "meshfile",
    "resultsfile",
    "dg_resultsfile",
    "dg_p_order",
    "dg_ndof",
    "c_x",
    "c_y",
    "cl",
    "reference_case",
    "reference_cl",
    "cl_error",
    "abs_cl_error",
    "residual_initial",
    "residual_final",
    "residual_points",
]

UNIFORM_ROW_FIELDS = ROW_FIELDS + [
    "series",
    "source_results_root",
    "source_mesh_root",
]


def natural_key(value: str) -> list[object]:
    parts = re.split(r"(\d+)", value)
    return [int(part) if part.isdigit() else part for part in parts]


def case_plot_label(case: str) -> str:
    parts = case.split("/")
    if len(parts) == 3:
        ncells, _curvature, solver_order = parts
        return f"{ncells} {solver_order}"
    return case


def read_residual(filename: Path) -> np.ndarray | None:
    if not filename.exists():
        return None
    with filename.open("rb") as f:
        count = np.fromfile(f, dtype=np.int32, count=1)
        if count.size == 0:
            return np.array([], dtype=np.float64)
        return np.fromfile(f, dtype=np.float64, count=int(count[0]))


def infer_iteration(results_path: Path) -> int:
    match = re.fullmatch(r"iter(\d+)_soln\.bin", results_path.name)
    if not match:
        raise ValueError(f"Cannot infer iteration from {results_path}")
    return int(match.group(1))


def normalize_curvature_filter(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip("/")
    if re.fullmatch(r"\d+", value):
        value = f"q{value}"
    if not re.fullmatch(r"q\d+", value):
        raise ValueError("curvature must look like q1, q2, q3, or a number such as 1")
    return value


def discover_solution_files(
    final_solutions: Path,
    case_filter: str | None,
    curvature_filter: str | None,
) -> list[Path]:
    files = sorted(
        final_solutions.glob("*/*/*/iter*_soln.bin"),
        key=lambda path: [natural_key(part) for part in path.parts],
    )
    normalized_case = case_filter.strip("/") if case_filter is not None else None

    selected = []
    for path in files:
        rel = path.relative_to(final_solutions)
        case = "/".join(rel.parts[:3])
        curvature = rel.parts[1]
        if normalized_case is not None and case != normalized_case:
            continue
        if curvature_filter is not None and curvature != curvature_filter:
            continue
        selected.append(path)
    return selected


def compute_force_summary(meshfile: Path, resultsfile: Path) -> dict:
    mesh = read_gri_mesh(str(meshfile))
    dg_filename, u_dg, p_order, ndof = maybe_read_dg_results(str(resultsfile))
    if u_dg is None or p_order is None:
        raise FileNotFoundError(f"Missing DG solution next to {resultsfile}")

    c_x, c_y = integrate_wall_forces(mesh, u_dg, p_order)
    return {
        "n_nodes": len(mesh["nodes"]),
        "n_elements": len(mesh["elements"]),
        "dg_resultsfile": dg_filename,
        "dg_p_order": int(p_order),
        "dg_ndof": int(ndof),
        "c_x": float(c_x),
        "c_y": float(c_y),
        "cl": float(c_y),
    }


def compute_reference_summary(meshfile: Path, resultsfile: Path, label: str) -> dict:
    if not meshfile.exists():
        raise FileNotFoundError(f"Reference mesh does not exist: {meshfile}")
    if not resultsfile.exists():
        raise FileNotFoundError(f"Reference results file does not exist: {resultsfile}")

    summary = compute_force_summary(meshfile, resultsfile)
    summary.update(
        {
            "case": label,
            "meshfile": str(meshfile),
            "resultsfile": str(resultsfile),
        }
    )
    return summary


def mesh_name_for_case(ncells: str, curvature: str) -> str:
    if curvature == "q1":
        return f"{ncells}.gri"
    return f"{ncells}_{curvature}.gri"


def results_name_for_case(ncells: str, curvature: str, solver_order: str) -> str:
    if curvature == "q1":
        return f"steady_{ncells}_{solver_order}_results.bin"
    return f"steady_{ncells}_{curvature}_{solver_order}_results.bin"


def parse_uniform_grids(value: str) -> list[str]:
    grids = [part.strip() for part in value.split(",") if part.strip()]
    return grids or DEFAULT_UNIFORM_GRIDS


def default_uniform_sources() -> list[tuple[Path, Path]]:
    return [
        (Path("data_steady"), Path("grids")),
        (DEFAULT_PROJECT3_DATA, DEFAULT_PROJECT3_GRIDS),
    ]


def uniform_source_pairs(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    if args.uniform_results_root is None and args.uniform_mesh_root is None:
        return default_uniform_sources()

    result_roots = [Path(root) for root in (args.uniform_results_root or ["data_steady"])]
    mesh_roots = [Path(root) for root in (args.uniform_mesh_root or ["grids"])]
    if len(mesh_roots) == 1 and len(result_roots) > 1:
        mesh_roots *= len(result_roots)
    if len(result_roots) != len(mesh_roots):
        raise ValueError("--uniform-results-root and --uniform-mesh-root counts must match")
    return list(zip(result_roots, mesh_roots))


def uniform_paths_for_source(
    results_root: Path,
    mesh_root: Path,
    ncells: str,
    curvature: str,
    solver_order: str,
) -> tuple[Path, Path]:
    return (
        mesh_root / mesh_name_for_case(ncells, curvature),
        results_root / results_name_for_case(ncells, curvature, solver_order),
    )


def complete_uniform_source(
    sources: list[tuple[Path, Path]],
    grids: list[str],
    curvature: str,
    solver_order: str,
) -> tuple[Path, Path] | None:
    for results_root, mesh_root in sources:
        complete = True
        for ncells in grids:
            meshfile, resultsfile = uniform_paths_for_source(
                results_root, mesh_root, ncells, curvature, solver_order
            )
            if not meshfile.exists() or not resultsfile.exists():
                complete = False
                break
        if complete:
            return results_root, mesh_root
    return None


def compute_uniform_rows(args: argparse.Namespace, reference: dict) -> tuple[list[dict], list[dict]]:
    if args.no_uniform:
        return [], []

    grids = parse_uniform_grids(args.uniform_grids)
    sources = uniform_source_pairs(args)
    selected_source = complete_uniform_source(
        sources, grids, args.uniform_curvature, args.uniform_order
    )
    skipped: list[dict] = []

    rows: list[dict] = []
    for ncells in grids:
        candidate_sources = [selected_source] if selected_source else sources
        selected_paths = None
        for source in candidate_sources:
            if source is None:
                continue
            results_root, mesh_root = source
            meshfile, resultsfile = uniform_paths_for_source(
                results_root,
                mesh_root,
                ncells,
                args.uniform_curvature,
                args.uniform_order,
            )
            if meshfile.exists() and resultsfile.exists():
                selected_paths = results_root, mesh_root, meshfile, resultsfile
                break

        if selected_paths is None:
            skipped.append(
                {
                    "case": f"{ncells}/{args.uniform_curvature}/{args.uniform_order}",
                    "reason": "missing uniform mesh or solution in configured roots",
                }
            )
            continue

        results_root, mesh_root, meshfile, resultsfile = selected_paths
        try:
            force_summary = compute_force_summary(meshfile, resultsfile)
        except Exception as exc:
            skipped.append(
                {
                    "case": f"{ncells}/{args.uniform_curvature}/{args.uniform_order}",
                    "meshfile": str(meshfile),
                    "resultsfile": str(resultsfile),
                    "reason": str(exc),
                }
            )
            continue

        reference_cl = float(reference["cl"])
        row = {
            "series": "uniform",
            "case": f"uniform/{args.uniform_curvature}/{args.uniform_order}",
            "ncells": ncells,
            "curvature": args.uniform_curvature,
            "solver_order": args.uniform_order,
            "iteration": "",
            "meshfile": str(meshfile),
            "resultsfile": str(resultsfile),
            "reference_case": reference["case"],
            "reference_cl": reference_cl,
            "cl_error": force_summary["cl"] - reference_cl,
            "abs_cl_error": abs(force_summary["cl"] - reference_cl),
            "residual_initial": "",
            "residual_final": "",
            "residual_points": "",
            "source_results_root": str(results_root),
            "source_mesh_root": str(mesh_root),
        }
        row.update(force_summary)
        rows.append(row)

    rows.sort(key=lambda row: natural_key(row["ncells"]))
    return rows, skipped


def write_json(path: Path, payload: dict | list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] = ROW_FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_plots(
    outdir: Path,
    rows: list[dict],
    reference_cl: float,
    reference_label: str,
    uniform_rows: list[dict],
) -> None:
    if not rows and not uniform_rows:
        return

    mpl_config = Path(tempfile.gettempdir()) / "project4_matplotlib"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available; skipping CL plots")
        return

    rows_by_case: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_case[row["case"]].append(row)

    for case, case_rows in rows_by_case.items():
        case_rows.sort(key=lambda row: row["iteration"])
        case_dir = outdir / case
        case_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(
            [row["iteration"] for row in case_rows],
            [row["cl"] for row in case_rows],
            marker="o",
            linewidth=1.5,
        )
        ax.axhline(reference_cl, color="0.25", linestyle=":", linewidth=1.5, label=reference_label)
        ax.set_xlabel("Adaptation iteration")
        ax.set_ylabel("CL")
        ax.set_title(case)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(case_dir / "cl_convergence.png", dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for case, case_rows in sorted(rows_by_case.items(), key=lambda item: natural_key(item[0])):
        case_rows.sort(key=lambda row: row["iteration"])
        ax.plot(
            [row["iteration"] for row in case_rows],
            [row["cl"] for row in case_rows],
            marker="o",
            linewidth=1.5,
            label=case_plot_label(case),
        )
    ax.axhline(reference_cl, color="0.25", linestyle=":", linewidth=1.5, label=reference_label)
    ax.set_xlabel("Adaptation iteration")
    ax.set_ylabel("CL")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "cl_convergence.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for case, case_rows in sorted(rows_by_case.items(), key=lambda item: natural_key(item[0])):
        case_rows.sort(key=lambda row: row["n_elements"])
        ax.plot(
            [row["n_elements"] for row in case_rows],
            [row["cl"] for row in case_rows],
            marker="o",
            linewidth=1.5,
            label=case_plot_label(case),
        )
    if uniform_rows:
        uniform_rows.sort(key=lambda row: row["n_elements"])
        uniform_label = f"uniform {uniform_rows[0]['solver_order']}"
        ax.plot(
            [row["n_elements"] for row in uniform_rows],
            [row["cl"] for row in uniform_rows],
            color="black",
            linestyle="-.",
            marker="s",
            linewidth=1.8,
            label=uniform_label,
        )
    ax.axhline(reference_cl, color="0.25", linestyle=":", linewidth=1.5, label=reference_label)
    ax.set_xscale("log")
    ax.set_xlabel("Number of cells")
    ax.set_ylabel("CL")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "cl_vs_cells.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for case, case_rows in sorted(rows_by_case.items(), key=lambda item: natural_key(item[0])):
        case_rows.sort(key=lambda row: row["n_elements"])
        ax.plot(
            [np.log10(row["n_elements"]) for row in case_rows],
            [row["cl"] for row in case_rows],
            marker="o",
            linewidth=1.5,
            label=case_plot_label(case),
        )
    if uniform_rows:
        uniform_rows.sort(key=lambda row: row["n_elements"])
        uniform_label = f"uniform {uniform_rows[0]['solver_order']}"
        ax.plot(
            [np.log10(row["n_elements"]) for row in uniform_rows],
            [row["cl"] for row in uniform_rows],
            color="black",
            linestyle="-.",
            marker="s",
            linewidth=1.8,
            label=uniform_label,
        )
    ax.axhline(reference_cl, color="0.25", linestyle=":", linewidth=1.5, label=reference_label)
    ax.set_xlabel("log10(Number of cells)")
    ax.set_ylabel("CL")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "cl_vs_logh.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for case, case_rows in sorted(rows_by_case.items(), key=lambda item: natural_key(item[0])):
        case_rows.sort(key=lambda row: row["n_elements"])
        error_rows = [row for row in case_rows if row["abs_cl_error"] > 0.0]
        if not error_rows:
            continue
        ax.loglog(
            [row["n_elements"] for row in error_rows],
            [row["abs_cl_error"] for row in error_rows],
            marker="o",
            linewidth=1.5,
            label=case_plot_label(case),
        )
    if uniform_rows:
        uniform_rows.sort(key=lambda row: row["n_elements"])
        uniform_error_rows = [row for row in uniform_rows if row["abs_cl_error"] > 0.0]
        if uniform_error_rows:
            uniform_label = f"uniform {uniform_rows[0]['solver_order']}"
            ax.loglog(
                [row["n_elements"] for row in uniform_error_rows],
                [row["abs_cl_error"] for row in uniform_error_rows],
                color="black",
                linestyle="-.",
                marker="s",
                linewidth=1.8,
                label=uniform_label,
            )
    ax.set_xlabel("Number of cells")
    ax.set_ylabel(r"$|C_l - C_{l,\mathrm{ref}}|$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "cl_error.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for case, case_rows in sorted(rows_by_case.items(), key=lambda item: natural_key(item[0])):
        case_rows.sort(key=lambda row: row["iteration"])
        ax.semilogy(
            [row["iteration"] for row in case_rows],
            [max(row["abs_cl_error"], np.finfo(float).tiny) for row in case_rows],
            marker="o",
            linewidth=1.5,
            label=case_plot_label(case),
        )
    ax.set_xlabel("Adaptation iteration")
    ax.set_ylabel(r"$|C_l - C_{l,\mathrm{ref}}|$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "cl_error_by_iteration.png", dpi=200)
    plt.close(fig)


def process_cases(args: argparse.Namespace) -> list[dict]:
    final_grids = Path(args.final_grids)
    final_solutions = Path(args.final_solutions)
    outdir = Path(args.outdir)

    reference = compute_reference_summary(
        Path(args.reference_mesh), Path(args.reference_results), args.reference_label
    )
    reference_cl = float(reference["cl"])
    write_json(outdir / "reference_summary.json", reference)

    rows: list[dict] = []
    skipped: list[dict] = []
    curvature_filter = normalize_curvature_filter(args.curvature)
    solution_files = discover_solution_files(final_solutions, args.case, curvature_filter)
    if args.iteration is not None:
        wanted = set(args.iteration)
        solution_files = [path for path in solution_files if infer_iteration(path) in wanted]

    for resultsfile in solution_files:
        rel = resultsfile.relative_to(final_solutions)
        ncells, curvature, solver_order = rel.parts[:3]
        iteration = infer_iteration(resultsfile)
        meshfile = final_grids / ncells / curvature / solver_order / f"iter{iteration}.gri"

        case = f"{ncells}/{curvature}/{solver_order}"
        if not meshfile.exists():
            skipped.append(
                {
                    "case": case,
                    "iteration": iteration,
                    "resultsfile": str(resultsfile),
                    "reason": f"missing mesh {meshfile}",
                }
            )
            continue

        try:
            force_summary = compute_force_summary(meshfile, resultsfile)
        except Exception as exc:
            skipped.append(
                {
                    "case": case,
                    "iteration": iteration,
                    "resultsfile": str(resultsfile),
                    "meshfile": str(meshfile),
                    "reason": str(exc),
                }
            )
            continue

        residual = read_residual(resultsfile.with_name(f"iter{iteration}_residual.bin"))
        row = {
            "case": case,
            "ncells": ncells,
            "curvature": curvature,
            "solver_order": solver_order,
            "iteration": iteration,
            "meshfile": str(meshfile),
            "resultsfile": str(resultsfile),
            "reference_case": reference["case"],
            "reference_cl": reference_cl,
            "cl_error": force_summary["cl"] - reference_cl,
            "abs_cl_error": abs(force_summary["cl"] - reference_cl),
            "residual_initial": float(residual[0]) if residual is not None and residual.size else "",
            "residual_final": float(residual[-1]) if residual is not None and residual.size else "",
            "residual_points": int(residual.size) if residual is not None else 0,
        }
        row.update(force_summary)
        rows.append(row)

        iter_dir = outdir / case / f"iter{iteration}"
        write_json(iter_dir / "summary.json", row)
        write_csv(iter_dir / "summary.csv", [row])

    rows.sort(
        key=lambda row: (
            natural_key(row["ncells"]),
            natural_key(row["curvature"]),
            natural_key(row["solver_order"]),
            row["iteration"],
        )
    )
    write_csv(outdir / "cl_values.csv", rows)
    write_json(outdir / "cl_values.json", rows)

    rows_by_case: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_case[row["case"]].append(row)
    for case, case_rows in rows_by_case.items():
        case_rows.sort(key=lambda row: row["iteration"])
        write_csv(outdir / case / "cl_values.csv", case_rows)
        write_json(outdir / case / "cl_values.json", case_rows)

    uniform_rows, uniform_skipped = compute_uniform_rows(args, reference)
    write_csv(outdir / "uniform_cl_values.csv", uniform_rows, fieldnames=UNIFORM_ROW_FIELDS)
    write_json(outdir / "uniform_cl_values.json", uniform_rows)

    if skipped:
        write_json(outdir / "skipped_cases.json", skipped)
    if uniform_skipped:
        write_json(outdir / "skipped_uniform_cases.json", uniform_skipped)

    if args.plots:
        write_plots(outdir, rows, reference_cl, reference["case"], uniform_rows)

    print(f"Reference CL: {reference_cl:.12g} ({reference['case']})")
    print(f"Processed {len(rows)} final solution(s)")
    print(f"Processed {len(uniform_rows)} uniform solution(s)")
    if skipped:
        print(f"Skipped {len(skipped)} solution(s); see {outdir / 'skipped_cases.json'}")
    if uniform_skipped:
        print(f"Skipped {len(uniform_skipped)} uniform solution(s); see {outdir / 'skipped_uniform_cases.json'}")
    print(f"Wrote {outdir / 'cl_values.csv'}")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute CL values for final adapted-grid solutions."
    )
    parser.add_argument("--final-grids", default="final_grids", help="Root final grid directory")
    parser.add_argument(
        "--final-solutions", default="final_solutions", help="Root final solution directory"
    )
    parser.add_argument("--outdir", default="final_cases", help="Output directory")
    parser.add_argument(
        "--reference-mesh",
        default=str(DEFAULT_REFERENCE_MESH),
        help="Reference mesh",
    )
    parser.add_argument(
        "--reference-results",
        default=str(DEFAULT_REFERENCE_RESULTS),
        help="Reference solution",
    )
    parser.add_argument(
        "--reference-label",
        default=DEFAULT_REFERENCE_LABEL,
        help="Reference label to use in tables and plot legends",
    )
    parser.add_argument(
        "--case",
        default=None,
        help="Optional case filter like 8k/q3/p0",
    )
    parser.add_argument(
        "--curvature",
        default=None,
        help="Optional curvature filter like q1, q2, q3, or 1",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        action="append",
        help="Optional iteration to process; repeat for multiple iterations",
    )
    parser.add_argument(
        "--no-uniform",
        action="store_true",
        help="Skip uniform-refinement comparison data and curves",
    )
    parser.add_argument(
        "--uniform-curvature",
        default="q3",
        help="Uniform comparison curvature label (default: q3)",
    )
    parser.add_argument(
        "--uniform-order",
        default="p0",
        help="Uniform comparison solver order label (default: p0)",
    )
    parser.add_argument(
        "--uniform-grids",
        default=",".join(DEFAULT_UNIFORM_GRIDS),
        help="Comma-separated uniform grids to plot (default: 2k,8k,32k,128k)",
    )
    parser.add_argument(
        "--uniform-results-root",
        action="append",
        default=None,
        help="Uniform results root to search; repeat to add roots",
    )
    parser.add_argument(
        "--uniform-mesh-root",
        action="append",
        default=None,
        help="Uniform mesh root paired with --uniform-results-root; repeat to add roots",
    )
    parser.add_argument(
        "--no-plots",
        dest="plots",
        action="store_false",
        help="Skip CL convergence plots",
    )
    parser.set_defaults(plots=True)
    args = parser.parse_args()

    process_cases(args)


if __name__ == "__main__":
    main()
