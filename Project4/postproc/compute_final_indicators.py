#!/usr/bin/env python3
"""
Compute adjoint-weighted error indicators on fixed final grids.

This script runs euler_solver from a temporary working directory so the solver's
normal steady outputs do not land in this project's data_steady directory.
The requested indicator binaries are written to final_cases/indicators/.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


def natural_key(value: str) -> list[object]:
    parts = re.split(r"(\d+)", value)
    return [int(part) if part.isdigit() else part for part in parts]


def infer_iteration(path: Path) -> int:
    match = re.fullmatch(r"iter(\d+)_soln_dg\.bin", path.name)
    if not match:
        raise ValueError(f"Cannot infer iteration from {path}")
    return int(match.group(1))


def case_prefix(ncells: str, curvature: str, solver_order: str) -> str:
    if curvature == "q1":
        return f"steady_{ncells}_{solver_order}_"
    return f"steady_{ncells}_{curvature}_{solver_order}_"


def normalize_curvature_filter(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip("/")
    if re.fullmatch(r"\d+", value):
        value = f"q{value}"
    if not re.fullmatch(r"q\d+", value):
        raise ValueError("curvature must look like q1, q2, q3, or a number such as 1")
    return value


def discover_cases(
    final_grids: Path,
    final_solutions: Path,
    case_filter: str | None,
    curvature_filter: str | None,
    iteration: int | None,
    latest_only: bool,
    skip_if_any_indicator_roots: list[Path],
) -> list[tuple[str, int, Path]]:
    by_case: dict[str, list[tuple[int, Path]]] = {}
    for dg_file in final_solutions.glob("*/*/*/iter*_soln_dg.bin"):
        rel = dg_file.relative_to(final_solutions)
        case = "/".join(rel.parts[:3])
        if case_filter is not None and case != case_filter.strip("/"):
            continue
        if curvature_filter is not None and rel.parts[1] != curvature_filter:
            continue
        iter_num = infer_iteration(dg_file)
        if iteration is not None and iter_num != iteration:
            continue
        ncells, curvature, solver_order = rel.parts[:3]
        meshfile = final_grids / ncells / curvature / solver_order / f"iter{iter_num}.gri"
        if not meshfile.exists():
            print(f"[SKIP] missing mesh for solved iteration: {meshfile}")
            continue
        by_case.setdefault(case, []).append((iter_num, dg_file))

    selected: list[tuple[str, int, Path]] = []
    for case, entries in sorted(by_case.items(), key=lambda item: natural_key(item[0])):
        entries.sort(key=lambda item: item[0])
        ncells, curvature, solver_order = case.split("/")
        prefix = case_prefix(ncells, curvature, solver_order)
        if skip_if_any_indicator_roots:
            existing_roots = [
                root
                for root in skip_if_any_indicator_roots
                if any(root.glob(f"{prefix}adjoint_indicators_cycle*.bin"))
            ]
            if existing_roots and latest_only and iteration is None:
                roots_text = ", ".join(str(root) for root in existing_roots)
                print(f"[SKIP] {case}: existing adaptation indicators in {roots_text}")
                continue
        if latest_only and iteration is None:
            iter_num, dg_file = entries[-1]
            selected.append((case, iter_num, dg_file))
        else:
            selected.extend((case, iter_num, dg_file) for iter_num, dg_file in entries)
    return selected


def run_one(
    solver: Path,
    final_grids: Path,
    outdir: Path,
    case: str,
    iteration: int,
    dg_file: Path,
    cfl: float,
    flux: str,
    skip_existing: bool,
) -> bool:
    ncells, curvature, solver_order = case.split("/")
    p_order = int(solver_order.removeprefix("p"))
    meshfile = final_grids / ncells / curvature / solver_order / f"iter{iteration}.gri"
    if not meshfile.exists():
        print(f"[SKIP] missing mesh: {meshfile}")
        return False

    indicator_file = outdir / f"{case_prefix(ncells, curvature, solver_order)}adjoint_indicators_cycle{iteration}.bin"
    log_file = outdir / ncells / curvature / solver_order / f"iter{iteration}_indicator.log"
    indicator_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    if skip_existing and indicator_file.exists():
        print(f"[SKIP] {indicator_file}")
        return True

    cmd = [
        str(solver),
        str(meshfile.resolve()),
        str(p_order),
        str(cfl),
        flux,
        "0",
        "steady",
        str(dg_file.resolve()),
        "--write-adjoint-indicators",
        str(indicator_file.resolve()),
    ]

    print(f"[RUN] {case} iter{iteration} -> {indicator_file}")
    with tempfile.TemporaryDirectory(prefix="project4_indicators_") as tmpdir:
        proc = subprocess.run(
            cmd,
            cwd=tmpdir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    log_file.write_text(proc.stdout)
    if proc.returncode != 0:
        print(f"[FAIL] {case} iter{iteration}; see {log_file}")
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute fixed-mesh adjoint indicators for final grids."
    )
    parser.add_argument("--solver", default="./euler_solver")
    parser.add_argument("--final-grids", default="final_grids")
    parser.add_argument("--final-solutions", default="final_solutions")
    parser.add_argument("--outdir", default="final_cases/indicators")
    parser.add_argument("--case", default=None, help="Optional case filter like 8k/q3/p0")
    parser.add_argument("--curvature", default=None, help="Optional curvature filter like q1")
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--all-iterations", action="store_true")
    parser.add_argument("--cfl", type=float, default=1.0)
    parser.add_argument("--flux", default="roe")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--skip-if-any-indicator-root",
        action="append",
        default=[],
        type=Path,
        help=(
            "Skip latest-only computation for a case if this root already "
            "contains any matching case indicator files. Repeatable."
        ),
    )
    args = parser.parse_args()

    solver = Path(args.solver)
    if not solver.exists():
        alt = Path("./euler_solver.exe")
        if args.solver == "./euler_solver" and alt.exists():
            solver = alt
        else:
            raise SystemExit(f"Solver binary not found: {solver}")
    solver = solver.resolve()

    cases = discover_cases(
        Path(args.final_grids),
        Path(args.final_solutions),
        args.case,
        normalize_curvature_filter(args.curvature),
        args.iteration,
        latest_only=not args.all_iterations,
        skip_if_any_indicator_roots=args.skip_if_any_indicator_root,
    )
    if not cases:
        print("No fixed-grid indicator computations needed.")
        return

    outdir = Path(args.outdir)
    if outdir.exists():
        shutil.rmtree(outdir / "_tmp", ignore_errors=True)
    ok = 0
    for case, iteration, dg_file in cases:
        if run_one(
            solver,
            Path(args.final_grids),
            outdir,
            case,
            iteration,
            dg_file,
            args.cfl,
            args.flux,
            args.skip_existing,
        ):
            ok += 1
    print(f"Computed {ok}/{len(cases)} indicator file(s)")


if __name__ == "__main__":
    main()
