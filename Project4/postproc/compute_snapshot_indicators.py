#!/usr/bin/env python3
"""Compute indicators for companion mesh snapshots such as plotting_extras/*.bin."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

from export_adapted_mesh_gri import read_companion_mesh, write_gri


SNAPSHOT_RE = re.compile(
    r"(?P<prefix>steady_(?P<ncells>[^_]+)(?:_(?P<curvature>q\d+))?_(?P<order>p\d+)_"
    r")adjoint_mesh_cycle(?P<cycle>\d+)\.bin$"
)


def natural_key(value: str) -> list[object]:
    parts = re.split(r"(\d+)", value)
    return [int(part) if part.isdigit() else part for part in parts]


def parse_snapshot(path: Path) -> dict | None:
    match = SNAPSHOT_RE.fullmatch(path.name)
    if match is None:
        return None
    return {
        "prefix": match.group("prefix"),
        "ncells": match.group("ncells"),
        "curvature": match.group("curvature") or "q1",
        "order": match.group("order"),
        "cycle": int(match.group("cycle")),
    }


def discover_snapshots(root: Path, case: str | None) -> list[tuple[Path, dict]]:
    found = []
    for path in root.glob("*_adjoint_mesh_cycle*.bin"):
        parsed = parse_snapshot(path)
        if parsed is None:
            continue
        snapshot_case = f"{parsed['ncells']}/{parsed['curvature']}/{parsed['order']}"
        if case is not None and snapshot_case != case.strip("/"):
            continue
        found.append((path, parsed))
    return sorted(found, key=lambda item: natural_key(item[0].name))


def select_cycles(
    snapshots: list[tuple[Path, dict]],
    requested_cycles: list[int] | None,
) -> list[tuple[Path, dict]]:
    if requested_cycles:
        wanted = set(requested_cycles)
        return [(path, parsed) for path, parsed in snapshots if parsed["cycle"] in wanted]
    if not snapshots:
        return []
    by_cycle = {parsed["cycle"]: (path, parsed) for path, parsed in snapshots}
    return [by_cycle[min(by_cycle)], by_cycle[max(by_cycle)]]


def export_snapshot(snapshot: Path, out_gri: Path) -> None:
    out_gri.parent.mkdir(parents=True, exist_ok=True)
    mesh = read_companion_mesh(snapshot)
    write_gri(mesh, out_gri)


def run_indicator_case(
    solver: Path,
    snapshot: Path,
    parsed: dict,
    root: Path,
    grid_dir: Path,
    run_root: Path,
    cfl: float,
    flux: str,
    itercap: int,
    skip_existing: bool,
    dry_run: bool,
) -> bool:
    cycle = parsed["cycle"]
    p_order = int(parsed["order"].removeprefix("p"))
    grid_file = grid_dir / f"{snapshot.stem}.gri"
    indicator_file = root / f"{parsed['prefix']}adjoint_indicators_cycle{cycle}.bin"
    run_dir = run_root / f"{snapshot.stem}"
    log_file = run_dir / "indicator_solve.log"

    if skip_existing and indicator_file.exists():
        print(f"[SKIP] {indicator_file}")
        return True

    export_snapshot(snapshot, grid_file)
    run_dir.mkdir(parents=True, exist_ok=True)
    indicator_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(solver.resolve()),
        str(grid_file.resolve()),
        str(p_order),
        str(cfl),
        flux,
        str(itercap),
        "steady",
        "--write-adjoint-indicators",
        str(indicator_file.resolve()),
    ]

    print(f"[RUN] cycle {cycle}: {snapshot.name} -> {indicator_file}")
    print("      " + " ".join(cmd))
    if dry_run:
        return True

    proc = subprocess.run(
        cmd,
        cwd=run_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_file.write_text(proc.stdout)
    if proc.returncode != 0:
        print(f"[FAIL] cycle {cycle}; see {log_file}")
        return False
    print(f"[OK] cycle {cycle}; log/results under {run_dir}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate fixed-mesh adjoint indicators from solver companion mesh "
            "snapshots. By default, only the first and last matching cycles are run."
        )
    )
    parser.add_argument("--root", default="plotting_extras", help="Snapshot/indicator directory")
    parser.add_argument("--solver", default="./euler_solver")
    parser.add_argument("--case", default=None, help="Optional case filter, e.g. 2k/q3/p1")
    parser.add_argument("--cycle", type=int, action="append", default=None, help="Cycle to run; repeatable")
    parser.add_argument("--cfl", type=float, default=1.0)
    parser.add_argument("--flux", default="roe")
    parser.add_argument("--itercap", type=int, default=1_000_000)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    solver = Path(args.solver)
    if not solver.exists():
        raise SystemExit(f"Solver not found: {solver}")

    snapshots = discover_snapshots(root, args.case)
    selected = select_cycles(snapshots, args.cycle)
    if not selected:
        raise SystemExit(f"No matching companion mesh snapshots found in {root}")

    grid_dir = root / "generated_grids"
    run_root = root / "solver_runs"
    ok = 0
    for snapshot, parsed in selected:
        if run_indicator_case(
            solver=solver,
            snapshot=snapshot,
            parsed=parsed,
            root=root,
            grid_dir=grid_dir,
            run_root=run_root,
            cfl=args.cfl,
            flux=args.flux,
            itercap=args.itercap,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
        ):
            ok += 1
    print(f"Computed {ok}/{len(selected)} requested indicator file(s)")


if __name__ == "__main__":
    main()
