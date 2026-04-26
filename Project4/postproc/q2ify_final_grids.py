#!/usr/bin/env python3
"""
Create q2 final_grids from q1 final_grids.

This walks:

    final_grids/<ncells>/q1/<solver_order>/*.gri

and writes matching lifted meshes to:

    final_grids/<ncells>/q2/<solver_order>/*.gri

The actual wall-geometry lifting is delegated to curve_mesh.py so this uses the
same spline projection convention as the repository's base q2/q3 meshes.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from curve_mesh import curve_mesh, read_gri_tokens, write_gri  # noqa: E402


def normalize_solver_order(value: str) -> str:
    if re.fullmatch(r"\d+", value):
        value = f"p{value}"
    if not re.fullmatch(r"p\d+", value):
        raise argparse.ArgumentTypeError(
            "solver order must look like p0, p1, p2, or a number such as 0"
        )
    return value


def normalize_target(value: str) -> str:
    if re.fullmatch(r"\d+", value):
        value = f"q{value}"
    if value not in {"q2", "q3"}:
        raise argparse.ArgumentTypeError("target curvature must be q2, q3, 2, or 3")
    return value


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(path))]


def iter_source_cases(final_grids: Path, ncells: str | None, order: str | None):
    if ncells is not None and order is not None:
        case_dir = final_grids / ncells / "q1" / order
        if case_dir.is_dir():
            yield ncells, order, case_dir
        return

    if ncells is not None:
        for case_dir in sorted((final_grids / ncells / "q1").glob("p*"), key=natural_key):
            if case_dir.is_dir() and (order is None or case_dir.name == order):
                yield ncells, case_dir.name, case_dir
        return

    for case_dir in sorted(final_grids.glob("*/q1/p*"), key=natural_key):
        if not case_dir.is_dir():
            continue
        case_ncells = case_dir.parts[-3]
        case_order = case_dir.name
        if order is not None and case_order != order:
            continue
        yield case_ncells, case_order, case_dir


def is_q1_only(elem_blocks) -> bool:
    return all(degree == 1 for degree, _, _ in elem_blocks)


def convert_one(source: Path, dest: Path, q_target: int, overwrite: bool, dry_run: bool) -> str:
    if dest.exists() and not overwrite:
        return "skipped"

    if dry_run:
        return "would-write"

    verts, bgroups, elem_blocks, trailing = read_gri_tokens(source)
    if not is_q1_only(elem_blocks):
        raise ValueError(f"{source} is not q1-only; refusing to q2ify it")

    verts_new, bgroups_new, blocks_new = curve_mesh(
        verts, bgroups, elem_blocks, q_target=q_target
    )

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    write_gri(tmp, verts_new, bgroups_new, blocks_new, trailing)
    tmp.replace(dest)
    return "converted"


def prune_stale_targets(source_dir: Path, dest_dir: Path, dry_run: bool) -> int:
    source_names = {path.name for path in source_dir.glob("*.gri")}
    if not dest_dir.exists():
        return 0
    removed = 0
    for target in sorted(dest_dir.glob("*.gri"), key=natural_key):
        if target.name in source_names:
            continue
        removed += 1
        if dry_run:
            print(f"[DRY]  would remove stale target {target}")
        else:
            target.unlink()
            print(f"[PRUNE] removed stale target {target}")
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch-lift final_grids q1 meshes into q2/q3 wall geometry."
    )
    parser.add_argument(
        "ncells",
        nargs="?",
        help="Optional mesh-size case to convert, e.g. 2k or 8k.",
    )
    parser.add_argument(
        "solver_order",
        nargs="?",
        type=normalize_solver_order,
        help="Optional solver-order case to convert, e.g. p0 or 0.",
    )
    parser.add_argument(
        "--final-grids",
        type=Path,
        default=ROOT / "final_grids",
        help="Root final_grids directory.",
    )
    parser.add_argument(
        "--target",
        type=normalize_target,
        default="q2",
        help="Target curvature directory/order: q2 or q3. Default: q2.",
    )
    parser.add_argument(
        "--iter",
        dest="iteration",
        help="Only convert iter<N>.gri for this iteration number.",
    )
    parser.add_argument(
        "--include-latest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also convert latest_accepted.gri when present. Default: true.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing target grids.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be converted without writing files.",
    )
    parser.add_argument(
        "--prune-stale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Remove target q2/q3 grids that no longer have a matching q1 source "
            "when converting a full case. Default: true."
        ),
    )
    args = parser.parse_args()

    if args.iteration is not None:
        iter_text = args.iteration
        if iter_text.startswith("iter"):
            iter_text = iter_text[4:]
        if not re.fullmatch(r"\d+", iter_text):
            parser.error("--iter must be an integer or iter<N>")
        wanted_names = {f"iter{iter_text}.gri"}
    else:
        wanted_names = None

    q_target = int(args.target[1:])
    converted = 0
    skipped = 0
    would_write = 0
    failed = 0
    pruned = 0

    cases = list(iter_source_cases(args.final_grids, args.ncells, args.solver_order))
    if not cases:
        print("No q1 final_grid cases found.")
        return 1

    for ncells, solver_order, source_dir in cases:
        dest_dir = args.final_grids / ncells / args.target / solver_order
        if args.prune_stale and wanted_names is None:
            pruned += prune_stale_targets(source_dir, dest_dir, args.dry_run)
        sources = sorted(source_dir.glob("*.gri"), key=natural_key)
        for source in sources:
            if wanted_names is not None and source.name not in wanted_names:
                continue
            if source.name == "latest_accepted.gri" and not args.include_latest:
                continue

            dest = dest_dir / source.name
            try:
                status = convert_one(source, dest, q_target, args.overwrite, args.dry_run)
            except Exception as exc:  # keep batch conversion going
                failed += 1
                print(f"[FAIL] {source} -> {dest}: {exc}", file=sys.stderr)
                continue

            if status == "converted":
                converted += 1
                print(f"[OK]   {source} -> {dest}")
            elif status == "would-write":
                would_write += 1
                print(f"[DRY]  {source} -> {dest}")
            else:
                skipped += 1
                print(f"[SKIP] {dest} exists")

    print(
        "Summary: "
        f"converted={converted}, skipped={skipped}, dry_run={would_write}, "
        f"pruned={pruned}, failed={failed}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
