#!/usr/bin/env python3
# /// script
# dependencies = [
#   "numpy",
# ]
# ///
"""
Export an adapted mesh snapshot (*.bin) to a solver-readable .gri mesh.

Usage:
    python3 postproc/export_adapted_mesh_gri.py <mesh_cycle.bin> <output.gri>
"""

import argparse
import struct
from collections import defaultdict

import numpy as np


HO_MAGIC = 0x484F3031  # "HO01"
PG_MAGIC = 0x50473031  # "PG01"


def read_companion_mesh(path):
    with open(path, "rb") as f:
        data = f.read()

    offset = 0
    nn = struct.unpack_from("i", data, offset)[0]
    offset += 4
    nodes = np.frombuffer(data[offset: offset + 16 * nn], dtype=np.float64).reshape(nn, 2).copy()
    offset += 16 * nn

    ne = struct.unpack_from("i", data, offset)[0]
    offset += 4
    elements = []
    for _ in range(ne):
        q, v0, v1, v2 = struct.unpack_from("4i", data, offset)
        offset += 16
        elements.append({"q": q, "corners": [v0, v1, v2], "row": [v0, v1, v2]})

    bdry_edges = []
    bdry_names = {}
    nb = struct.unpack_from("i", data, offset)[0]
    offset += 4
    for _ in range(nb):
        v0, v1, bidx = struct.unpack_from("3i", data, offset)
        offset += 12
        bdry_edges.append((v0, v1, bidx))

    nnames = struct.unpack_from("i", data, offset)[0]
    offset += 4
    for i in range(nnames):
        slen = struct.unpack_from("i", data, offset)[0]
        offset += 4
        bdry_names[i] = data[offset:offset + slen].decode("utf-8")
        offset += slen

    periodic_groups = []
    if offset + 4 <= len(data):
        marker = struct.unpack_from("I", data, offset)[0]
        if marker == HO_MAGIC:
            offset += 4
            for elem in elements:
                nrow = struct.unpack_from("i", data, offset)[0]
                offset += 4
                row = list(struct.unpack_from(f"{nrow}i", data, offset))
                offset += 4 * nrow
                elem["row"] = row

    if offset + 4 <= len(data):
        marker = struct.unpack_from("I", data, offset)[0]
        if marker == PG_MAGIC:
            offset += 4
            ngroups = struct.unpack_from("i", data, offset)[0]
            offset += 4
            for _ in range(ngroups):
                type_len = struct.unpack_from("i", data, offset)[0]
                offset += 4
                pg_type = data[offset:offset + type_len].decode("utf-8")
                offset += type_len
                npairs = struct.unpack_from("i", data, offset)[0]
                offset += 4
                pairs = []
                for _ in range(npairs):
                    n0, n1 = struct.unpack_from("2i", data, offset)
                    offset += 8
                    pairs.append((n0, n1))
                periodic_groups.append({"type": pg_type, "pairs": pairs})

    return {
        "nodes": nodes,
        "elements": elements,
        "boundary_edges": bdry_edges,
        "boundary_names": bdry_names,
        "periodic_groups": periodic_groups,
    }


def write_gri(mesh, outpath):
    nodes = mesh["nodes"]
    elements = mesh["elements"]
    boundary_edges = mesh["boundary_edges"]
    boundary_names = mesh["boundary_names"]
    periodic_groups = mesh["periodic_groups"]

    boundary_groups = defaultdict(list)
    for v0, v1, bidx in boundary_edges:
        boundary_groups[bidx].append((v0, v1))

    elem_blocks = defaultdict(list)
    for elem in elements:
        elem_blocks[elem["q"]].append(elem["row"])

    with open(outpath, "w") as f:
        f.write(f"{len(nodes)} {len(elements)} 2\n")
        for x, y in nodes:
            f.write(f"{x:.15e} {y:.15e}\n")

        f.write(f"{len(boundary_groups)}\n")
        for bidx in sorted(boundary_groups):
            edges = boundary_groups[bidx]
            name = boundary_names.get(bidx, f"boundary_{bidx}")
            f.write(f"{len(edges)} 2 {name}\n")
            for v0, v1 in edges:
                f.write(f"{v0 + 1} {v1 + 1}\n")

        for q in sorted(elem_blocks):
            rows = elem_blocks[q]
            f.write(f"{len(rows)} {q} TriLagrange\n")
            for row in rows:
                f.write(" ".join(str(idx + 1) for idx in row) + "\n")

        if periodic_groups:
            f.write(f"{len(periodic_groups)} PeriodicGroup\n")
            for pg in periodic_groups:
                f.write(f"{len(pg['pairs'])} {pg['type']}\n")
                for n0, n1 in pg["pairs"]:
                    f.write(f"{n0 + 1} {n1 + 1}\n")


def main():
    parser = argparse.ArgumentParser(description="Export adapted mesh snapshot to GRI")
    parser.add_argument("snapshot", help="Path to adjoint_mesh_cycle*.bin")
    parser.add_argument("output", help="Output .gri filename")
    args = parser.parse_args()

    mesh = read_companion_mesh(args.snapshot)
    if not mesh["periodic_groups"]:
        raise SystemExit(
            "Snapshot does not contain periodic pairing metadata. "
            "Rebuild and rerun adaptation, then export from the new adjoint_mesh_cycle*.bin."
        )

    write_gri(mesh, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
