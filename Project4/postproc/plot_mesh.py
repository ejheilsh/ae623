#!/usr/bin/env python3
# /// script
# dependencies = [
#   "matplotlib",
#   "numpy",
# ]
# ///
"""
Plot a single mesh snapshot with boundary coloring.

Usage:
    python postproc/plot_mesh.py <mesh_file.bin> [indicators.bin] [--output plot.png]
"""

import argparse
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection

def read_companion_mesh(filename):
    with open(filename, "rb") as f:
        data = f.read()
    offset = 0
    nn = struct.unpack_from("i", data, offset)[0]; offset += 4
    nodes = np.frombuffer(data[offset : offset + 8 * 2 * nn], dtype=np.float64).reshape(nn, 2)
    offset += 8 * 2 * nn
    ne = struct.unpack_from("i", data, offset)[0]; offset += 4
    elements = []
    for _ in range(ne):
        q, v0, v1, v2 = struct.unpack_from("iiii", data, offset); offset += 16
        elements.append([v0, v1, v2])
    
    bdry_edges = []
    bdry_names = {} # index -> name
    if offset < len(data):
        nb = struct.unpack_from("i", data, offset)[0]; offset += 4
        for _ in range(nb):
            v0, v1, bidx = struct.unpack_from("iii", data, offset); offset += 12
            bdry_edges.append((v0, v1, bidx))
        
        if offset < len(data):
            nnames = struct.unpack_from("i", data, offset)[0]; offset += 4
            for i in range(nnames):
                slen = struct.unpack_from("i", data, offset)[0]; offset += 4
                name = data[offset:offset+slen].decode("utf-8")
                bdry_names[i] = name.lower()
                offset += slen
                
    return nodes, elements, bdry_edges, bdry_names

def read_indicators(filename):
    with open(filename, "rb") as f:
        ne = struct.unpack("i", f.read(4))[0]
        return np.frombuffer(f.read(8 * ne), dtype=np.float64)

def main():
    parser = argparse.ArgumentParser(description="Plot a single mesh snapshot.")
    parser.add_argument("mesh", help="Path to adjoint_mesh_cycleN.bin")
    parser.add_argument("indicators", nargs="?", help="Optional path to indicators.bin")
    parser.add_argument("-o", "--output", default="mesh_snapshot.png", help="Output image file")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    nodes, elements, bdry_edges, bdry_names = read_companion_mesh(args.mesh)
    verts = [nodes[e] for e in elements]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if args.indicators:
        inds = read_indicators(args.indicators)
        ind_log = np.log10(np.maximum(inds, 1e-30))
        pc = PolyCollection(verts, cmap="hot", edgecolors="black", linewidths=0.1)
        pc.set_array(ind_log)
        ax.add_collection(pc)
        plt.colorbar(pc, ax=ax, label="log10(indicator)")
    else:
        pc = PolyCollection(verts, facecolors="white", edgecolors="gray", linewidths=0.25)
        ax.add_collection(pc)

    # Boundary coloring logic — dynamic mapping based on names
    def get_color_info(bidx, name):
        if "wall" in name: return "crimson", "Wall"
        if "inflow" in name: return "dodgerblue", "Inflow"
        if "outflow" in name: return "forestgreen", "Outflow"
        # Fallback to standard 2k.gri indices if names weren't found
        if not name:
            if bidx == 0: return "dodgerblue", "Inflow"
            if bidx == 1: return "forestgreen", "Outflow"
            if bidx == 2: return "crimson", "Wall"
        return "orange", f"Group {bidx} ({name})"

    groups = {}
    for v0, v1, bidx in bdry_edges:
        if bidx not in groups:
            groups[bidx] = {"segs": [], "name": bdry_names.get(bidx, "")}
        groups[bidx]["segs"].append([nodes[v0], nodes[v1]])

    for bidx, info in groups.items():
        color, label = get_color_info(bidx, info["name"])
        lc = LineCollection(info["segs"], colors=color, linewidths=2.0, label=label)
        ax.add_collection(lc)

    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_title(f"Mesh Snapshot: {os.path.basename(args.mesh)}")
    ax.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=args.dpi)
    print(f"Saved plot to {args.output}")

if __name__ == "__main__":
    main()
