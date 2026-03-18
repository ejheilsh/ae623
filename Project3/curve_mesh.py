#!/usr/bin/env python3
"""
curve_mesh.py
=============
Convert a straight (q=1) .gri mesh to a q=2 curved mesh by projecting
wall-edge midpoints onto the blade surface CubicSpline.

Interior elements stay q=1 (3 nodes).  Wall-adjacent elements become
q=2 (6 nodes) with the standard GRI row-by-row ordering:
    v0, mid(v0,v1), v1, mid(v0,v2), mid(v1,v2), v2

Usage:
    python curve_mesh.py grids/2k.gri -o grids/2k_q2.gri
    python curve_mesh.py grids/coarse.gri --upper data/bladeupper.txt --lower data/bladelower.txt
"""

import argparse
import os
import sys
import numpy as np
from scipy.interpolate import CubicSpline


# ── GRI reader (token-stream, handles any whitespace layout) ─────────────

def read_gri_tokens(fname):
    """Read a .gri file into structured data using token-stream parsing."""
    with open(fname) as f:
        raw = f.read()
    tokens = raw.split()
    pos = [0]  # mutable index

    def tok():
        t = tokens[pos[0]]; pos[0] += 1; return t
    def tint():
        return int(tok())
    def tfloat():
        return float(tok())

    Nn, Ne, dim = tint(), tint(), tint()

    V = np.zeros((Nn, 2))
    for i in range(Nn):
        V[i, 0], V[i, 1] = tfloat(), tfloat()

    NB = tint()
    bgroups = []
    for _ in range(NB):
        nb, nn = tint(), tint()
        name = tok()
        edges = np.zeros((nb, nn), dtype=int)
        for j in range(nb):
            for k in range(nn):
                edges[j, k] = tint()
        bgroups.append((name, edges))

    # Element blocks
    elem_blocks = []
    ne0 = 0
    while ne0 < Ne:
        ne_block, deg = tint(), tint()
        etype = tok()
        npe = (deg + 1) * (deg + 2) // 2
        elems = np.zeros((ne_block, npe), dtype=int)
        for j in range(ne_block):
            for k in range(npe):
                elems[j, k] = tint()
        elem_blocks.append((deg, etype, elems))
        ne0 += ne_block

    # Remaining tokens — periodic groups etc.
    remaining = tokens[pos[0]:]

    return V, bgroups, elem_blocks, remaining


def write_gri(fname, V, bgroups, elem_blocks, trailing_tokens):
    """Write a .gri file."""
    Ne_total = sum(e.shape[0] for _, _, e in elem_blocks)
    with open(fname, "w") as f:
        f.write(f"{V.shape[0]} {Ne_total} 2\n")
        for i in range(V.shape[0]):
            f.write(f"{V[i,0]:.15e} {V[i,1]:.15e}\n")
        f.write(f"{len(bgroups)}\n")
        for name, edges in bgroups:
            nn = edges.shape[1]
            f.write(f"{edges.shape[0]} {nn} {name}\n")
            for j in range(edges.shape[0]):
                f.write(" ".join(str(edges[j, k]) for k in range(nn)) + "\n")
        for deg, etype, elems in elem_blocks:
            npe = elems.shape[1]
            f.write(f"{elems.shape[0]} {deg} {etype}\n")
            for j in range(elems.shape[0]):
                f.write(" ".join(str(elems[j, k]) for k in range(npe)) + "\n")
        # Trailing tokens (periodic groups etc.)
        if trailing_tokens:
            i = 0
            while i < len(trailing_tokens):
                # Try to reconstruct periodic group format
                if trailing_tokens[i].isdigit() and i + 1 < len(trailing_tokens) and trailing_tokens[i + 1] == "PeriodicGroup":
                    n_groups = int(trailing_tokens[i])
                    f.write(f"{n_groups} PeriodicGroup\n")
                    i += 2
                    for _ in range(n_groups):
                        n_pairs = int(trailing_tokens[i])
                        ptype = trailing_tokens[i + 1]
                        f.write(f"{n_pairs} {ptype}\n")
                        i += 2
                        for _ in range(n_pairs):
                            f.write(f"{trailing_tokens[i]} {trailing_tokens[i+1]}\n")
                            i += 2
                else:
                    f.write(trailing_tokens[i] + "\n")
                    i += 1


# ── Wall chain tracing and spline fitting (from plot_q1_q2.py) ──────────

def trace_wall_chains(wall_edges_0based):
    """Walk wall boundary edges into ordered chains."""
    if len(wall_edges_0based) == 0:
        return []
    adj = {}
    for a, b in wall_edges_0based:
        a, b = int(a), int(b)
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    visited = set()
    chains = []
    for seed in list(adj.keys()):
        if seed in visited:
            continue
        component = set()
        stack = [seed]
        while stack:
            nn = stack.pop()
            if nn in component:
                continue
            component.add(nn)
            for nb in adj[nn]:
                if nb not in component:
                    stack.append(nb)
        visited |= component
        # Find an endpoint (degree-1 node) to start ordered walk
        ep = next((nn for nn in component if len(adj[nn] & component) == 1),
                  next(iter(component)))
        ordered, prev, cur = [ep], None, ep
        while True:
            nxt_set = (adj[cur] & component) - ({prev} if prev is not None else set())
            if not nxt_set:
                break
            nxt = next(iter(nxt_set))
            ordered.append(nxt)
            prev, cur = cur, nxt
        chains.append(np.array(ordered, dtype=int))
    return chains


def chain_splines(V, chain):
    """Fit arc-length-parameterized CubicSplines along a wall chain."""
    pts = V[chain]
    ds = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    s_total = s[-1]
    if s_total < 1e-14:
        return None, None, None, None
    s_norm = s / s_total
    sx = CubicSpline(s_norm, pts[:, 0])
    sy = CubicSpline(s_norm, pts[:, 1])
    return sx, sy, s_norm, s_total


def build_curved_midpoint_lookup(V, chains):
    """For every wall edge (a,b), compute the curved midpoint by evaluating
    the arc-length spline at the midpoint parameter.
    Returns dict: (min(a,b), max(a,b)) -> (x_mid, y_mid)."""
    lookup = {}
    for chain in chains:
        sx, sy, s_norm, s_total = chain_splines(V, chain)
        if sx is None:
            continue
        node_s = {int(chain[i]): s_norm[i] for i in range(len(chain))}
        for i in range(len(chain) - 1):
            a, b = int(chain[i]), int(chain[i + 1])
            sm = 0.5 * (node_s[a] + node_s[b])
            key = (min(a, b), max(a, b))
            lookup[key] = np.array([float(sx(sm)), float(sy(sm))])
    return lookup


# ── Main curving logic ───────────────────────────────────────────────────

def curve_mesh(V, bgroups, elem_blocks):
    """Convert wall-adjacent elements to q=2 with curved midpoints.

    Returns new V, bgroups (unchanged), and new elem_blocks.
    """
    # Merge all element blocks into one flat 0-based connectivity
    all_E = []
    for deg, etype, elems in elem_blocks:
        if deg != 1:
            print(f"Warning: input already has deg={deg} block; skipping curving.")
            return V, bgroups, elem_blocks
        all_E.append(elems - 1)  # to 0-based
    E = np.vstack(all_E)
    Ne = E.shape[0]
    print(f"  Input: {V.shape[0]} nodes, {Ne} elements")

    # Find wall boundary group(s)
    wall_edges_0based = []
    wall_edge_set = set()
    for name, edges in bgroups:
        if name.lower() == "wall":
            e0 = edges - 1  # to 0-based
            wall_edges_0based.append(e0)
            for j in range(e0.shape[0]):
                a, b = int(e0[j, 0]), int(e0[j, 1])
                wall_edge_set.add((min(a, b), max(a, b)))
    if not wall_edges_0based:
        print("  No 'Wall' boundary group found; nothing to curve.")
        return V, bgroups, elem_blocks
    wall_edges_0based = np.vstack(wall_edges_0based)
    print(f"  Wall edges: {len(wall_edge_set)}")

    # Trace chains and compute curved midpoints
    chains = trace_wall_chains(wall_edges_0based)
    print(f"  Wall chains: {len(chains)} (lengths: {[len(c) for c in chains]})")
    curved_lookup = build_curved_midpoint_lookup(V, chains)
    print(f"  Curved midpoints computed: {len(curved_lookup)}")

    # Build element -> edge mapping to find wall-adjacent elements
    elem_edges = []  # for each element: list of 3 edge keys
    for e in range(Ne):
        edges_e = []
        for i in range(3):
            a, b = int(E[e, i]), int(E[e, (i + 1) % 3])
            edges_e.append((min(a, b), max(a, b)))
        elem_edges.append(edges_e)

    is_wall_adjacent = np.zeros(Ne, dtype=bool)
    for e in range(Ne):
        for ek in elem_edges[e]:
            if ek in wall_edge_set:
                is_wall_adjacent[e] = True
                break

    n_curved = int(is_wall_adjacent.sum())
    n_straight = Ne - n_curved
    print(f"  Wall-adjacent elements (q=2): {n_curved}")
    print(f"  Interior elements (q=1):      {n_straight}")

    # Build midpoint nodes for all edges of wall-adjacent elements
    # midpoint_map: edge_key -> new_node_index (0-based)
    V_list = list(V)  # will append new midpoint nodes
    midpoint_map = {}

    def get_or_create_midpoint(a, b):
        key = (min(a, b), max(a, b))
        if key in midpoint_map:
            return midpoint_map[key]
        # Curved or straight?
        if key in curved_lookup:
            mid_xy = curved_lookup[key]
        else:
            mid_xy = 0.5 * (V[a] + V[b])
        new_idx = len(V_list)
        V_list.append(mid_xy)
        midpoint_map[key] = new_idx
        return new_idx

    # Build q=2 elements (GRI row-by-row: v0, mid01, v1, mid02, mid12, v2)
    curved_elems = []
    for e in range(Ne):
        if not is_wall_adjacent[e]:
            continue
        v0, v1, v2 = int(E[e, 0]), int(E[e, 1]), int(E[e, 2])
        m01 = get_or_create_midpoint(v0, v1)
        m12 = get_or_create_midpoint(v1, v2)
        m02 = get_or_create_midpoint(v0, v2)
        # GRI row-by-row order for q=2:
        #   position 0: v0      (0, 0)
        #   position 1: mid01   (0.5, 0)
        #   position 2: v1      (1, 0)
        #   position 3: mid02   (0, 0.5)
        #   position 4: mid12   (0.5, 0.5)
        #   position 5: v2      (0, 1)
        curved_elems.append([v0 + 1, m01 + 1, v1 + 1,
                             m02 + 1, m12 + 1, v2 + 1])  # 1-based

    # Build q=1 elements (unchanged)
    straight_elems = []
    for e in range(Ne):
        if is_wall_adjacent[e]:
            continue
        straight_elems.append([int(E[e, 0]) + 1, int(E[e, 1]) + 1,
                               int(E[e, 2]) + 1])

    # Assemble output
    V_new = np.array(V_list)
    new_blocks = []
    if curved_elems:
        new_blocks.append((2, "TriLagrange", np.array(curved_elems, dtype=int)))
    if straight_elems:
        new_blocks.append((1, "TriLagrange", np.array(straight_elems, dtype=int)))

    n_new_nodes = len(V_list) - V.shape[0]
    n_curved_mids = sum(1 for k in midpoint_map if k in curved_lookup)
    print(f"  New midpoint nodes added: {n_new_nodes} "
          f"({n_curved_mids} curved, {n_new_nodes - n_curved_mids} straight)")
    print(f"  Output: {V_new.shape[0]} nodes, {Ne} elements")

    return V_new, bgroups, new_blocks


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Curve a q=1 mesh to q=2 on wall boundaries")
    parser.add_argument("input", help="Input .gri file (q=1)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output .gri file (default: input_q2.gri)")
    args = parser.parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = base + "_q2" + ext

    print(f"Reading {args.input} ...")
    V, bgroups, elem_blocks, trailing = read_gri_tokens(args.input)

    print("Curving wall-adjacent elements to q=2 ...")
    V_new, bgroups_new, blocks_new = curve_mesh(V, bgroups, elem_blocks)

    print(f"Writing {args.output} ...")
    write_gri(args.output, V_new, bgroups_new, blocks_new, trailing)
    print("Done.")


if __name__ == "__main__":
    main()
