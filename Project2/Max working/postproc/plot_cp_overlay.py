#!/usr/bin/env python3
"""
plot_cp_overlay.py

Overlay Cp distributions (upper/lower surfaces) for one or more solution files.

Usage:
  python plot_cp_overlay.py <mesh.gri> <out.png> <results1.bin> <label1> [<results2.bin> <label2> ...]
Example:
  python postproc/plot_cp_overlay.py grids/2k.gri cp_overlay.png data_steady/steady_2k_o1_roe_results.bin "1st-order Roe"

  python postproc/plot_cp_overlay.py grids/2k.gri cp_overlay.png data_steady/steady_2k_o1_roe_results.bin "1st-order Roe" data_steady/steady_2k_o2_roe_results.bin "2nd-order Roe"

  ***NOT WORKING***
  python postproc/plot_cp_overlay.py grids/2k.gri cp_overlay.png data_steady/steady_2k_o1_roe_results.bin "1st-order Roe" data_steady/steady_2k_o2_roe_results.bin "2nd-order Roe" data_steady/steady_2k_o1_hlle_results.bin "1st-order HLLE"

  python postproc/plot_cp_overlay.py grids/2k.gri cp_overlay.png data_steady/steady_2k_o1_roe_results.bin "1st-order Roe" data_steady/steady_2k_o2_roe_results.bin "2nd-order Roe" data_steady/steady_2k_o1_hlle_results.bin "1st-order HLLE" data_steady/steady_2k_o2_hlle_results.bin "2nd-order HLLE"

  python postproc/plot_cp_overlay.py grids/2k.gri cp_overlay.pngdata_steady/steady_2k_o1_hlle_results.bin "1st-order HLLE" data_steady/steady_2k_o2_hlle_results.bin "2nd-order HLLE"

Notes:
- Wall boundary being:
  (a) two open chains (upper/lower) with 4 endpoints total, or
  (b) a single closed cycle.
- Cp definition:
    cp = (p - p_out) / q_out
    p_out = 0.7 * p0
    p0 = a0^2 * rho0 / gamma, with rho0=a0=1.0
    M_out^2 = (2/(gamma-1)) * [ (p0/p_out)^((gamma-1)/gamma) - 1 ]
    q_out = 0.5 * gamma * p_out * M_out^2
"""

import sys
import struct
import numpy as np
import matplotlib.pyplot as plt


def read_results_bin(filename: str) -> np.ndarray:
    with open(filename, "rb") as f:
        ne = struct.unpack("i", f.read(4))[0]
        data = struct.unpack("d" * (4 * ne), f.read(8 * 4 * ne))
    return np.array(data, dtype=float).reshape(ne, 4)


def read_gri_mesh(meshfile: str):
    """
    Returns:
      V: (Nn,2) float
      E: (Ne,3) int (0-based)
      boundary_edges: dict[str -> list[(n1,n2)]], node indices 0-based
    Handles boundary entries in either:
      - node-pair format, or
      - elem/face format (triangles, face in {1,2,3}).
    """
    with open(meshfile, "r") as f:
        header = f.readline().split()
        if len(header) < 3:
            raise RuntimeError("Invalid .gri header.")
        nn, ne, dim = map(int, header[:3])
        if dim != 2:
            raise RuntimeError(f"Expected dim=2, got dim={dim}")

        V = np.zeros((nn, 2), dtype=float)
        for i in range(nn):
            V[i, 0], V[i, 1] = map(float, f.readline().split())

        nb_line = f.readline().split()
        if not nb_line:
            raise RuntimeError("Missing NB line.")
        nb = int(nb_line[0])

        # Read raw boundary blocks first
        Bname = []
        Bnnode = []
        Braw = []  # list of list-of-rows, each row is list[int] length Bnnode
        for _ in range(nb):
            parts = f.readline().split()
            if len(parts) < 3:
                raise RuntimeError("Invalid boundary header line.")
            nseg = int(parts[0])
            bnn = int(parts[1])
            bname = parts[2]
            Bname.append(bname)
            Bnnode.append(bnn)
            rows = []
            for _s in range(nseg):
                row = list(map(int, f.readline().split()))
                if len(row) < bnn:
                    raise RuntimeError(f"Boundary row too short for {bname}.")
                rows.append(row[:bnn])
            Braw.append(rows)

        # Read element blocks until we collect ne elements
        E = []
        ne_read = 0
        while ne_read < ne:
            line = f.readline()
            if not line:
                break
            parts = line.split()
            if len(parts) < 3:
                continue
            nblk = int(parts[0])
            # degree = int(parts[1])   # unused
            # etype = parts[2]         # unused
            for _ in range(nblk):
                tri = list(map(int, f.readline().split()))
                if len(tri) < 3:
                    raise RuntimeError("Invalid element connectivity line.")
                # 1-based -> 0-based
                E.append([tri[0] - 1, tri[1] - 1, tri[2] - 1])
            ne_read += nblk

        if ne_read != ne:
            raise RuntimeError(f"Read {ne_read} elements, expected {ne}.")

    E = np.array(E, dtype=int)

    # Convert boundaries to node-pair edges (0-based)
    boundary_edges = {}
    for bname, bnn, rows in zip(Bname, Bnnode, Braw):
        edges = []

        # Decide if elem/face format
        is_elem_face = (bnn == 2 and len(rows) > 0)
        if is_elem_face:
            # check if any second >3 or any first >Ne => NOT elem/face
            for r in rows:
                if r[1] > 3 or r[0] > E.shape[0] or r[0] < 1 or r[1] < 1:
                    is_elem_face = False
                    break

        if is_elem_face:
            # rows are (elem_index, face_index), both 1-based
            for (elem_i_1b, face_i_1b) in rows:
                elem = elem_i_1b - 1
                face = face_i_1b - 1  # 0,1,2
                # edge is between the other two vertices
                n1 = E[elem, (face + 1) % 3]
                n2 = E[elem, (face + 2) % 3]
                edges.append((int(n1), int(n2)))
        else:
            # rows are node indices (1-based) (or higher-order, we take first two)
            for r in rows:
                n1 = r[0] - 1
                n2 = r[1] - 1
                edges.append((int(n1), int(n2)))

        boundary_edges[bname] = edges

    return V, E, boundary_edges


# -----------------------------
# Physics helpers
# -----------------------------
def compute_pressure(U: np.ndarray, gamma: float) -> np.ndarray:
    rho = U[:, 0]
    rhou = U[:, 1]
    rhov = U[:, 2]
    rhoE = U[:, 3]
    u = rhou / rho
    v = rhov / rho
    qsq = u * u + v * v
    p = (gamma - 1.0) * (rhoE - 0.5 * rho * qsq)
    return p


def compute_reference_conditions(gamma: float, pout_ratio: float = 0.7):
    # per project setup: rho0=a0=1.0
    rho0 = 1.0
    a0 = 1.0
    p0 = (a0 * a0 * rho0) / gamma
    p_out = pout_ratio * p0

    pr = p0 / p_out
    M_out_sq = (2.0 / (gamma - 1.0)) * (pr ** ((gamma - 1.0) / gamma) - 1.0)
    q_out = 0.5 * gamma * p_out * M_out_sq
    return p0, p_out, np.sqrt(M_out_sq), q_out


# wall ordering
def _unique_undirected_edges(edges):
    out = []
    seen = set()
    for a, b in edges:
        if a == b:
            continue
        k = (a, b) if a < b else (b, a)
        if k in seen:
            continue
        seen.add(k)
        out.append((a, b))
    return out


def get_ordered_wall_paths(V: np.ndarray, wall_edges):
    """
    Returns a list of node-paths, each path is a list[int] of nodes ordered along the boundary.
    Each component becomes either:
      - open chain (endpoints degree 1)
      - closed cycle (all degree 2), returned with path[0]==path[-1]
    """
    edges = _unique_undirected_edges(wall_edges)

    # adjacency
    adj = {}
    nodes = set()
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
        nodes.add(a)
        nodes.add(b)

    degs = {n: len(adj.get(n, [])) for n in nodes}
    degree_counts = {}
    for d in degs.values():
        degree_counts[d] = degree_counts.get(d, 0) + 1

    # connected components
    unvisited = set(nodes)
    components = []
    while unvisited:
        s = next(iter(unvisited))
        stack = [s]
        unvisited.remove(s)
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj.get(u, []):
                if v in unvisited:
                    unvisited.remove(v)
                    stack.append(v)
        components.append(comp)

    # print(f"[wall debug] edges={len(edges)}, nodes={len(nodes)}, degree_counts={degree_counts}")
    # print(f"[wall debug] components={len(components)} sizes={[len(c) for c in components]}")

    def walk_chain(start):
        path = [start]
        prev = None
        cur = start
        guard = 0
        while True:
            nbrs = adj.get(cur, [])
            nxts = [x for x in nbrs if x != prev]
            if len(nxts) == 0:
                break
            if len(nxts) > 1:
                raise RuntimeError(f"Branching detected at node {cur}; boundary not a simple chain.")
            nxt = nxts[0]
            path.append(nxt)
            prev, cur = cur, nxt
            guard += 1
            if guard > 10 * len(nodes):
                raise RuntimeError("Chain walk guard tripped.")
        return path

    def walk_cycle(start):
        nbrs = adj.get(start, [])
        if len(nbrs) != 2:
            raise RuntimeError("Cannot walk cycle from non-degree-2 node.")
        # pick one neighbor and traverse until back
        path = [start, nbrs[0]]
        prev, cur = start, nbrs[0]
        guard = 0
        while True:
            nxts = [x for x in adj[cur] if x != prev]
            if len(nxts) != 1:
                raise RuntimeError(f"Cycle not simple at node {cur}.")
            nxt = nxts[0]
            if nxt == start:
                break
            path.append(nxt)
            prev, cur = cur, nxt
            guard += 1
            if guard > 10 * len(nodes):
                raise RuntimeError("Cycle walk guard tripped.")
        path.append(start)  # close explicitly
        return path

    ordered_paths = []
    for comp in components:
        comp_degs = {n: degs[n] for n in comp}
        endpoints = [n for n, d in comp_degs.items() if d == 1]
        is_cycle = (len(endpoints) == 0 and all(d == 2 for d in comp_degs.values()))

        if is_cycle:
            start = min(comp, key=lambda i: V[i, 0])
            cyc = walk_cycle(start)
            ordered_paths.append(cyc)
        else:
            if len(endpoints) >= 2:
                start = min(endpoints, key=lambda i: V[i, 0])
            else:
                # fallback: start at min-x node
                start = min(comp, key=lambda i: V[i, 0])
            ch = walk_chain(start)
            ordered_paths.append(ch)

    # orient open chains from LE-->TE (increasing x)
    for i, p in enumerate(ordered_paths):
        if len(p) >= 2 and p[0] != p[-1]:
            if V[p[0], 0] > V[p[-1], 0]:
                ordered_paths[i] = list(reversed(p))

    ordered_paths.sort(key=lambda p: len(p), reverse=True)
    return ordered_paths


def split_cycle_into_two_paths(V: np.ndarray, cycle_path):
    """
    cycle_path: list[int] with cycle_path[0] == cycle_path[-1]
    Returns two node paths from LE->TE.
    """
    cyc = cycle_path[:-1]  # drop duplicate end
    nodes = list(dict.fromkeys(cyc))

    le = min(nodes, key=lambda i: V[i, 0])
    te = max(nodes, key=lambda i: V[i, 0])

    # adjacency on this cycle (only edges along the cycle ordering)
    adj = {}
    for i in range(len(cyc)):
        a = cyc[i]
        b = cyc[(i + 1) % len(cyc)]
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    if len(adj.get(le, [])) != 2 or len(adj.get(te, [])) != 2:
        raise RuntimeError("Cycle split failed: LE/TE not degree-2 on cycle.")

    def walk(first_neighbor):
        path = [le, first_neighbor]
        prev, cur = le, first_neighbor
        guard = 0
        while cur != te:
            nxts = [x for x in adj[cur] if x != prev]
            if len(nxts) != 1:
                raise RuntimeError(f"Non-simple cycle split at node {cur}.")
            nxt = nxts[0]
            path.append(nxt)
            prev, cur = cur, nxt
            guard += 1
            if guard > 10 * len(nodes):
                raise RuntimeError("Cycle split guard tripped.")
        return path

    n0, n1 = adj[le][0], adj[le][1]
    pA = walk(n0)
    pB = walk(n1)

    # orient both LE-->TE by x
    if V[pA[0], 0] > V[pA[-1], 0]:
        pA = list(reversed(pA))
    if V[pB[0], 0] > V[pB[-1], 0]:
        pB = list(reversed(pB))

    return pA, pB


def mean_mid_y(V: np.ndarray, path):
    if path is None or len(path) < 2:
        return 0.0
    ys = []
    for i in range(len(path) - 1):
        mid = 0.5 * (V[path[i]] + V[path[i + 1]])
        ys.append(mid[1])
    return float(np.mean(ys)) if ys else 0.0


def build_edge_to_elem(E: np.ndarray):
    """
    Returns dict[(min(n1,n2),max(n1,n2))] -> list[elem_idx]
    """
    edge_to_elem = {}
    for ei, tri in enumerate(E):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for n1, n2 in ((a, b), (b, c), (c, a)):
            k = (n1, n2) if n1 < n2 else (n2, n1)
            edge_to_elem.setdefault(k, []).append(ei)
    return edge_to_elem


# -----------------------------
# Cp & forces
# -----------------------------
def cp_along_path(V, edge_to_elem, p_cell, p_out, q_out, path, x0, chord):
    xs = []
    cps = []
    for i in range(len(path) - 1):
        n1 = int(path[i])
        n2 = int(path[i + 1])
        k = (n1, n2) if n1 < n2 else (n2, n1)
        if k not in edge_to_elem or len(edge_to_elem[k]) == 0:
            continue
        ei = edge_to_elem[k][0]
        p = p_cell[ei]
        cp = (p - p_out) / q_out
        xmid = 0.5 * (V[n1, 0] + V[n2, 0])
        xs.append((xmid - x0) / chord)
        cps.append(cp)
    xs = np.array(xs, dtype=float)
    cps = np.array(cps, dtype=float)
    # ensure increasing x for plotting
    if xs.size > 1:
        order = np.argsort(xs)
        xs = xs[order]
        cps = cps[order]
    return xs, cps


def compute_forces(V, E, wall_edges, edge_to_elem, p_cell, q_out, chord):
    """
    Integrate pressure forces over wall edges
    Normal is oriented outward using the adjacent element centroid test.
    """
    edges = _unique_undirected_edges(wall_edges)
    Fx = 0.0
    Fy = 0.0

    for a, b in edges:
        n1 = int(a)
        n2 = int(b)
        k = (n1, n2) if n1 < n2 else (n2, n1)
        if k not in edge_to_elem or len(edge_to_elem[k]) == 0:
            continue
        ei = edge_to_elem[k][0]
        p = float(p_cell[ei])

        p1 = V[n1]
        p2 = V[n2]
        edge_vec = p2 - p1
        ds = float(np.linalg.norm(edge_vec))
        if ds <= 0.0:
            continue

        # candidate normal
        n = np.array([edge_vec[1], -edge_vec[0]], dtype=float)
        n /= np.linalg.norm(n)

        # flip to point outward: compare with vector from cell centroid to edge midpoint
        tri = E[ei]
        centroid = (V[tri[0]] + V[tri[1]] + V[tri[2]]) / 3.0
        mid = 0.5 * (p1 + p2)
        to_edge = mid - centroid
        if float(np.dot(n, to_edge)) < 0.0:
            n = -n

        Fx += p * n[0] * ds
        Fy += p * n[1] * ds

    cx = Fx / (q_out * chord)
    cy = Fy / (q_out * chord)
    return cx, cy


# -----------------------------
# Main overlay
# -----------------------------
def plot_cp_overlay(meshfile: str, cases, outpng: str, gamma: float = 1.4):
    V, E, bnd = read_gri_mesh(meshfile)

    # Find wall boundary
    wall_edges = None
    wall_key = None
    for k in bnd.keys():
        if k.lower() == "wall":
            wall_edges = bnd[k]
            wall_key = k
            break
    if wall_edges is None:
        raise RuntimeError(f"No 'wall' boundary found in mesh. Boundaries: {list(bnd.keys())}")

    wall_edges = [(int(a), int(b)) for (a, b) in wall_edges]

    # chord and LE reference (works for both open and closed)
    wall_nodes = set()
    for a, b in wall_edges:
        wall_nodes.add(a)
        wall_nodes.add(b)
    le_node = min(wall_nodes, key=lambda i: V[i, 0])
    te_node = max(wall_nodes, key=lambda i: V[i, 0])
    x0 = float(V[le_node, 0])
    chord = float(V[te_node, 0] - V[le_node, 0])
    if chord <= 0:
        raise RuntimeError("Computed non-positive chord from wall nodes.")
    # print(f"[wall debug] x_le={x0:.6f}, chord={chord:.6f}")

    # Ordered paths
    paths = get_ordered_wall_paths(V, wall_edges)

    # Determine upper/lower paths
    upper_path = None
    lower_path = None

    if len(paths) == 1 and len(paths[0]) >= 2 and paths[0][0] == paths[0][-1]:
        # closed cycle: split into 2 LE-->TE paths
        pA, pB = split_cycle_into_two_paths(V, paths[0])
        # pick upper by mean y
        if mean_mid_y(V, pA) >= mean_mid_y(V, pB):
            upper_path, lower_path = pA, pB
        else:
            upper_path, lower_path = pB, pA
    else:
        # open chains: take two longest if available, otherwise single
        if len(paths) >= 2:
            p0c, p1c = paths[0], paths[1]
            # ensure both oriented LE->TE by x
            if V[p0c[0], 0] > V[p0c[-1], 0]:
                p0c = list(reversed(p0c))
            if V[p1c[0], 0] > V[p1c[-1], 0]:
                p1c = list(reversed(p1c))
            if mean_mid_y(V, p0c) >= mean_mid_y(V, p1c):
                upper_path, lower_path = p0c, p1c
            else:
                upper_path, lower_path = p1c, p0c
        elif len(paths) == 1:
            upper_path = paths[0]
            lower_path = None
        else:
            raise RuntimeError("No usable wall paths found.")

    # Edge-->element map
    edge_to_elem = build_edge_to_elem(E)

    # Reference conditions
    p0, p_out, M_out, q_out = compute_reference_conditions(gamma=gamma, pout_ratio=0.7)
    print(f"[ref] gamma={gamma:.3f}  p0={p0:.6f}  p_out={p_out:.6f}  M_out={M_out:.6f}  q_out={q_out:.6f}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    for results_file, label in cases:
        U = read_results_bin(results_file)
        if U.shape[0] != E.shape[0]:
            raise RuntimeError(f"{results_file}: Ne={U.shape[0]} does not match mesh Ne={E.shape[0]}")

        p_cell = compute_pressure(U, gamma)

        xu, cpu = cp_along_path(V, edge_to_elem, p_cell, p_out, q_out, upper_path, x0, chord)

        line_u, = ax.plot(xu, cpu, linewidth=1.8, label=f"{label} (upper)")

        if lower_path is not None:
            xl, cpl = cp_along_path(V, edge_to_elem, p_cell, p_out, q_out, lower_path, x0, chord)
            ax.plot(xl, cpl, linewidth=1.8, linestyle="--", color=line_u.get_color(), label=f"{label} (lower)")

        cx, cy = compute_forces(V, E, wall_edges, edge_to_elem, p_cell, q_out, chord)
        print(f"[forces] {label}: c_x={cx:.6f}, c_y={cy:.6f}")

    ax.axhline(0.0, linestyle="--", linewidth=0.8)
    ax.set_xlabel("x/c")
    ax.set_ylabel("$c_p$")
    ax.set_title(f"Cp Overlay ({wall_key})")
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    ax.legend(fontsize=9, ncol=2)

    plt.tight_layout()
    plt.savefig(outpng, dpi=200)
    print(f"Saved {outpng}")


def main():
    if len(sys.argv) < 5 or ((len(sys.argv) - 3) % 2 != 0):
        print(__doc__)
        sys.exit(1)

    meshfile = sys.argv[1]
    outpng = sys.argv[2]
    args = sys.argv[3:]

    cases = []
    for i in range(0, len(args), 2):
        results_file = args[i]
        label = args[i + 1]
        cases.append((results_file, label))

    plot_cp_overlay(meshfile, cases, outpng, gamma=1.4)


if __name__ == "__main__":
    main()
