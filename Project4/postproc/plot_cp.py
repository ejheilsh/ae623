#!/usr/bin/env python3
"""
Plot pressure coefficient distribution on blade surface.
Based on project requirements with proper normalization.
"""
import numpy as np
import matplotlib.pyplot as plt
import struct
import sys

from dg_utils import (
    GAMMA,
    infer_dg_filename,
    integrate_wall_forces,
    map_to_physical,
    maybe_read_dg_results,
    primitive_from_state,
    read_gri_mesh,
    reconstruct_state,
)

def read_results(filename):
    """Read solution data from binary file."""
    with open(filename, 'rb') as f:
        ne = struct.unpack('i', f.read(4))[0]
        data = struct.unpack('d' * 4 * ne, f.read(8 * 4 * ne))
        return np.array(data).reshape(ne, 4)

def read_mesh(filename):
    """Read mesh vertices."""
    with open(filename, 'r') as f:
        nn, ne, dim = map(int, f.readline().split())
        v = np.array([[float(s) for s in f.readline().split()] for _ in range(nn)])
        return v

def get_elements_from_gri(filename):
    """Read element connectivity using triangle corner nodes only.

    Supports mixed-order TriLagrange blocks by keeping the first three corner
    nodes from each element line. That is sufficient for the current edge map
    and wall integration logic.
    """
    with open(filename, 'r') as f:
        nn, ne, dim = map(int, f.readline().split())
        for _ in range(nn): 
            f.readline()
        nb = int(f.readline())
        for _ in range(nb):
            line = f.readline().split()
            nb_in = int(line[0])
            for _ in range(nb_in): 
                f.readline()
        
        elements = []
        ne_total = 0
        while ne_total < ne:
            line = f.readline().split()
            if not line: 
                break
            nei = int(line[0])
            for _ in range(nei):
                row = [int(s)-1 for s in f.readline().split()]
                if len(row) < 3:
                    raise ValueError(f"Invalid element row in {filename}: {row}")
                elements.append(row[:3])
            ne_total += nei
        return np.array(elements)

def get_boundary_edges_from_gri(filename):
    """Read boundary edges grouped by boundary name."""
    boundary_groups = {}
    with open(filename, 'r') as f:
        nn, ne, dim = map(int, f.readline().split())
        for _ in range(nn): 
            f.readline()
        nb = int(f.readline())
        for _ in range(nb):
            line = f.readline().split()
            nb_edges = int(line[0])
            bname = line[2] if len(line) > 2 else f"boundary_{len(boundary_groups)}"
            edges = []
            for _ in range(nb_edges):
                edge_line = f.readline().split()
                n1, n2 = int(edge_line[0])-1, int(edge_line[1])-1
                edges.append((n1, n2))
            boundary_groups[bname] = edges
    return boundary_groups

def compute_reference_conditions(p_out, gamma=1.4):
    """
    Compute reference conditions based on outlet pressure.
    Following the formulas in project requirements:
    q_out = 0.5 * gamma * p_out * M_out^2
    M_out^2 = (2/(gamma-1)) * [(p0/p_out)^((gamma-1)/gamma) - 1]
    
    Assumes:
    - p0 = stagnation pressure (from BC: p0 = a0^2 * rho0 / gamma)
    - rho0 = 1.0
    - a0 = 1.0
    - gamma = 1.4
    """
    rho0 = 1.0
    a0 = 1.0
    p0 = a0**2 * rho0 / gamma
    
    # Calculate M_out^2
    pressure_ratio = p0 / p_out
    M_out_sq = (2.0 / (gamma - 1.0)) * (pressure_ratio**((gamma - 1.0) / gamma) - 1.0)
    M_out = np.sqrt(M_out_sq)
    
    # Calculate q_out
    q_out = 0.5 * gamma * p_out * M_out_sq
    
    return q_out, M_out, p0

def compute_pressure_coefficients(v, elements, boundary_groups, u, gamma=1.4):
    """
    Compute pressure coefficient on blade surface.
    c_p = (p - p_out) / q_out
    Returns arrays of (x, c_p) sorted by x-coordinate.
    """
    # Find wall boundary
    wall_edges = None
    for bname, edges in boundary_groups.items():
        if bname.lower() == 'wall':
            wall_edges = edges
            break
    
    if wall_edges is None:
        print("Warning: No wall boundary found!")
        return np.array([]), np.array([]), np.array([])
    
    # Build element-to-edge map
    elem_to_edge = {}
    for elem_idx, elem in enumerate(elements):
        edges_in_elem = [
            tuple(sorted([elem[0], elem[1]])),
            tuple(sorted([elem[1], elem[2]])),
            tuple(sorted([elem[2], elem[0]]))
        ]
        for edge in edges_in_elem:
            if edge not in elem_to_edge:
                elem_to_edge[edge] = []
            elem_to_edge[edge].append(elem_idx)
    
    # Calculate primitive variables
    rho = u[:, 0]
    rhou = u[:, 1]
    rhov = u[:, 2]
    rhoe = u[:, 3]
    vel_u = rhou / rho
    vel_v = rhov / rho
    qsq = vel_u**2 + vel_v**2
    p = (gamma - 1) * (rhoe - 0.5 * rho * qsq)
    
    # Assume outlet pressure (from solver BC: pout = 0.7 * p0)
    rho0 = 1.0
    a0 = 1.0
    p0 = a0**2 * rho0 / gamma
    p_out = 0.7 * p0  # This is the actual BC used in solver
    
    # Compute reference conditions
    q_out, M_out, p0 = compute_reference_conditions(p_out, gamma)
    
    print(f"Reference conditions:")
    print(f"  p0 = {p0:.6f}")
    print(f"  p_out = {p_out:.6f}")
    print(f"  M_out = {M_out:.6f}")
    print(f"  q_out = {q_out:.6f}")
    
    # Collect (x, c_p) pairs along wall
    x_coords = []
    y_coords = []
    cp_values = []
    
    for edge in wall_edges:
        n1, n2 = edge[0], edge[1]
        edge_key = tuple(sorted([n1, n2]))
        
        if edge_key in elem_to_edge:
            elem_idx = elem_to_edge[edge_key][0]
            p_elem = p[elem_idx]
            
            # Pressure coefficient
            c_p = (p_elem - p_out) / q_out
            
            # Edge midpoint x-coordinate
            edge_mid = 0.5 * (v[n1] + v[n2])
            x_mid = edge_mid[0]
            y_mid = edge_mid[1]
            
            x_coords.append(x_mid)
            y_coords.append(y_mid)
            cp_values.append(c_p)
    
    # Sort by x-coordinate
    sorted_indices = np.argsort(x_coords)
    x_coords = np.array(x_coords)[sorted_indices]
    y_coords = np.array(y_coords)[sorted_indices]
    cp_values = np.array(cp_values)[sorted_indices]
    
    return x_coords, y_coords, cp_values


def _unique_undirected_edges(edges):
    out = []
    seen = set()
    for a, b in edges:
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        out.append((a, b))
    return out


def get_ordered_wall_paths(v, wall_edges):
    edges = _unique_undirected_edges(wall_edges)

    adj = {}
    nodes = set()
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
        nodes.add(a)
        nodes.add(b)

    degs = {n: len(adj.get(n, [])) for n in nodes}

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
            for w in adj.get(u, []):
                if w in unvisited:
                    unvisited.remove(w)
                    stack.append(w)
        components.append(comp)

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
                raise RuntimeError(f"Branching detected at wall node {cur}")
            nxt = nxts[0]
            path.append(nxt)
            prev, cur = cur, nxt
            guard += 1
            if guard > 10 * len(nodes):
                raise RuntimeError("Wall chain walk guard tripped")
        return path

    def walk_cycle(start):
        nbrs = adj.get(start, [])
        if len(nbrs) != 2:
            raise RuntimeError("Cannot walk cycle from non-degree-2 node")
        path = [start, nbrs[0]]
        prev, cur = start, nbrs[0]
        guard = 0
        while True:
            nxts = [x for x in adj[cur] if x != prev]
            if len(nxts) != 1:
                raise RuntimeError(f"Cycle not simple at node {cur}")
            nxt = nxts[0]
            if nxt == start:
                break
            path.append(nxt)
            prev, cur = cur, nxt
            guard += 1
            if guard > 10 * len(nodes):
                raise RuntimeError("Wall cycle walk guard tripped")
        path.append(start)
        return path

    ordered_paths = []
    for comp in components:
        endpoints = [n for n in comp if degs[n] == 1]
        is_cycle = (len(endpoints) == 0 and all(degs[n] == 2 for n in comp))
        if is_cycle:
            start = min(comp, key=lambda i: v[i, 0])
            ordered_paths.append(walk_cycle(start))
        else:
            start = min(endpoints, key=lambda i: v[i, 0]) if len(endpoints) >= 2 else min(comp, key=lambda i: v[i, 0])
            ordered_paths.append(walk_chain(start))

    for i, path in enumerate(ordered_paths):
        if len(path) >= 2 and path[0] != path[-1] and v[path[0], 0] > v[path[-1], 0]:
            ordered_paths[i] = list(reversed(path))

    ordered_paths.sort(key=len, reverse=True)
    return ordered_paths


def split_cycle_into_two_paths(v, cycle_path):
    cyc = cycle_path[:-1]
    nodes = list(dict.fromkeys(cyc))

    le = min(nodes, key=lambda i: v[i, 0])
    te = max(nodes, key=lambda i: v[i, 0])

    adj = {}
    for i in range(len(cyc)):
        a = cyc[i]
        b = cyc[(i + 1) % len(cyc)]
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    def walk(first_neighbor):
        path = [le, first_neighbor]
        prev, cur = le, first_neighbor
        guard = 0
        while cur != te:
            nxts = [x for x in adj[cur] if x != prev]
            if len(nxts) != 1:
                raise RuntimeError(f"Non-simple cycle split at node {cur}")
            nxt = nxts[0]
            path.append(nxt)
            prev, cur = cur, nxt
            guard += 1
            if guard > 10 * len(nodes):
                raise RuntimeError("Cycle split guard tripped")
        return path

    n0, n1 = adj[le][0], adj[le][1]
    p_a = walk(n0)
    p_b = walk(n1)
    if v[p_a[0], 0] > v[p_a[-1], 0]:
        p_a = list(reversed(p_a))
    if v[p_b[0], 0] > v[p_b[-1], 0]:
        p_b = list(reversed(p_b))
    return p_a, p_b


def mean_mid_y(v, path):
    if path is None or len(path) < 2:
        return 0.0
    mids = [0.5 * (v[path[i], 1] + v[path[i + 1], 1]) for i in range(len(path) - 1)]
    return float(np.mean(mids)) if mids else 0.0


def get_upper_lower_wall_paths(v, wall_edges):
    paths = get_ordered_wall_paths(v, wall_edges)
    if not paths:
        return None, None

    if len(paths) == 1 and len(paths[0]) >= 2 and paths[0][0] == paths[0][-1]:
        p_a, p_b = split_cycle_into_two_paths(v, paths[0])
        if mean_mid_y(v, p_a) <= mean_mid_y(v, p_b):
            return p_a, p_b
        return p_b, p_a

    if len(paths) >= 2:
        p0c, p1c = paths[0], paths[1]
        if mean_mid_y(v, p0c) <= mean_mid_y(v, p1c):
            return p0c, p1c
        return p1c, p0c

    return paths[0], None


def directed_edge_reference_data(element, edge):
    v0, v1, v2 = element["corners"]
    n1, n2 = edge
    if (n1, n2) == (v0, v1):
        return np.array([0.0, 0.0]), np.array([1.0, 0.0])
    if (n1, n2) == (v1, v0):
        return np.array([1.0, 0.0]), np.array([0.0, 0.0])
    if (n1, n2) == (v1, v2):
        return np.array([1.0, 0.0]), np.array([0.0, 1.0])
    if (n1, n2) == (v2, v1):
        return np.array([0.0, 1.0]), np.array([1.0, 0.0])
    if (n1, n2) == (v2, v0):
        return np.array([0.0, 1.0]), np.array([0.0, 0.0])
    if (n1, n2) == (v0, v2):
        return np.array([0.0, 0.0]), np.array([0.0, 1.0])
    return None


def sample_dg_cp_along_path(mesh, U_dg, p_order, path, num_samples=5):
    if path is None or len(path) < 2:
        return np.array([]), np.array([])

    edge_to_elem = {}
    for elem_idx, element in enumerate(mesh["elements"]):
        c = element["corners"]
        for edge in [(c[0], c[1]), (c[1], c[2]), (c[2], c[0])]:
            edge_to_elem[tuple(sorted(edge))] = elem_idx

    rho0 = 1.0
    a0 = 1.0
    p0 = a0 * a0 * rho0 / GAMMA
    p_out = 0.7 * p0
    pressure_ratio = p0 / p_out
    m_out_sq = (2.0 / (GAMMA - 1.0)) * (pressure_ratio ** ((GAMMA - 1.0) / GAMMA) - 1.0)
    q_out = 0.5 * GAMMA * p_out * m_out_sq

    x_vals = []
    cp_vals = []
    sample_t = np.linspace(0.0, 1.0, num_samples)
    for i in range(len(path) - 1):
        edge = (int(path[i]), int(path[i + 1]))
        elem_idx = edge_to_elem.get(tuple(sorted(edge)))
        if elem_idx is None:
            continue
        element = mesh["elements"][elem_idx]
        endpoints = directed_edge_reference_data(element, edge)
        if endpoints is None:
            continue
        ref0, ref1 = endpoints
        # avoid duplicating nodes at joins
        t_values = sample_t if i == 0 else sample_t[1:]
        for t in t_values:
            xi_eta = (1.0 - t) * ref0 + t * ref1
            xi, eta = xi_eta
            xy = map_to_physical(element, mesh["nodes"], xi, eta)
            state = reconstruct_state(U_dg[elem_idx], p_order, xi, eta)
            p = primitive_from_state(state)["p"]
            cp = (p - p_out) / q_out
            x_vals.append(xy[0])
            cp_vals.append(cp)
    x_vals = np.array(x_vals)
    cp_vals = np.array(cp_vals)
    if x_vals.size > 1:
        order = np.argsort(x_vals)
        x_vals = x_vals[order]
        cp_vals = cp_vals[order]
    return x_vals, cp_vals


def sample_cellavg_cp_along_path(v, elements, u, path, gamma=1.4):
    if path is None or len(path) < 2:
        return np.array([]), np.array([])

    elem_to_edge = {}
    for elem_idx, elem in enumerate(elements):
        edges_in_elem = [
            tuple(sorted([elem[0], elem[1]])),
            tuple(sorted([elem[1], elem[2]])),
            tuple(sorted([elem[2], elem[0]]))
        ]
        for edge in edges_in_elem:
            elem_to_edge.setdefault(edge, []).append(elem_idx)

    rho = u[:, 0]
    rhou = u[:, 1]
    rhov = u[:, 2]
    rhoe = u[:, 3]
    vel_u = rhou / rho
    vel_v = rhov / rho
    qsq = vel_u**2 + vel_v**2
    p_cell = (gamma - 1) * (rhoe - 0.5 * rho * qsq)

    rho0 = 1.0
    a0 = 1.0
    p0 = a0**2 * rho0 / gamma
    p_out = 0.7 * p0
    q_out, _, _ = compute_reference_conditions(p_out, gamma)

    x_vals = []
    cp_vals = []
    for i in range(len(path) - 1):
        n1 = int(path[i])
        n2 = int(path[i + 1])
        edge_key = tuple(sorted([n1, n2]))
        if edge_key not in elem_to_edge:
            continue
        elem_idx = elem_to_edge[edge_key][0]
        cp = (p_cell[elem_idx] - p_out) / q_out
        x_mid = 0.5 * (v[n1, 0] + v[n2, 0])
        x_vals.append(x_mid)
        cp_vals.append(cp)
    x_vals = np.array(x_vals)
    cp_vals = np.array(cp_vals)
    if x_vals.size > 1:
        order = np.argsort(x_vals)
        x_vals = x_vals[order]
        cp_vals = cp_vals[order]
    return x_vals, cp_vals

def compute_force_coefficients(v, elements, boundary_groups, u, gamma=1.4):
    """
    Compute force coefficients c_x and c_y.
    Normalized by q_out * chord with chord = 18.804 mm = 0.018804 m
    """
    # Find wall boundary
    wall_edges = None
    for bname, edges in boundary_groups.items():
        if bname.lower() == 'wall':
            wall_edges = edges
            break
    
    if wall_edges is None:
        return 0.0, 0.0
    
    # Build element-to-edge map
    elem_to_edge = {}
    for elem_idx, elem in enumerate(elements):
        edges_in_elem = [
            tuple(sorted([elem[0], elem[1]])),
            tuple(sorted([elem[1], elem[2]])),
            tuple(sorted([elem[2], elem[0]]))
        ]
        for edge in edges_in_elem:
            if edge not in elem_to_edge:
                elem_to_edge[edge] = []
            elem_to_edge[edge].append(elem_idx)
    
    # Calculate primitives
    rho = u[:, 0]
    rhou = u[:, 1]
    rhov = u[:, 2]
    rhoe = u[:, 3]
    vel_u = rhou / rho
    vel_v = rhov / rho
    qsq = vel_u**2 + vel_v**2
    p = (gamma - 1) * (rhoe - 0.5 * rho * qsq)
    
    # Reference conditions
    rho0 = 1.0
    a0 = 1.0
    p0 = a0**2 * rho0 / gamma
    p_out = 0.7 * p0
    q_out, M_out, p0 = compute_reference_conditions(p_out, gamma)
    
    # Integrate forces
    Fx = 0.0
    Fy = 0.0
    
    for edge in wall_edges:
        n1, n2 = edge[0], edge[1]
        edge_key = tuple(sorted([n1, n2]))
        
        if edge_key in elem_to_edge:
            elem_idx = elem_to_edge[edge_key][0]
            p_elem = p[elem_idx]
            
            # Edge vector and outward normal
            edge_vec = v[n2] - v[n1]
            edge_length = np.linalg.norm(edge_vec)
            normal = np.array([edge_vec[1], -edge_vec[0]])
            normal = normal / np.linalg.norm(normal)
            
            # Check if normal points outward (away from blade interior)
            centroid = v[elements[elem_idx]].mean(axis=0)
            edge_mid = 0.5 * (v[n1] + v[n2])
            to_edge = edge_mid - centroid
            if np.dot(normal, to_edge) < 0:
                normal = -normal
            
            # Force = p * n * length
            Fx += p_elem * normal[0] * edge_length
            Fy += p_elem * normal[1] * edge_length
    
    # Force coefficients (normalized by q_out * chord)
    # Note: mesh is in mm, so chord = 18.804 mm (not meters)
    chord = 18.804  # Chord length in mm (same units as mesh)
    c_x = Fx / (q_out * chord)
    c_y = Fy / (q_out * chord)
    
    return c_x, c_y

def plot_cp_distribution(meshfile, resultsfile, output_file='cp_distribution.png', show_plot=True):
    """Main function to plot pressure coefficient distribution."""
    print(f"Reading mesh: {meshfile}")
    mesh = read_gri_mesh(meshfile)

    print(f"Reading results: {resultsfile}")
    dg_filename, U_dg, p_order, _ = maybe_read_dg_results(resultsfile)
    if U_dg is not None:
        print(f"Using DG coefficients from {dg_filename}")
        print("Computing pressure coefficients from DG wall reconstruction...")
        wall_edges = next((edges for name, edges in mesh["boundary_groups"].items() if name.lower() == "wall"), None)
        if wall_edges is None:
            print("Error: No wall boundary data found!")
            return
        upper_path, lower_path = get_upper_lower_wall_paths(mesh["nodes"], wall_edges)
        x_upper, cp_upper = sample_dg_cp_along_path(mesh, U_dg, p_order, upper_path)
        x_lower, cp_lower = sample_dg_cp_along_path(mesh, U_dg, p_order, lower_path)
        c_x, c_y = integrate_wall_forces(mesh, U_dg, p_order)
    else:
        print(f"No DG coefficient file found; using cell averages. Expected {infer_dg_filename(resultsfile)}")
        v = read_mesh(meshfile)
        e = get_elements_from_gri(meshfile)
        boundary_groups = get_boundary_edges_from_gri(meshfile)
        u = read_results(resultsfile)
        print("Computing pressure coefficients...")
        wall_edges = next((edges for bname, edges in boundary_groups.items() if bname.lower() == "wall"), None)
        if wall_edges is None:
            print("Error: No wall boundary data found!")
            return
        upper_path, lower_path = get_upper_lower_wall_paths(v, wall_edges)
        x_upper, cp_upper = sample_cellavg_cp_along_path(v, e, u, upper_path)
        x_lower, cp_lower = sample_cellavg_cp_along_path(v, e, u, lower_path)
        c_x, c_y = compute_force_coefficients(v, e, boundary_groups, u)

    if len(x_upper) == 0 and len(x_lower) == 0:
        print("Error: No wall boundary data found!")
        return
    
    print(f"\nForce coefficients:")
    print(f"  c_x = {c_x:.6f}")
    print(f"  c_y = {c_y:.6f}")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    if len(x_upper):
        ax.plot(x_upper, cp_upper, 'b-', linewidth=1.5, label='Upper Surface')
    if len(x_lower):
        ax.plot(x_lower, cp_lower, 'r-', linewidth=1.5, label='Lower Surface')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('x [m]', fontsize=12)
    ax.set_ylabel('$c_p$', fontsize=12)
    ax.set_title('Pressure Coefficient Distribution on Blade Surfaces', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Invert y-axis (traditional aerodynamics convention)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"\nSaved: {output_file}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)

if __name__ == "__main__":
    args = [arg for arg in sys.argv[1:] if arg != "--no-show"]
    show_plot = "--no-show" not in sys.argv[1:]

    if len(args) < 2:
        print("Usage: python plot_cp.py <meshfile> <results.bin> [output.png] [--no-show]")
        print("Example: python plot_cp.py grids/2k.gri results.bin cp_distribution.png")
        sys.exit(1)
    
    meshfile = args[0]
    resultsfile = args[1]
    output_file = args[2] if len(args) > 2 else 'cp_distribution.png'
    
    plot_cp_distribution(meshfile, resultsfile, output_file, show_plot=show_plot)
