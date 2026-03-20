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
    infer_dg_filename,
    integrate_wall_forces,
    maybe_read_dg_results,
    read_gri_mesh,
    wall_edge_samples,
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
        return np.array([]), np.array([])
    
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
            
            x_coords.append(x_mid)
            cp_values.append(c_p)
    
    # Sort by x-coordinate
    sorted_indices = np.argsort(x_coords)
    x_coords = np.array(x_coords)[sorted_indices]
    cp_values = np.array(cp_values)[sorted_indices]
    
    return x_coords, cp_values

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
        x, cp = wall_edge_samples(mesh, U_dg, p_order)
        c_x, c_y = integrate_wall_forces(mesh, U_dg, p_order)
    else:
        print(f"No DG coefficient file found; using cell averages. Expected {infer_dg_filename(resultsfile)}")
        v = read_mesh(meshfile)
        e = get_elements_from_gri(meshfile)
        boundary_groups = get_boundary_edges_from_gri(meshfile)
        u = read_results(resultsfile)
        print("Computing pressure coefficients...")
        x, cp = compute_pressure_coefficients(v, e, boundary_groups, u)
        c_x, c_y = compute_force_coefficients(v, e, boundary_groups, u)
    
    if len(x) == 0:
        print("Error: No wall boundary data found!")
        return
    
    print(f"\nForce coefficients:")
    print(f"  c_x = {c_x:.6f}")
    print(f"  c_y = {c_y:.6f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, cp, 'b-', linewidth=1.5, label='Blade Surface')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('x [m]', fontsize=12)
    ax.set_ylabel('$c_p$', fontsize=12)
    ax.set_title('Pressure Coefficient Distribution on Blade Surface', fontsize=14)
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
