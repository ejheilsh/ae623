#!/usr/bin/env python3
"""
Postprocessing script for unsteady Euler solver results.
Plots entropy and Mach fields at various timesteps, and computes force coefficients.

Usage:
    python postproc/plot_unsteady.py <meshfile> <results_dir>
    
Example:
    python postproc/plot_unsteady.py grids/coarse.gri data/
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import struct
import sys
import glob
import os

def read_results(filename):
    """Read binary results file."""
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
    """Extract element connectivity."""
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
                elements.append([int(s)-1 for s in f.readline().split()])
            ne_total += nei
        return np.array(elements)

def get_boundary_edges_from_gri(filename):
    """Extract boundary edges for force calculation."""
    with open(filename, 'r') as f:
        nn, ne, dim = map(int, f.readline().split())
        for _ in range(nn): 
            f.readline()
        
        nb = int(f.readline())
        boundary_groups = {}
        
        for ib in range(nb):
            line = f.readline().split()
            nb_edges = int(line[0])
            bname = line[1] if len(line) > 1 else f"boundary_{ib}"
            
            edges = []
            for _ in range(nb_edges):
                edge_nodes = [int(s)-1 for s in f.readline().split()]
                edges.append(edge_nodes)
            boundary_groups[bname] = np.array(edges)
        
        return boundary_groups

def extract_time_from_filename(filename):
    """Extract time value from filename pattern like 'results_1.234567_0004.bin'."""
    basename = os.path.basename(filename)
    try:
        parts = basename.replace('results_', '').replace('.bin', '').split('_')
        if len(parts) >= 1:
            return float(parts[0])
    except:
        pass
    return 0.0

def compute_force_coefficients(v, elements, boundary_groups, u, gamma=1.4):
    """Compute force coefficients from wall pressure."""
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
            
            # Check if normal points outward
            centroid = v[elements[elem_idx]].mean(axis=0)
            edge_mid = 0.5 * (v[n1] + v[n2])
            to_edge = edge_mid - centroid
            if np.dot(normal, to_edge) < 0:
                normal = -normal
            
            # Force = p * n * length
            Fx += p_elem * normal[0] * edge_length
            Fy += p_elem * normal[1] * edge_length
    
    # Force coefficients (normalized by reference dynamic pressure * chord)
    q_ref = 0.5 * 1.0 * 1.0**2  # rho_ref * a_ref^2
    chord = 1.0
    Cf_x = Fx / (q_ref * chord)
    Cf_y = Fy / (q_ref * chord)
    
    return Cf_x, Cf_y

def plot_unsteady_results(meshfile, results_dir, output_dir='unsteady_plots'):
    """Main plotting function."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Read mesh
    print(f"Reading mesh: {meshfile}")
    v = read_mesh(meshfile)
    e = get_elements_from_gri(meshfile)
    boundary_groups = get_boundary_edges_from_gri(meshfile)
    verts = v[e]
    
    # Find all result files
    pattern = os.path.join(results_dir, "results_*.bin")
    result_files = sorted(glob.glob(pattern))
    
    if len(result_files) == 0:
        print(f"Error: No files found matching {pattern}")
        return
    
    print(f"Found {len(result_files)} result files")
    
    gamma = 1.4
    times = []
    Cf_x_list = []
    Cf_y_list = []
    
    # Determine global ranges for consistent colormaps
    entropy_min, entropy_max = np.inf, -np.inf
    mach_min, mach_max = np.inf, -np.inf
    
    for resfile in result_files:
        u = read_results(resfile)
        rho = u[:, 0]
        rhou = u[:, 1]
        rhov = u[:, 2]
        rhoe = u[:, 3]
        vel_u = rhou / rho
        vel_v = rhov / rho
        qsq = vel_u**2 + vel_v**2
        p = (gamma - 1) * (rhoe - 0.5 * rho * qsq)
        mach = np.sqrt(qsq) / np.sqrt(gamma * p / rho)
        entropy = np.log(p / rho**gamma)
        
        entropy_min = min(entropy_min, entropy.min())
        entropy_max = max(entropy_max, entropy.max())
        mach_min = min(mach_min, mach.min())
        mach_max = max(mach_max, mach.max())
    
    print(f"Entropy range: [{entropy_min:.6f}, {entropy_max:.6f}]")
    print(f"Mach range: [{mach_min:.6f}, {mach_max:.6f}]")
    
    # Plot each timestep
    for idx, resfile in enumerate(result_files):
        time = extract_time_from_filename(resfile)
        u = read_results(resfile)
        
        # Calculate primitives
        rho = u[:, 0]
        rhou = u[:, 1]
        rhov = u[:, 2]
        rhoe = u[:, 3]
        vel_u = rhou / rho
        vel_v = rhov / rho
        qsq = vel_u**2 + vel_v**2
        p = (gamma - 1) * (rhoe - 0.5 * rho * qsq)
        mach = np.sqrt(qsq) / np.sqrt(gamma * p / rho)
        entropy = np.log(p / rho**gamma)
        
        # Compute forces
        Cf_x, Cf_y = compute_force_coefficients(v, e, boundary_groups, u, gamma)
        times.append(time)
        Cf_x_list.append(Cf_x)
        Cf_y_list.append(Cf_y)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        
        # Entropy
        pc1 = PolyCollection(verts, cmap='viridis', edgecolors='none')
        pc1.set_array(entropy)
        pc1.set_clim(entropy_min, entropy_max)
        ax1.add_collection(pc1)
        ax1.autoscale()
        ax1.set_aspect('equal')
        fig.colorbar(pc1, ax=ax1, label='Entropy: log(p/ρ^γ)')
        ax1.set_title(f'Entropy at t = {time:.6f}')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        
        # Mach number
        pc2 = PolyCollection(verts, cmap='jet', edgecolors='none')
        pc2.set_array(mach)
        pc2.set_clim(mach_min, mach_max)
        ax2.add_collection(pc2)
        ax2.autoscale()
        ax2.set_aspect('equal')
        fig.colorbar(pc2, ax=ax2, label='Mach Number')
        ax2.set_title(f'Mach Number at t = {time:.6f}')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        
        plt.tight_layout()
        outfile = os.path.join(output_dir, f'entropy_field_{idx:04d}_t{time:.6f}.png')
        plt.savefig(outfile, dpi=150)
        plt.close()
        print(f"Saved {outfile}")
    
    # Plot force history
    times = np.array(times)
    Cf_x_list = np.array(Cf_x_list)
    Cf_y_list = np.array(Cf_y_list)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(times, Cf_x_list, 'b-', linewidth=2)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Cf_x')
    ax1.set_title('Axial Force Coefficient vs Time')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(times, Cf_y_list, 'r-', linewidth=2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Cf_y')
    ax2.set_title('Normal Force Coefficient vs Time')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    force_file = os.path.join(output_dir, 'force_history.png')
    plt.savefig(force_file, dpi=150)
    print(f"\nSaved {force_file}")
    
    # Print statistics
    print("\n" + "="*60)
    print("Force Coefficient Statistics:")
    print("="*60)
    print(f"Cf_x: mean={Cf_x_list.mean():.6f}, std={Cf_x_list.std():.6f}")
    print(f"Cf_y: mean={Cf_y_list.mean():.6f}, std={Cf_y_list.std():.6f}")
    print("="*60)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        print("\nUsage: python postproc/plot_unsteady.py <meshfile> <results_dir> [output_dir]")
        print("Example: python postproc/plot_unsteady.py grids/coarse.gri data/")
        print("         python postproc/plot_unsteady.py grids/coarse.gri data/ unsteady_plots/")
        sys.exit(1)
    
    meshfile = sys.argv[1]
    results_dir = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else 'unsteady_plots'
    
    plot_unsteady_results(meshfile, results_dir, output_dir)
