import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
import os
from utils import plotgri, E2N
import matplotlib.pyplot as plt
from utils.plotgri import plot_solution

"""
Plot the files from unsteady solution history to create the plots needed for report
Unsteady runs. Run simulations at both first and second order, on at least two
meshes: coarse and fine, for a sufficiently long time horizon to observe several cycles of
periodic output behavior. Show a few entropy field plots at various times and time histories
of the force coefficients (overlay some of these to make comparisons). Discuss the effect of
mesh resolution and order on the unsteady results.
"""


def load_mesh_from_gri(mesh_file):
    """
    Load mesh using Project1's robust reader
    
    Parameters:
    -----------
    mesh_file : str
        Path to .gri file
    
    Returns:
    --------
    Mesh : dict
        Mesh dictionary with V, E, IE, BE, Bname, periodic_map
    """
    print(f"Reading mesh: {mesh_file}")
    
    # Use Project1's robust reader
    nodes, elements, boundary_groups, periodic_pairs = plotgri.read_gri_file(mesh_file)
    
    # Convert to 0-based indexing
    V = nodes
    E = elements - 1  # Convert from 1-based to 0-based
    
    print(f"  Nodes: {V.shape[0]}, Elements: {E.shape[0]}")
    
    # Build interior and boundary edge connectivity using Project1's E2N
    I2E_matrix, B2E_matrix = E2N.edgehash(E + 1)  # edgehash expects 1-based
    
    # Convert I2E_matrix to IE format: [n1, n2, elemL, elemR] (0-based)
    IE = I2E_matrix.copy()
    IE[:, 0:2] -= 1  # Convert node indices to 0-based
    IE[:, 2:4] -= 1  # Convert element indices to 0-based
    
    # Build boundary name list
    Bname = list(boundary_groups.keys())
    
    # Build BE: [n1, n2, elem, bgroup] with bgroup column added
    # B2E_matrix has only 3 columns: [n1, n2, elem]
    BE = np.zeros((B2E_matrix.shape[0], 4), dtype=np.int32)
    BE[:, 0:3] = B2E_matrix  # Copy [n1, n2, elem]
    BE[:, 0:2] -= 1  # Convert node indices to 0-based
    BE[:, 2] -= 1    # Convert element index to 0-based
    BE[:, 3] = -1    # Initialize boundary group to -1 (unassigned)
    
    # Map boundary edges to their group indices
    # Build a lookup dict for faster matching
    be_lookup = {}
    for i in range(BE.shape[0]):
        n1, n2 = BE[i, 0], BE[i, 1]
        # Store both orderings
        be_lookup[(n1, n2)] = i
        be_lookup[(n2, n1)] = i
    
    for bname_idx, (bname, edges) in enumerate(boundary_groups.items()):
        # Find which BE entries correspond to this boundary group
        for edge in edges:
            n1_target, n2_target = edge[0] - 1, edge[1] - 1  # Convert to 0-based
            # Look up this edge
            if (n1_target, n2_target) in be_lookup:
                idx = be_lookup[(n1_target, n2_target)]
                BE[idx, 3] = bname_idx
            elif (n2_target, n1_target) in be_lookup:
                idx = be_lookup[(n2_target, n1_target)]
                BE[idx, 3] = bname_idx
    
    print(f"  Interior edges: {IE.shape[0]}, Boundary edges: {BE.shape[0]}")
    print(f"  Boundary groups: {Bname}")
    
    # Convert unassigned boundary edges to interior
    E, IE, BE = convert_unassigned_edges(E, IE, BE)
    
    # Build periodic boundary mapping
    periodic_map = build_periodic_map(BE, periodic_pairs, Bname)
    if periodic_map is not None:
        print(f"  Periodic pairs: {len(periodic_pairs)} node pairs, {len(periodic_map)} edge pairs")
    
    Mesh = {
        'V': V,
        'E': E,
        'IE': IE,
        'BE': BE,
        'Bname': Bname,
        'periodic_map': periodic_map
    }
    
    return Mesh



def build_periodic_map(BE, periodic_pairs, Bname): # ADDED NEW FROM DR. FIDOWSKI -> Project 1 -> NOW ->> SEE UTILS FOR ELABORATION
    """
    Build a mapping from periodic bottom edges to periodic top edges
    
    Parameters:
    -----------
    BE : ndarray
        Boundary edges [n1, n2, elem, bgroup]
    periodic_pairs : list
        List of [bottom_node, top_node] pairs (1-based from .gri file)
    Bname : list
        Boundary names
    
    Returns:
    --------
    periodic_map : dict
        Maps BE index on bottom -> BE index on top (and vice versa)
    """
    if len(periodic_pairs) == 0:
        return None
    
    # Build node mapping (0-based)
    node_map = {}
    for pair in periodic_pairs:
        n_bottom = pair[0] - 1  # Convert to 0-based
        n_top = pair[1] - 1
        node_map[n_bottom] = n_top
        node_map[n_top] = n_bottom  # Bidirectional
    
    # Find periodicbottom and periodictop indices
    pb_idx = -1
    pt_idx = -1
    for i, name in enumerate(Bname):
        if name.lower() == 'periodicbottom':
            pb_idx = i
        elif name.lower() == 'periodictop':
            pt_idx = i
    
    if pb_idx == -1 or pt_idx == -1:
        print("  Warning: Could not find periodic boundary indices")
        return None
    
    # Build edge mapping
    periodic_map = {}
    
    # For each periodic boundary edge, find its matching edge
    for i in range(BE.shape[0]):
        if BE[i, 3] == pb_idx or BE[i, 3] == pt_idx:
            n1, n2 = BE[i, 0], BE[i, 1]
            
            # Find matching nodes
            if n1 in node_map and n2 in node_map:
                n1_match = node_map[n1]
                n2_match = node_map[n2]
                
                # Find the BE index with these matching nodes
                for j in range(BE.shape[0]):
                    if i != j:  # Don't match to self
                        m1, m2 = BE[j, 0], BE[j, 1]
                        # Check both orderings
                        if (m1 == n1_match and m2 == n2_match) or (m1 == n2_match and m2 == n1_match):
                            periodic_map[i] = j
                            break
    
    return periodic_map

def convert_unassigned_edges(E, IE, BE): 
    """
    hey guys we had some unassigned edges in the boundary edge list (bgroup = -1) 
    which means they are not actually boundary edges but should be interior edges. 
    This can happen if the .gri file has some edges that are not included in any boundary group. 
    To fix this, we will check for any BE entries with bgroup = -1 and convert them to interior edges by creating 
    a new element if necessary. This is a bit of a hack, but it allows us to handle meshes that might have some
    inconsistencies without crashing the solver.
    """
    unassigned = np.where(BE[:, 3] == -1)[0]
    if len(unassigned) == 0:
        return E, IE, BE
    
    print(f"  Converting {len(unassigned)} unassigned boundary edges to interior...")
    new_interior = []
    edges_to_remove = set()
    
    # Collect all unassigned edges
    unassigned_edges = []
    for idx in unassigned:
        n1, n2, elemL = BE[idx, 0], BE[idx, 1], BE[idx, 2]
        unassigned_edges.append((idx, n1, n2, elemL))
    
    # Check if they form a closed loop (triangle)
    if len(unassigned_edges) == 3:
        # Get unique nodes from these edges
        all_nodes = set()
        for idx, n1, n2, elemL in unassigned_edges:
            all_nodes.add(n1)
            all_nodes.add(n2)
        
        if len(all_nodes) == 3:
            # These 3 edges form a triangle - create missing element
            new_nodes = list(all_nodes)
            new_elem_idx = E.shape[0]
            E = np.vstack([E, np.array([new_nodes], dtype=np.int32)])
            print(f"  Created missing element {new_elem_idx} with nodes {new_nodes}")
            
            # Now convert these edges to interior
            for idx, n1, n2, elemL in unassigned_edges:
                elemR = new_elem_idx
                new_interior.append([n1, n2, elemL, elemR])
                edges_to_remove.add(idx)
                print(f"    Edge {idx}: nodes [{n1},{n2}], elemL={elemL}, elemR={elemR} (NEW)")
    
    # Add converted interior edges
    if new_interior:
        new_ie_array = np.array(new_interior, dtype=np.int32)
        IE = np.vstack([IE, new_ie_array])
    
    # Remove converted edges from BE
    if edges_to_remove:
        keep_mask = np.ones(BE.shape[0], dtype=bool)
        for idx in edges_to_remove:
            keep_mask[idx] = False
        BE = BE[keep_mask]
    
    print(f"  New counts: Elements={E.shape[0]}, Interior={IE.shape[0]}, Boundary={BE.shape[0]}")
    
    return E, IE, BE


# IMPORT data/unsteady/coarse iterations, whole folder
data_folder = 'data/unsteady/coarse'
data_organized = {}
time_values = {}  # Store time values for each iteration
N_files_load_limit = 20
N_plot_frames = 10
# Load mesh using prior infrastructure
mesh_file = "data/coarse_blade_mesh.gri"
Mesh = load_mesh_from_gri(mesh_file)

for i in range(1, N_files_load_limit + 1):  # Load files 1 to N_files_load_limit
    try:
        # Use glob to find files matching pattern with sequential number
        pattern = f'{data_folder}/solution_*_{i:04d}.txt'
        matching_files = glob.glob(pattern)
        
        if matching_files:
            filename = matching_files[0]  # Take first match
            data_organized[i] = np.loadtxt(filename)
            
            # Extract time from filename (e.g., solution_34.42_0004.txt -> 34.42)
            base_name = os.path.basename(filename)  # Get just filename without path
            # Split by underscores and get the time part
            parts = base_name.split('_')
            if len(parts) >= 2:
                time_str = parts[1]  # Get the time part (e.g., "34.42")
                try:
                    time_values[i] = float(time_str)
                except ValueError:
                    time_values[i] = 0.0  # Default if can't parse
            else:
                time_values[i] = 0.0
                
            print(f"Loaded {filename}, time = {time_values[i]:.3f}")
        else:
            print(f"No file found matching pattern: {pattern}")
            
    except Exception as e:
        print(f"Error loading file {i}: {e}")

# Plot every N_plot_frames
for i in range(1, N_files_load_limit + 1, N_plot_frames):
    if i in data_organized:
        U = data_organized[i]
        time_value = time_values.get(i, 0.0)  # Get time for this iteration
        # now find the map in image form
        # Plotting the solution (e.g. density) on the mesh
        plot_solution(Mesh, U[:, 1] / U[:, 0], title='Vel_x Field', cmap='viridis')
        plt.title(f'Solution Snapshot at Iteration {i}, Time = {time_value:.3f} Seconds')
        plt.xlabel('X Index')
        plt.ylabel('Y Index')
        plt.savefig(f'data/unsteady/coarse/plots/unsteady_snapshot_{i:04d}.png')
        plt.show()
        plt.close()
    else:
        print(f"No data to plot for iteration {i}")