"""
Write mesh to .gri format with proper boundary classification.
"""
import numpy as np


def write_gri_file(filename, V, E, I2E_matrix, B2E_matrix, 
                   y_bottom_left, y_top_left, y_bottom_right, y_top_right, x_left, x_right,
                   airfoil_x_min, airfoil_x_max):
    """
    Write mesh to .gri format.
    
    Boundary classification:
    - "Wall": airfoil upper and lower surfaces (x in [airfoil_x_min, airfoil_x_max])
    - "PeriodicBottom": bottom periodic boundary (y ≈ y_bottom, x outside airfoil)
    - "PeriodicTop": top periodic boundary (y ≈ y_top, x outside airfoil)
    - "Inflow": left boundary (x ≈ x_left)
    - "Outflow": right boundary (x ≈ x_right)
    
    Parameters
    ----------
    V : (n_nodes, 2) array of vertex coordinates (0-based indexing)
    E : (n_elem, 3) array of element connectivity (0-based indexing)
    I2E_matrix : interior edges from edgehash (1-based)
    B2E_matrix : boundary edges from edgehash (1-based, [n1, n2, elem])
    y_bottom_left, y_top_left, y_bottom_right, y_top_right : y-coordinates of bottom and top periodic boundaries on left and right sides    
    x_left, x_right : x-coordinates of left (inflow) and right (outflow) boundaries
    airfoil_x_min, airfoil_x_max : x-extent of airfoil region
    """
    
    nnode = V.shape[0]
    nelem = E.shape[0]
    
    # Classify boundary edges
    tol_y = 0.5  # tolerance for identifying top/bottom boundaries
    tol_x = 0.5  # tolerance for identifying left/right boundaries
    
    wall_edges = []
    periodic_bottom_edges = []
    periodic_top_edges = []
    inflow_edges = []
    outflow_edges = []
    
    for i in range(B2E_matrix.shape[0]):
        n1, n2, elem = B2E_matrix[i, :]
        p1 = V[n1 - 1, :]  # convert to 0-based
        p2 = V[n2 - 1, :]
        
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2
        
        # Classify based on priority:
        # 1. Inflow/Outflow (vertical boundaries at left/right)
        # 2. Wall (anything within airfoil x-range)
        # 3. Periodic (top/bottom boundaries outside airfoil)
        
        if abs(mid_x - x_left) < tol_x:
            # Left boundary = Inflow
            inflow_edges.append([n1, n2])
        elif abs(mid_x - x_right) < tol_x:
            # Right boundary = Outflow
            outflow_edges.append([n1, n2])
        elif (airfoil_x_min - tol_x) <= mid_x <= (airfoil_x_max + tol_x):
            # Within airfoil x-range = Wall (airfoil surface)
            wall_edges.append([n1, n2])
        elif abs(mid_y - y_bottom_left) < tol_y or abs(mid_y - y_bottom_right) < tol_y:
            # Bottom boundary outside airfoil = Periodic
            periodic_bottom_edges.append([n1, n2])
        elif abs(mid_y - y_top_left) < tol_y or abs(mid_y - y_top_right) < tol_y:
            # Top boundary outside airfoil = Periodic
            periodic_top_edges.append([n1, n2])
    
    # Find periodic node pairs (nodes at same x on top/bottom periodic boundaries)
    periodic_pairs = []
    tol_match = 1e-6
    
    # Get nodes on periodic boundaries
    bottom_nodes = set()
    for edge in periodic_bottom_edges:
        bottom_nodes.add(edge[0])
        bottom_nodes.add(edge[1])
    
    top_nodes = set()
    for edge in periodic_top_edges:
        top_nodes.add(edge[0])
        top_nodes.add(edge[1])
    
    # Match by x-coordinate
    for n_bot in bottom_nodes:
        x_bot = V[n_bot - 1, 0]
        for n_top in top_nodes:
            x_top = V[n_top - 1, 0]
            if abs(x_bot - x_top) < tol_match:
                periodic_pairs.append([n_bot, n_top])
                break
    
    # Sort periodic pairs by x-coordinate for readability
    periodic_pairs.sort(key=lambda p: V[p[0] - 1, 0])
    
    # Write .gri file
    with open(filename, 'w') as f:
        # Header: nNode nElem Dim
        f.write(f"{nnode} {nelem} 2\n")
        
        # Node coordinates (1-based indexing is implicit in line number)
        for i in range(nnode):
            f.write(f"{V[i, 0]:.12f} {V[i, 1]:.12f}\n")
        
        # Boundary groups
        boundary_groups = []
        if len(wall_edges) > 0:
            boundary_groups.append(("Wall", wall_edges))
        if len(inflow_edges) > 0:
            boundary_groups.append(("Inflow", inflow_edges))
        if len(outflow_edges) > 0:
            boundary_groups.append(("Outflow", outflow_edges))
        if len(periodic_bottom_edges) > 0:
            boundary_groups.append(("PeriodicBottom", periodic_bottom_edges))
        if len(periodic_top_edges) > 0:
            boundary_groups.append(("PeriodicTop", periodic_top_edges))
        
        f.write(f"{len(boundary_groups)}\n")
        
        for bname, edges in boundary_groups:
            f.write(f"{len(edges)} 2 {bname}\n")
            for edge in edges:
                f.write(f"{edge[0]} {edge[1]}\n")
        
        # Element groups (all triangles in one group)
        f.write(f"1 3 TriLagrange\n")
        for i in range(nelem):
            # Convert to 1-based indexing
            f.write(f"{E[i, 0] + 1} {E[i, 1] + 1} {E[i, 2] + 1}\n")
        
        # Periodic groups
        if len(periodic_pairs) > 0:
            f.write(f"1 PeriodicGroup\n")
            f.write(f"2 Translational\n")  # type 2 = translational periodicity
            for pair in periodic_pairs:
                f.write(f"{pair[0]} {pair[1]}\n")
    
    print(f"Wrote {filename}")
    print(f"  Nodes: {nnode}")
    print(f"  Elements: {nelem}")
    print(f"  Boundary groups:")
    for bname, edges in boundary_groups:
        print(f"    {bname}: {len(edges)} edges")
    print(f"  Periodic pairs: {len(periodic_pairs)}")


def save_mesh_from_coarse_mesh(Mesh_obj, I2E_matrix, B2E_matrix, 
                                 y_bottom_left, y_top_left, y_bottom_right, y_top_right, x_left, x_right,
                                 airfoil_x_min, airfoil_x_max,
                                 filename='output.gri'):
    """
    Wrapper for coarse_mesh.py usage.
    
    Example:
        save_mesh_from_coarse_mesh(
            Mesh_obj, I2E_matrix, B2E_matrix,
            y_bottom_left=geom_info['y_bottom_left_box'],
            y_top_left=geom_info['y_top_left_box'],
            y_bottom_right=geom_info['y_bottom_right_box'],
            y_top_right=geom_info['y_top_right_box'],
            x_left=geom_info['x_left_box'],
            x_right=geom_info['x_right_box'],
            airfoil_x_min=geom_info['x_left_upper_airfoil'],
            airfoil_x_max=geom_info['x_right_upper_airfoil'],
            filename='blade_mesh.gri'
        )
    """
    write_gri_file(
        filename,
        Mesh_obj.V,
        Mesh_obj.E,
        I2E_matrix,
        B2E_matrix,
        y_bottom_left, y_top_left, y_bottom_right, y_top_right, x_left, x_right,
        airfoil_x_min, airfoil_x_max
    )
