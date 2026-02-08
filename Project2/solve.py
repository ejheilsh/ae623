import numpy as np
import sys
import os

# Add Project1 utilities to path
#sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Project1'))
#sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Project1', 'utils'))

from utils import plotgri, E2N


def solve(Mesh, flow_params, niter, U0=None):
    """
    Euler solver
    
    Parameters:
    -----------
    Mesh : dict
        Computational mesh with keys: 'V' (nodes), 'E' (elements), 
        'IE' (interior edges), 'BE' (boundary edges), 'Bname' (boundary names)
    flow_params : dict
        Flow parameters: rho0, a0, p0, alpha, pout, gamma
    niter : int
        Maximum number of iterations
    U0 : ndarray, optional
        Initial condition, or None to initialize here
    
    Returns:
    --------
    U : ndarray
        Solved state [nElem x 4]
    Rhist : list
        Residual history
    
    Notes:
    ------
    If convergence fails, try decreasing CFL or adjusting IC.
    """
    # Build normals
    Normals = get_normals(Mesh)
    
    # Initialize state
    if U0 is None:
        nElem = Mesh['E'].shape[0]
        # Initialize with low Mach number as per project specs
        Minf_init = 0.1
        alpha = flow_params['alpha']
        Uinit = get_uinf(Minf_init, alpha)
        U = np.tile(Uinit, (nElem, 1))
    else:
        U = U0.copy()
    
    # Time step loop parameters
    CFL = 0.1  # Very conservative to start
    Rtol = 1e-5  # Relaxed tolerance
    Rhist = []
    
    for iiter in range(niter):
        # Calculate residual
        R, dtA = calc_res(Mesh, Normals, flow_params, U)
        
        # Check convergence
        if iiter % 10 == 0:
            Rnorm = np.linalg.norm(R)
            Rhist.append(Rnorm)
            print(f'iiter = {iiter:5d}, Rnorm = {Rnorm:10.4e}')
            
            # Check for NaN/Inf
            if np.any(np.isnan(U)) or np.any(np.isinf(U)):
                n_bad = np.sum(np.isnan(U).any(axis=1) | np.isinf(U).any(axis=1))
                print(f'  WARNING: {n_bad} elements have NaN/Inf values!')
            
            import matplotlib.pyplot as plt
            from utils.plotgri import plot_solution
            plot_solution(Mesh, U[:, 0], title='Density Field', cmap='viridis')
            plt.show()
            if Rnorm < Rtol:
                print('Converged!')
                break
        
        # Update the state (explicit Euler time step)
        dtA = 2 * CFL / dtA
        for k in range(4):
            U[:, k] = U[:, k] - dtA * R[:, k]
    
    return U, Rhist


def get_normals(Mesh):
    """
    Calculate normals and lengths for interior and boundary edges
    
    Parameters:
    -----------
    Mesh : dict
        Mesh data structure
    
    Returns:
    --------
    Normals : dict
        Dictionary with interior and boundary normals and lengths
    """
    V = Mesh['V']      # Nodes
    IE = Mesh['IE']    # Interior edges
    BE = Mesh['BE']    # Boundary edges
    
    Normals = {}
    
    # Interior edge normals (perpendicular to edge, pointing from n1 to n2 right-hand rule)
    # For edge from n1 to n2: normal = [dy, -dx] (rotated 90 degrees CCW)
    inormal = np.zeros((IE.shape[0], 2))
    inormal[:, 0] = V[IE[:, 1], 1] - V[IE[:, 0], 1]   # dy
    inormal[:, 1] = V[IE[:, 0], 0] - V[IE[:, 1], 0]   # -dx
    
    ilength = np.sqrt(inormal[:, 0]**2 + inormal[:, 1]**2)
    Normals['inormal'] = inormal / ilength[:, np.newaxis]
    Normals['ilength'] = ilength
    
    # Boundary edge normals (outward-pointing)
    bnormal = np.zeros((BE.shape[0], 2))
    bnormal[:, 0] = V[BE[:, 1], 1] - V[BE[:, 0], 1]   # dy
    bnormal[:, 1] = V[BE[:, 0], 0] - V[BE[:, 1], 0]   # -dx
    
    blength = np.sqrt(bnormal[:, 0]**2 + bnormal[:, 1]**2)
    Normals['bnormal'] = bnormal / blength[:, np.newaxis]
    Normals['blength'] = blength
    
    return Normals


def get_param():
    """Return ratio of specific heats"""
    return 1.4


def get_uinf(Minf, aoa):
    """
    Calculate freestream conservative state vector
    
    Parameters:
    -----------
    Minf : float
        Mach number
    aoa : float
        Angle of attack (radians)
    
    Returns:
    --------
    Uinf : ndarray
        Conservative state [rho, rho*u, rho*v, rho*E]
    """
    gamma = get_param()
    Uinf = np.array([
        1.0,
        Minf * np.cos(aoa),
        Minf * np.sin(aoa),
        1.0 / ((gamma - 1) * gamma) + 0.5 * Minf**2
    ])
    return Uinf


def calc_res(Mesh, Normals, flow_params, U):
    """
    Calculate residual (sum of fluxes) for each element
    
    Parameters:
    -----------
    Mesh : dict
        Mesh data structure
    Normals : dict
        Normal vectors and lengths
    flow_params : dict
        Flow parameters: rho0, a0, p0, alpha, pout, gamma
    U : ndarray
        Current state [nElem x 4]
    
    Returns:
    --------
    R : ndarray
        Residual [nElem x 4]
    dtA : ndarray
        Inverse time step combination [nElem]
    """
    V = Mesh['V']
    E = Mesh['E']
    IE = Mesh['IE']
    BE = Mesh['BE']
    
    Ne = E.shape[0]
    Ni = IE.shape[0]
    Nb = BE.shape[0]
    
    gamma = flow_params['gamma']
    BN = get_bname(Mesh)
    
    R = np.zeros((Ne, 4))      # Residual
    dtA = np.zeros(Ne)         # Inverse time step + area combination
    
    # Interior edge flux contributions
    for i in range(Ni):
        eL = IE[i, 2]  # Left element
        eR = IE[i, 3]  # Right element
        ilen = Normals['ilength'][i]
        
        F, smag = flux_function(U[eL, :], U[eR, :], Normals['inormal'][i, :], gamma)
        
        R[eL, :] += F * ilen
        R[eR, :] -= F * ilen
        dtA[eL] += smag * ilen
        dtA[eR] += smag * ilen
    
    # Boundary edge flux contributions
    for i in range(Nb):
        eL = BE[i, 2]  # Adjacent element
        ib = BE[i, 3]  # Boundary group
        blen = Normals['blength'][i]
        
        if ib == BN['inflow']:
            # Subsonic inflow: specify stagnation conditions
            F, smag = subsonic_inflow_bc(
                U[eL, :], Normals['bnormal'][i, :],
                flow_params['rho0'], flow_params['p0'],
                flow_params['alpha'], gamma
            )
        elif ib == BN['outflow']:
            # Subsonic outflow: specify back pressure
            F, smag = subsonic_outflow_bc(
                U[eL, :], Normals['bnormal'][i, :],
                flow_params['pout'], gamma
            )
        elif ib == BN['wall']:
            # Wall boundary condition (blade surface)
            F, smag = wall_flux(U[eL, :], Normals['bnormal'][i, :], gamma)
        elif ib == BN['periodicbottom'] or ib == BN['periodictop']:
            j_match = Mesh['periodic_map'][i]  # Matching boundary edge index
            eR = BE[j_match, 2]  # Element on other side of periodic boundary
            F, smag = flux_function(U[eL, :], U[eR, :], Normals['bnormal'][i, :], gamma)
        else:
            # Shouldn't happen - use extrapolation as fallback
            F, smag = flux_function(U[eL, :], U[eL, :], Normals['bnormal'][i, :], gamma)
        
        R[eL, :] += F * blen
        dtA[eL] += smag * blen
    
    return R, dtA


def get_bname(Mesh):
    """
    Get boundary name indices
    
    Parameters:
    -----------
    Mesh : dict
        Mesh data structure with 'Bname' key
    
    Returns:
    --------
    BN : dict
        Dictionary mapping boundary names to indices
    """
    BN = {
        'inflow': -1, 
        'outflow': -1, 
        'wall': -1,
        'periodicbottom': -1,
        'periodictop': -1
    }
    
    for i, name in enumerate(Mesh['Bname']):
        name_lower = name.lower()
        if name_lower == 'inflow':
            BN['inflow'] = i
        elif name_lower == 'outflow':
            BN['outflow'] = i
        elif name_lower == 'wall':
            BN['wall'] = i
        elif name_lower == 'periodicbottom':
            BN['periodicbottom'] = i
        elif name_lower == 'periodictop':
            BN['periodictop'] = i
        else:
            print(f'Warning: Unrecognized boundary name: {name}')
    
    return BN


def flux_function(UL, UR, n, gamma):
    """HLLE flux function for Euler equations with robustness"""
    
    # Left state with floors
    rL = max(UL[0], 1e-6)
    uL = UL[1] / rL
    vL = UL[2] / rL
    unL = uL * n[0] + vL * n[1]
    qL = np.sqrt(UL[1]**2 + UL[2]**2) / rL
    pL = (gamma - 1) * (UL[3] - 0.5 * rL * qL**2)
    pL = max(pL, 1e-6)  # Floor on pressure
    
    rHL = UL[3] + pL
    cL = np.sqrt(gamma * pL / rL)
    sLmin = min(0, unL - cL)
    sLmax = max(0, unL + cL)
    sLmag = abs(unL) + cL
    
    FL = np.array([
        rL * unL,
        UL[1] * unL + pL * n[0],
        UL[2] * unL + pL * n[1],
        rHL * unL
    ])
    
    # Right state with floors
    rR = max(UR[0], 1e-6)
    uR = UR[1] / rR
    vR = UR[2] / rR
    unR = uR * n[0] + vR * n[1]
    qR = np.sqrt(UR[1]**2 + UR[2]**2) / rR
    pR = (gamma - 1) * (UR[3] - 0.5 * rR * qR**2)
    pR = max(pR, 1e-6)  # Floor on pressure
    
    rHR = UR[3] + pR
    cR = np.sqrt(gamma * pR / rR)
    sRmin = min(0, unR - cR)
    sRmax = max(0, unR + cR)
    sRmag = abs(unR) + cR
    
    FR = np.array([
        rR * unR,
        UR[1] * unR + pR * n[0],
        UR[2] * unR + pR * n[1],
        rHR * unR
    ])
    
    smag = max(sLmag, sRmag)
    
    sLRmin = min(sLmin, sRmin)
    sLRmax = max(sLmax, sRmax)
    
    # HLLE flux formula with safeguard
    denom = sLRmax - sLRmin
    if abs(denom) < 1e-12:
        F = 0.5 * (FL + FR)
    else:
        F = (0.5 * (FL + FR) 
             - 0.5 * (sLRmax + sLRmin) / denom * (FR - FL)
             + sLRmax * sLRmin / denom * (UR - UL))
    
    return F, smag


def wall_flux(UL, n, gamma):
    """
    Solid wall boundary flux (inviscid wall)
    
    Parameters:
    -----------
    UL : ndarray
        Interior state [rho, rho*u, rho*v, rho*E]
    n : ndarray
        Outward-pointing unit normal [nx, ny]
    gamma : float
        Ratio of specific heats
    
    Returns:
    --------
    F : ndarray
        Wall flux (only pressure contribution)
    smag : float
        Maximum wave speed
    """
    rL = UL[0]
    uL = UL[1] / rL
    vL = UL[2] / rL
    unL = uL * n[0] + vL * n[1]
    qL = np.sqrt(UL[1]**2 + UL[2]**2) / rL
    utL = np.sqrt(qL**2 - unL**2)
    pL = (gamma - 1) * (UL[3] - 0.5 * rL * utL**2)
    rHL = UL[3] + pL
    cL = np.sqrt(gamma * pL / rL)
    
    smag = abs(unL) + cL
    
    # Wall flux: no mass flux, only pressure force
    F = np.array([
        0.0,
        pL * n[0],
        pL * n[1],
        0.0
    ])
    
    return F, smag

def subsonic_inflow_bc(UL, n, rho0, p0, alpha, gamma):
    """Simplified subsonic inflow BC"""

    #return flux_function(UL, UR, n, gamma)
    pass


def subsonic_outflow_bc(UL, n, pout, gamma):
    """Simplified subsonic outflow BC"""
    
    
    #return flux_function(UL, UR, n, gamma)
    pass

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


if __name__ == '__main__':
    # Example usage
    if len(sys.argv) < 2:
        print("Usage: python solve.py <mesh.gri>")
        print("Example: python solve.py ../Project1/data/coarse_blade_mesh.gri")
        sys.exit(1)
    
    mesh_file = sys.argv[1]
    
    # Load mesh using Project1's infrastructure
    Mesh = load_mesh_from_gri(mesh_file)
    
    # Set flow conditions (from Project 2 specs)
    gamma = 1.4
    rho0 = 1.0                          # Inlet stagnation density
    a0 = 1.0                            # Inlet stagnation speed of sound
    p0 = rho0 * a0**2 / gamma          # Inlet stagnation pressure
    alpha = 50.0 * np.pi / 180.0       # Inlet angle of attack (50 degrees)
    pout = 0.7 * p0                     # Outflow static pressure
    
    flow_params = {
        'rho0': rho0,
        'a0': a0,
        'p0': p0,
        'alpha': alpha,
        'pout': pout,
        'gamma': gamma
    }
    
    niter = 10000
    
    print(f"\nSolving Euler equations:")
    print(f"  Inlet: rho0={rho0:.3f}, p0={p0:.3f}, alpha={alpha*180/np.pi:.1f} deg")
    print(f"  Outflow: pout={pout:.3f}")
    print(f"  Max iterations: {niter}")
    print("="*60)
    
    U, Rhist = solve(Mesh, flow_params, niter)
    
    print("="*60)
    if len(Rhist) > 0:
        print(f"Solution obtained. Final residual: {Rhist[-1]:10.4e}")
        print(f"Iterations: {len(Rhist)*10}")
    
    # Save solution
    output_file = 'solution.txt'
    np.savetxt(output_file, U)
    print(f"Solution saved to {output_file}")

    # now find the map in image form
    # Plotting the solution (e.g. density) on the mesh
    import matplotlib.pyplot as plt
    from utils.plotgri import plot_solution
    plot_solution(Mesh, U[:, 0], title='Density Field', cmap='viridis')
    plt.show()
