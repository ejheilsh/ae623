import sys, numpy as np
sys.path.insert(0, 'postproc')
from dg_utils import read_gri_mesh

m = read_gri_mesh('grids/2k.gri')
n = m['nodes']
wall = m['boundary_groups']['wall']

ny_sum = 0.0
for a, b in wall:
    dx = n[b, 0] - n[a, 0]
    ny_sum += dx  # n_y * dS = dx for 2D edge normal

print(f"Sum n_y*dS over wall = {ny_sum:.10f}")
print(f"So p_inf * sum = {0.714286 * ny_sum:.6f}")
print(f"Divided by normalization 0.005: {0.714286 * ny_sum / 0.005:.2f}")
print()

# Now compute Cl manually from solution data
from dg_utils import read_dg_results
psi, p_order, ndof = read_dg_results('data_steady/steady_2k_p0_results_dg.bin')
print(f"Solution shape: {psi.shape}")
# This is state not adjoint, let me check filenames
import glob
# Compute Cl manually using the 2k mesh solution
U_dg, p_order, ndof = read_dg_results('data_steady/steady_2k_p0_results_dg.bin')
print(f"p_order={p_order}, ndof={ndof}, Ne={U_dg.shape[0]}")

# Read the mesh to get element connectivity and boundary info
# Need to know which elements are adjacent to wall boundary edges
# and what their normals/lengths are

# Read the mesh using the solver's format
m2 = read_gri_mesh('grids/2k.gri')
nodes = m2['nodes']
elems = m2['elements']
wall_edges = m2['boundary_groups']['wall']

gamma = 1.4
rho0 = 1.0
a0 = 1.0
Minf = 0.1
p_inf = rho0 * a0**2 / gamma
qinf = Minf * a0
norm_factor = 0.5 * rho0 * qinf**2

# Build edge-to-element map
edge_to_elem = {}
for ei, e in enumerate(elems):
    corners = e['corners']
    for k in range(3):
        a, b = corners[k], corners[(k+1) % 3]
        key = (min(a, b), max(a, b))
        edge_to_elem.setdefault(key, []).append(ei)

Cl = 0.0
n_wall_edges_found = 0
for a, b in wall_edges:
    key = (min(a, b), max(a, b))
    if key not in edge_to_elem:
        continue
    eL = edge_to_elem[key][0]  # left element
    
    # Cell-average state (p=0)
    Uc = U_dg[eL, 0, :]  # [rho, rhou, rhov, rhoE]
    rho, rhou, rhov, rhoE = Uc
    p = (gamma - 1.0) * (rhoE - 0.5*(rhou**2 + rhov**2)/rho)
    
    # Edge normal and length
    dx = nodes[b, 0] - nodes[a, 0]
    dy = nodes[b, 1] - nodes[a, 1]
    L = np.sqrt(dx**2 + dy**2)
    # outward normal (depends on orientation): n = (-dy/L, dx/L)
    ny = dx / L  # n_y = dx/L for outward-pointing normal
    
    Cl += (p - p_inf) * ny * L / norm_factor
    n_wall_edges_found += 1

print(f"Wall edges matched: {n_wall_edges_found}/{len(wall_edges)}")
print(f"Cl (manual) = {Cl:.4f}")
print(f"Cl (solver) = 490.391")
print(f"p_inf = {p_inf:.6f}, norm_factor = {norm_factor:.6f}")

# Also check: what if we DON'T subtract p_inf?
Cl_raw = 0.0
for a, b in wall_edges:
    key = (min(a, b), max(a, b))
    if key not in edge_to_elem:
        continue
    eL = edge_to_elem[key][0]
    Uc = U_dg[eL, 0, :]
    rho, rhou, rhov, rhoE = Uc
    p = (gamma - 1.0) * (rhoE - 0.5*(rhou**2 + rhov**2)/rho)
    dx = nodes[b, 0] - nodes[a, 0]
    ny = dx / np.sqrt(dx**2 + (nodes[b,1]-nodes[a,1])**2)
    L = np.sqrt(dx**2 + (nodes[b,1]-nodes[a,1])**2)
    Cl_raw += p * ny * L / norm_factor

print(f"Cl_raw (no p_inf subtraction) = {Cl_raw:.4f}")
print(f"Difference = {abs(Cl_raw - Cl):.8f}")

# Check pressure field on wall
print("\nPressure on wall edges:")
p_vals = []
for a, b in wall_edges:
    key = (min(a, b), max(a, b))
    if key not in edge_to_elem:
        continue
    eL = edge_to_elem[key][0]
    Uc = U_dg[eL, 0, :]
    rho, rhou, rhov, rhoE = Uc
    p = (gamma - 1.0) * (rhoE - 0.5*(rhou**2 + rhov**2)/rho)
    u = rhou/rho
    v = rhov/rho
    M = np.sqrt(u**2 + v**2) / np.sqrt(gamma*p/rho)
    mid = 0.5*(nodes[a] + nodes[b])
    p_vals.append(p)
    if len(p_vals) <= 20:
        print(f"  edge ({a},{b}): mid=({mid[0]:.2f},{mid[1]:.2f}) p={p:.6f} M={M:.4f} rho={rho:.4f}")

p_vals = np.array(p_vals)
print(f"\nPressure stats: min={p_vals.min():.6f} max={p_vals.max():.6f} mean={p_vals.mean():.6f}")
print(f"p_inf = {p_inf:.6f}")
print(f"Max |p - p_inf| = {np.max(np.abs(p_vals - p_inf)):.6f}")
print(f"Max Cp = {np.max(np.abs(p_vals - p_inf)) / norm_factor:.2f}")

# What is the raw lift force?
raw_force = Cl * norm_factor  # = integral of (p-p_inf) * n_y dS
print(f"\nRaw lift force (integral of p*n_y*dS) = {raw_force:.6f}")
print(f"For Cl ~ 1, raw force should be ~ {norm_factor:.6f}")
print(f"Ratio = {abs(raw_force/norm_factor):.1f}")

# What chord would give Cl ~ 1?
blade_chord = abs(raw_force / norm_factor)
print(f"\nTo get Cl=1, need chord = {blade_chord:.2f}")
print(f"Actual blade x-extent = 18.80")
print(f"Even with chord=18.8, Cl = {abs(Cl)/18.8:.2f}")
