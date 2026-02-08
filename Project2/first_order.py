import numpy as np
from utils.mesh import read_mesh
from utils.geometry import compute_geometry
import utils.flux as flux
import utils.limiter as limiter


# ----- STEP 1: INITIALIZATION & PREPROCESSING -----
def main():
    # 1.1 Read mesh from .gri file
    Mesh = read_mesh('coarse_blade_mesh.gri')
    #     - Node coordinates (V)
    #     - Element connectivity (E)
    #     - Boundary edge info (BE) with boundary group IDs
    #     - Interior edge info (IE) connecting element pairs
    
    # 1.2 Compute geometric quantities (do this ONCE, reuse throughout)
    Geometry = compute_geometry(Mesh)
    #     - Element areas (for each cell)
    #     - Interior edge normals and lengths (pointing from Left to Right element)
    #     - Boundary edge normals and lengths (pointing outward)
    
    # 1.3 Set flow parameters
    rho0 = 1.0        # Inlet stagnation density
    a0 = 1.0          # Inlet stagnation speed of sound
    p0 = rho0*a0^2/γ  # Inlet stagnation pressure
    alpha = 50°       # Inlet angle of attack
    pout = 0.7*p0     # Outflow static pressure
    gamma = 1.4       # Ratio of specific heats
    
    # 1.4 Initialize solution
    U = initialize_state(Mesh, rho0, a0, alpha)
    #     U[i,:] = [rho, rho*u, rho*v, rho*E] for each element i
    #     Initial guess: uniform flow with Mach=0.1 at inlet conditions
    
    # 1.5 Time-stepping parameters
    CFL = 0.5         # CFL number for stability
    max_iter = 10000  # Maximum iterations
    tol = 1e-5        # Convergence tolerance (5 orders of magnitude)

