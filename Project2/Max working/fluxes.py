import numpy as np
from utils import Uutil
"""
Max Mah
2/11/26

Implemented two Riemann solvers: Roe flux and HLLE flux 
Implemented boundary conditions through fluxes on boundary faces:
inviscid wall, subsonic inflow, subsonic outflow 
"""

def inviscid_wall(Uplus: np.ndarray, n: np.ndarray, gamma: float):
    """
    Inviscid wall boundary condition (Fidkowski, pg. 61)
    Inputs:
    ------- 
    Uplus: np.ndarray
        Interior state vector
    n: np.ndarray
        Normal vector pointing out of domain
    gamma: float 
        Ratio of specific heats
    """    
    rhoint = Uplus[0]
    rhouint = Uplus[1]
    rhovint = Uplus[2]
    rhoEint = Uplus[3]
    
    vvecint = np.array([rhouint / rhoint, rhovint / rhoint]) 
    vvecb = vvecint - (vvecint @ n) * n # boundary velocity is tangential; interior velocity with wall-normal component removed 
    qsqb = vvecb @ vvecb
    pb = (gamma - 1) * (rhoEint - (0.5 * rhoint * qsqb))
    Fb = np.array([
        0, 
        pb * n[0], 
        pb * n[1],
        0
    ])
   
    # for wall, use interior state with normal velocity removed for the wave speed
    u = vvecb @ n # u = vvec dot n, with normal velocity component removed
    cb = np.sqrt((gamma * pb / rhoint)) # NOTE just using the interior density here, dont think we need to compute rho on boundary....
    smax = np.abs(u) + cb 
    return Fb, smax 

# NOTE MAKE SURE ALPHA IS PASSED IN IN RADIANS!!!!
def subsonic_inflow(Uplus: np.ndarray, n: float, rhot: float, ct: float, alpha: float, gamma: float):
    """
    Subsonic inflow boundary condition. Boundary flux Fb is determined by constructing boundary state ub from the interior state uplus. 
    (Fidkowski, pg 62)
    Inputs:
    -------
    Uplus: np.ndarray
       NInterior state vector
    n: float
       NNormal vector pointing out of domain (interior --> outer)
    rhot: float
       NTotal density, same as inlet stagnation density (rho0)
    ct: float
       NSpeed of sound, same as inlet stagnation speed of sound (a0)
    alpha: float
       NInlet angle of attack
    gamma: float
       NRatio of specific heats
    """
    pt = ct**2 * rhot / gamma

    # deconstruct interior state TODO offload this to helper function later
    rhoplus = Uplus[0]
    rhouplus = Uplus[1]
    rhovplus = Uplus[2]
    rhoEplus = Uplus[3]

    uplus = rhouplus/rhoplus
    vplus = rhovplus/rhoplus
    vvecplus = np.array([uplus, vplus])
    unplus = vvecplus @ n   # wall-normal velocity component from the interior (Fidkowski, pg 62)

    qsq = uplus**2 + vplus**2           # magnitude velocity squared 
    q = np.sqrt(qsq)            # magnitude velocity 

    pplus = (gamma - 1) * (rhoEplus - 0.5 * rhoplus * qsq) # pressure
    cplus = np.sqrt((gamma * pplus) / rhoplus)
    nin = np.array([np.cos(alpha), np.sin(alpha)]) # specified inflow direction
    dn = nin @ n
    
    Jplus = unplus + (2 * cplus)/(gamma - 1)    # Riemann invariant (Fidkowski, pg 62)

    # inflow Mach number Mb is calculated from Jplus and specified parameters by solving a quadratic of form A*M^2 + B*M + C = 0
    A = ((gamma * pt * dn**2) / rhot) - ((gamma - 1) / 2) * Jplus**2
    B = ((4 * gamma * pt * dn) / (rhot * (gamma - 1)))
    C = ((4 * gamma * pt) / (rhot * (gamma - 1)**2)) - Jplus**2 
    
    discr = B**2 - 4 * A * C
    tol = 1e-12
    roots = []

    # Degenerate case A ~= 0: solve the linearized equation B*M + C = 0
    if np.abs(A) < tol:
        if np.abs(B) >= tol:
            roots = [-C / B]
    else:
        # Guard tiny negative discriminants from roundoff
        if discr < -tol:
            discr = 0.0
        else:
            discr = max(discr, 0.0)
        sqrt_discr = np.sqrt(discr)
        roots = [
            (-B + sqrt_discr) / (2 * A),
            (-B - sqrt_discr) / (2 * A),
        ]

    # Prefer physical subsonic roots; if unavailable, clip nearest positive root
    Mb_candidates = [M for M in roots if np.isfinite(M) and (0.0 < M < 1.0)]
    if len(Mb_candidates) == 1:
        Mb = Mb_candidates[0]
    elif len(Mb_candidates) == 2:
        Mb = min(Mb_candidates)
    else:
        positive_roots = [M for M in roots if np.isfinite(M) and M > 0.0]
        if len(positive_roots) > 0:
            Mb = min(positive_roots)
        else:
            Mb = 0.1
        Mb = np.clip(Mb, tol, 1.0 - tol)
    
    # using Mb and stagnation quantities, construct exterior state (ub) quantities
    Tr = 1 + 0.5 * (gamma - 1) * Mb**2  # temperature ratio = exterior static temperature / total temperature (Tb / Tt)
    pb = pt * Tr**(gamma / (gamma - 1)) # exterior static pressure
    RTb = Tr * pt / rhot    
    rhob = pb / RTb                     # exterior static density
    cb = np.sqrt((gamma * pb) / rhob)   # exterior speed of sound
    vvecb = Mb * cb * nin               # exterior velocity
    qsqb = vvecb[0]**2 + vvecb[1]**2
    rhoEb = (pb / (gamma - 1)) + 0.5 * rhob * qsqb  # exterior total energy
    
    # exterior boundary state
    Ub = np.array([
        rhob, 
        rhob * vvecb[0],
        rhob * vvecb[1], 
        rhoEb 
    ])
    
    Fhatb, *_ = F_from_U(Ub, n, gamma)
    
    # compute maximum wave speed 
    u = vvecb @ n # incorporate the boundary state by using velocity vector at boundary (exterior velocity) dotted with the normal
    smax = np.abs(u) + cb
    return Fhatb, smax

# NOTE BE SURE TO PASS IN POUT FROM THE MAIN SOLVER
def subsonic_outflow(Uplus: np.ndarray, n: float, pb: float, gamma: float):
    """
    Subsonic outflow boundary condition. Boundary static pressure (pb) is specified (in this case pout)
    Inputs: 
    -------
    Uplus: np.ndarray
        Interior state vector
    n: float
        Normal vector pointing out of domain
    pb: float
        Boundary static pressure
    gamma: float
        Ratio of specific heats
    """
    Uplus = Uutil(Uplus, gamma)
    pplus = Uplus.p
    rhoplus = Uplus.rho
    vvecplus = Uplus.vvec
    cplus = Uplus.c
    
    Splus = pplus / rhoplus**gamma # interior entropy
    rhob = (pb / Splus)**(1/gamma) # exterior density
    
    unplus = vvecplus @ n
    Jplus = unplus + ((2 * cplus) / (gamma - 1))
    cb = np.sqrt(gamma * pb / rhob)
    
    unb = Jplus - ((2 * cb) / (gamma - 1)) # boundary normal velocity
    
    vvecb = vvecplus - ((vvecplus @ n) * n) + (unb * n)
    rhoEb = (pb / (gamma - 1)) + 0.5 * rhob * (vvecb @ vvecb)
    
    Ub = np.array([
        rhob, 
        rhob * vvecb[0],
        rhob * vvecb[1],
        rhoEb
    ])
    
    # compute maximum wave speed 
    u = vvecb @ n # incorporate the boundary state by using velocity vector at boundary (exterior velocity) dotted with the normal
    smax = np.abs(u) + cb
    Fhatb, *_ = F_from_U(Ub, n, gamma)
    return Fhatb, smax

def F_from_U(U, n, gamma):
    """
    Calculates flux vector for a given state vector
    """
    rho = U[0]
    rho_u = U[1]
    rho_v = U[2]
    rho_E = U[3]
    
    u = rho_u / rho # u, velocity in x-dir
    v = rho_v / rho # v, velocity in y-dir
    vvec = np.array([u, v]) # velocity vector
    vdotn = (u * n[0]) + (v * n[1]) # velocity vector dotted with normal 
    
    qsq = u**2 + v**2
    q = np.sqrt(qsq) # velocity magnitude 
    P = (gamma - 1) * (rho_E - 0.5 * rho * qsq)
    E = rho_E / rho 
    H = E + P / rho
    
    # NOTE Flux is actually F dot n! 
    F = np.array([
        rho * vdotn, 
        rho_u * vdotn + P * n[0], 
        rho_v * vdotn + P * n[1], 
        rho * H * vdotn,
    ])
    
    return F, rho, vvec, H, P 
    
def fluxRoe(UL, UR, n, gamma):
    """
    Roe flux implementation with entropy fix
    """
    FL, rhoL, vvecL, HL, _ = F_from_U(UL, n, gamma)
    FR, rhoR, vvecR, HR, _ = F_from_U(UR, n, gamma)
    
    sqrt_rhoL = np.sqrt(rhoL)
    sqrt_rhoR = np.sqrt(rhoR)
    
    # Roe states for Euler equations (AE623 course notes pg. 59)
    vvec_roe = (sqrt_rhoL * vvecL + sqrt_rhoR * vvecR) / (sqrt_rhoL + sqrt_rhoR)
    H_roe = (sqrt_rhoL * HL + sqrt_rhoR * HR) / (sqrt_rhoL + sqrt_rhoR)
    
    u = vvec_roe @ n # given in pg 59-60 of 623 notes
    qsq = vvec_roe[0]**2 + vvec_roe[1]**2 
    
    # must back out speed of sound, c, from roe state
    # E_roe = (H_roe + (0.5 * (gamma - 1) * qsq)) / gamma # E recalculated from Hroe avg state
    # c = np.sqrt(gamma * (H_roe - E_roe)) 
    c = np.sqrt((gamma - 1) * (H_roe - 0.5 * qsq)) 
    
    # eigenvalues of flux jacobian for Euler equations 
    # L1 = u + c; L2 = u - c; L3 = u; L4 = u
    eigs = np.array([u + c, u - c, u, u])
    mag_eigs = np.abs(eigs)
    
    # entropy fix
    eps = 0.1 * c
    mask = mag_eigs < eps # boolean mask array T/F if each eig magnitude is less than eps NOTE OR CAN USE np.where(condition, new_value, old_value)
    eigs[mask] = (eps**2 + eigs[mask]**2) / (2 * eps)
    
    # maximum eigenvalue used as the maximum wave speed for time step calculations     
    smax_tm = np.max(mag_eigs) # NOTE for wavespeed always use absolute value 
    
    s1 = 0.5 * (mag_eigs[0] + mag_eigs[1])
    s2 = 0.5 * (mag_eigs[0] - mag_eigs[1])

    dU = UR - UL # difference in right and left state vectors  
    drho = dU[0]
    drho_u = dU[1]
    drho_v = dU[2]
    drho_E = dU[3]
    
    drhovvec = np.array([drho_u, drho_v])
    G1 = (gamma - 1) * ((0.5 * qsq**2 * drho) - (vvec_roe @ drhovvec) + drho_E)
    G2 = -(u * drho) + (drhovvec @ n)
    C1 = (G1 / c**2) * (s1 - mag_eigs[2]) + (G2 / c) * s2
    C2 = (G1 / c) * s2 + (s1 - mag_eigs[2]) * G2 
    
    Fminus = np.array([
        mag_eigs[2] * drho + C1, 
        (mag_eigs[2] * drhovvec[0]) + (C1 * vvec_roe[0]) + (C2 * n[0]), 
        (mag_eigs[2] * drhovvec[1]) + (C1 * vvec_roe[1]) + (C2 * n[1]), 
        (mag_eigs[2] * drho_E) + (C1 * H_roe) + (C2 * u)
    ])
    
    F = 0.5 * (FL + FR) - 0.5 * Fminus
    return F, smax_tm

def fluxHLLE(UL, UR, n, gamma):
    """
    HLLE flux function (Fidkowski, pg 60)
    """
    FL, rhoL, vvecL, HL, PL = F_from_U(UL)
    FR, rhoR, vvecR, HR, PR = F_from_U(UR)
    
    uL = vvecL @ n 
    uR = vvecR @ n
    
    cL = np.sqrt(gamma * PL / rhoL)
    cR = np.sqrt(gamma * PR / rhoR)
    
    sLmin = min(0, uL - cL)
    sRmin = min(0, uR - cR)
    sLmax = max(0, uL + cL)
    sRmax = max(0, uR + cR)

    smin = min(sLmin, sRmin)
    smax = max(sLmax, sRmax)

    Fhat = 0.5 * (FL + FR) - 0.5 * ((smax + smin)/(smax - smin)) * (FR - FL) + ((smax * smin) / (smax - smin)) * (uR - uL)
    smax_tm = max(np.abs(uL) + cL, np.abs(uR) + cR) # maximum wave speed for time step calculations (AE 623 course notes, pg 60)
    
    return Fhat, smax_tm
