import numpy as np

# TODO: write a Rusanov flux and a wall flux
def fluxWall(UL, n, g):
    h = UL[0]
    p = 0.5*g*h**2 # boundary pressure taken from L (interior) element
    Fwall = np.zeros(3)
    Fwall[0] = 0
    Fwall[1] = n[0]*p
    Fwall[2] = n[1]*p
    
    # wave speed for CFL (treat like left state) 
    u = UL[1]/h if h>0 else 0.0
    v = UL[2]/h if h>0 else 0.0
    
    un = u*n[0] + v*n[1]
    c = np.sqrt(g*h)
    smax = np.abs(un) + c
    
    return Fwall, smax

 #-----------------------------------------
def fluxRoe(UL, UR, n, g):

    # process left state
    hL = UL[0]
    if (hL<0): print('Non-physical state!')
    uL = UL[1]/hL
    vL = UL[2]/hL
    unL = uL*n[0] + vL*n[1] # velocity dotted with normal
    pL = 0.5*g*hL**2

    # left flux
    FL = np.zeros(3)
    FL[0] = hL*unL
    FL[1] = hL*uL*unL + pL*n[0] 
    FL[2] = hL*vL*unL + pL*n[1]

    # process right state
    hR = UR[0]
    if (hR<0): print('Non-physical state!')
    uR = UR[1]/hR
    vR = UR[2]/hR
    unR = uR*n[0] + vR*n[1]
    pR = 0.5*g*hR**2

    # right flux
    FR = np.zeros(3)
    FR[0] = hR*unR
    FR[1] = hR*uR*unR + pR*n[0] 
    FR[2] = hR*vR*unR + pR*n[1]

    # difference in states
    du = UR - UL

    # Roe average
    h = 0.5*(hL+hR)
    sqhL = np.sqrt(hL); sqhR = np.sqrt(hR)
    u = (sqhL*uL + sqhR*uR)/(sqhL+sqhR)
    v = (sqhL*vL + sqhR*vR)/(sqhL+sqhR)
    un = u*n[0] + v*n[1]
    c = np.sqrt(g*h)
    
    # eigenvalues
    l = np.zeros(3)
    l[0] = un; l[1] = un-c; l[2] = un+c
    
    # entropy fix
    epsilon = c*.05
    for i in range(3):
        if ((l[i]<epsilon) and (l[i]>-epsilon)):
            l[i] = 0.5*(epsilon + l[i]*l[i]/epsilon)

    # absolute values of eigenvalues
    l = abs(l)

    # combination of eigenvalues
    s2 = 0.5*(l[1]-l[2])
    s3 = 0.5*(l[1]+l[2]-2*l[0])

    # eigenvetor product generator
    G1 = du[0]*un - du[1]*n[0] - du[2]*n[1]

    # functions of G1, s2, s3
    C1 = du[0]*s3 + G1*s2/c
    C2 = G1*s3 + s2*du[0]*c

    # flux assembly
    F = np.zeros(3)
    F[0] = 0.5*(FL[0]+FR[0])-0.5*(l[0]*du[0] + C1   )
    F[1] = 0.5*(FL[1]+FR[1])-0.5*(l[0]*du[1] + C1*u - C2*n[0])
    F[2] = 0.5*(FL[2]+FR[2])-0.5*(l[0]*du[2] + C1*v - C2*n[1])

    # max wave speed
    smag = max(l)

    return F, smag

def fluxRusanov(UL, UR, n, g):
    """
    Compute Rusanov (local Lax–Friedrichs) flux for 2D shallow water equations.
    Uses local wave speed |un| + sqrt(g*h) as dissipation term.
    """
    # --- Left state ---
    hL = UL[0]
    if hL <= 0:
        raise ValueError("Nonphysical left state (hL <= 0)")

    uL = UL[1] / hL
    vL = UL[2] / hL
    unL = uL * n[0] + vL * n[1]
    pL = 0.5 * g * hL**2

    FL = np.zeros(3)
    FL[0] = hL * unL
    FL[1] = hL * uL * unL + pL * n[0]
    FL[2] = hL * vL * unL + pL * n[1]

    # --- Right state ---
    hR = UR[0]
    if hR <= 0:
        raise ValueError("Nonphysical right state (hR <= 0)")

    uR = UR[1] / hR
    vR = UR[2] / hR
    unR = uR * n[0] + vR * n[1]
    pR = 0.5 * g * hR**2

    FR = np.zeros(3)
    FR[0] = hR * unR
    FR[1] = hR * uR * unR + pR * n[0]
    FR[2] = hR * vR * unR + pR * n[1]

    # --- Compute local max wave speed ---
    cL = np.sqrt(g * hL)
    cR = np.sqrt(g * hR)
    smax = max(abs(unL) + cL, abs(unR) + cR)

    # --- Rusanov (Lax–Friedrichs) flux ---
    F = 0.5 * (FL + FR) - 0.5 * smax * (UR - UL)

    return F, smax

