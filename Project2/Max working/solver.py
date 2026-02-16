import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.collections import LineCollection
from dataclasses import dataclass, field 
from readgri import readgri 
from pathlib import Path
# from viz import plotmesh
from fluxes import *
import time
from typing import Optional
from utilities import Uutil

"""
Max Mah 
2/11/26 - 2/13/26

Main first order finite volume solver
"""
@dataclass
class FiniteVol():
    U0: np.ndarray | None = None # if the user passes a state array U0, use it. Otherwise generate initial condition from inlet data
    # U0: Optional[np.ndarray] = None 
    meshname: str = '2k.gri'
    fluxname: str = 'roe'
    gamma: float = 1.4 # ratio of specific heats 
    CFL: float = 1.0 
    rtol: float = 1e-6 # convergence tolerance 
    
    # initialization quantities
    rho0: float = 1.0 # inlet stagnation density
    a0: float = 1.0 # inlet stagnation speed of sound
    alpha: float = 50 # angle of attack, deg 
    Minf: float = 0.1 # guessed Mach number for initial condition
    debug_checks: bool = False
    print_every: int = 50
    # pout: float = 0.7 * p0 # outflow static pressure TODO are we supposed to specify this???
    
    def __post_init__(self):
        # calculate some initialization quantities 
        self.p0 = (self.rho0 * self.a0**2) / self.gamma      # inlet stagnation pressure 
        self.pout = 0.7 * self.p0     # outflow static pressure 
        
        # preprocess mesh using utils     
        dir_base = Path(__file__).resolve().parent
        grifile = dir_base.joinpath(self.meshname)
        grifile.parent.mkdir(parents=True, exist_ok=True) # make sure script directory exists

        # read in .gri file and get associated data structures
        self.Mesh = readgri(grifile) 
        # plotmesh(meshdata, '2k', savefig=True)

        # convert alpha to rad
        self.alpha = np.deg2rad(self.alpha)

        self.V = self.Mesh['V']
        self.E = self.Mesh['E']
        self.IE = self.Mesh['IE']
        self.BE = self.Mesh['BE']
        self.Bname = self.Mesh['Bname'] 
        self.PeriodicGroups = self.Mesh.get('PeriodicGroups', [])
        self.PeriodicPairs = self.Mesh.get('PeriodicPairs', np.zeros((0, 2), dtype=int))
        
        # centroids and areas indexed by element
        self.Centroid = self.Mesh['Centroid']
        self.Area = self.Mesh['Area']

        # inormals and bnormals indexed by IE, BE respectively 
        self.In = self.Mesh['In']
        self.Bn = self.Mesh['Bn']

        # compute midpoints of edges and store in self.IE_mp and self.BE_mp
        self.IE_mp = 0.5 * (self.V[self.IE[:, 0], :] + self.V[self.IE[:, 1], :])
        self.BE_mp = 0.5 * (self.V[self.BE[:, 0], :] + self.V[self.BE[:, 1], :])
        
        # iteration quantities 
        self.Ne = self.E.shape[0] # number of elements
        self.Ni = self.IE.shape[0] # number of interior edges/faces
        self.Nb = self.BE.shape[0] # number of boundary edges/faces 

        # Precompute boundary condition code per BE row for fast branching in residual loop.
        # 0=inflow, 1=outflow, 2=wall
        self.bc_code = np.empty(self.Nb, dtype=np.int8)
        for i in range(self.Nb):
            bidx = int(self.BE[i, 3])
            bname = self.Bname[bidx]
            if bname == 'inflow':
                self.bc_code[i] = 0
            elif bname == 'outflow':
                self.bc_code[i] = 1
            elif bname == 'wall':
                self.bc_code[i] = 2
            else:
                raise ValueError(f"Unsupported boundary name '{bname}' at BE row {i}")
    
        # set fluxfunc
        if self.fluxname.lower()=='roe':
            self.flux = fluxRoe
        elif self.fluxname.lower()=='hlle':
            self.flux = fluxHLLE
        else: 
            raise ValueError(f"Unknown fluxfunc '{self.fluxname}'. Expected 'roe' or 'hlle'")
        
        if self.U0 is None:
        # initialize the flow to uniform state obtained using inlet conditions and guessed Mach number
            self.set_IC()
        else:
            self.validate_IC()

    def validate_IC(self):
        # ensures correct shape of passed in initial state vector U0
        if self.U0.shape != (self.Ne, 4):
            raise ValueError(f"Provided IC has shape {self.U0.shape}, expected {(self.Ne, 4)}")

    def load_state_from_npz(self, npz_path: str, key: str = 'U', set_as_initial: bool = True):
        """
        Load a state array from an .npz file and optionally set it as U0.
        """
        data = np.load(npz_path)
        if key not in data:
            raise KeyError(f"Key '{key}' not found in {npz_path}")
        U_loaded = np.asarray(data[key], dtype=float)
        if U_loaded.shape != (self.Ne, 4):
            raise ValueError(
                f"Loaded state shape {U_loaded.shape} does not match mesh shape {(self.Ne, 4)}"
            )
        if set_as_initial:
            self.U0 = U_loaded.copy()
        return U_loaded
        
    def testplotgri(self):
        plotmesh(Mesh=self.Mesh, fname='testmeshplot', savefig=True)
        
    def set_IC(self):
        """
        Initializes uniform state (Uinf or U0) from inlet conditions and guessed Mach number (Minf) 
        """
        # self.alpha *= np.pi/180
        
        rho_u = self.rho0 * self.Minf * self.a0 * np.cos(self.alpha)
        rho_v = self.rho0 * self.Minf * self.a0 * np.sin(self.alpha)
        rho_E = self.rho0 * (self.a0**2/((self.gamma - 1)*self.gamma) + (0.5 * self.Minf**2))
        U0 = np.zeros([self.Ne, 4]) # NOTE in matlab code its stored as a column vector (repmat(U0, Ne, 1))
        U0[:, 0] = self.rho0
        U0[:, 1] = rho_u
        U0[:, 2] = rho_v
        U0[:, 3] = rho_E
        self.U0 = U0 
        

    def solve_unsteady(self, runtime: str = True, itercap: int = 1e6):
        """
        Main time stepping loop with local time stepping 
        """
        # Record runtime 
        if runtime==True:
            start = time.time()
        
        U = self.U0.copy()
        Rhist = [] 
        niter = 0
        converged = False
        fail_reason = None

        ts = [0]
        
        while True:

            # determine dt









            # calculate residual
            R, sdl = self.calc_residual(U)
            # np.isfinite(x) reutrns True where x is real finite number and False for Nan, +inf, -inf. This catches numerical breakdown early.
            if (not np.all(np.isfinite(R))) or (not np.all(np.isfinite(sdl))):
                fail_reason = f"Non-finite residual or sdl detected at iter {niter}"
                break

            # RnormL1 = np.linalg.norm(R, ord=1) # monitor L1 norm of the discrete residual vector
            RnormL1 = np.sum(np.abs(R)) # global L1 residual norm
            if not np.isfinite(RnormL1):
                fail_reason = f"Residual norm became non-finite at iter {niter}"
                break
             
            Rhist.append(RnormL1)
            if niter % max(1, int(self.print_every)) == 0: 
                _, rho_min_now, p_min_now = self._state_is_physical(U)
                print(
                    f'niter: {niter}, RnormL1: {RnormL1:.6e}, '
                    f'min(rho): {rho_min_now:.6e}, min(p): {p_min_now:.6e}'
                )
            if RnormL1 <= self.rtol:
                converged = True
                break
            # update the state via local time stepping 
            dt = self.calc_dt(sdl, usecase='steady') # this ought to be an array with time steps for each element
            if not np.all(np.isfinite(dt)):
                fail_reason = f"Non-finite local dt detected at iter {niter}"
                break
            # TODO check the broadcasting on this 
            U -= dt[:, None] * (R / self.Area[:, None]) # TODO right now using FE for first order, build out additional time integration SSP-RK-.. later
            ok_state, rho_min, p_min = self._state_is_physical(U)
            if not ok_state:
                fail_reason = (
                    f"Non-physical state at iter {niter}: min(rho)={rho_min:.6e}, "
                    f"min(p)={p_min:.6e}"
                )
                self._print_state_diagnostics(U, niter, RnormL1)
                break
            niter += 1 # increment counter 
        
        if converged:
            print(f"Converged in {niter} iterations with RnormL1={RnormL1:.6e}")
        elif fail_reason is not None:
            print(f"WARNING: Solver aborted: {fail_reason}")
        elif niter==itercap:
            print("WARNING: Iteration cap reached without convergence")
            
        # store final state, time history of state
        self.U = U 
        self.Rhist = Rhist 
        
        # save only converged runs to avoid silently writing corrupted/nonphysical states
        if converged:
            datafile = f"{self.meshname}_{self.fluxname}_results.npz"
            np.savez(
                datafile,
                fluxfunc=self.fluxname,
                U=self.U,
                Rhist=self.Rhist
            )
            print(f"Saved results to {datafile}")
        else:
            print("Did not save results because solve did not converge cleanly.")

    def F_dudt(self, U, first_order=False, limited=False):
        if first_order:
            R, _ = self.calc_residual(U)
        else:
            R, _ = self.calc_residual_second_order(U, limited=limited)
        A = self.Area
        A = np.column_stack([A, A, A, A]) # for broadcasting?
        return - R / A

    def solve_steady(
        self,
        runtime: str = True,
        itercap: int = 1e6,
        first_order=False,
        warm_start_next: bool = True,
        init_npz: str | None = None,
        limited=False
    ):
        """
        Main time stepping loop with local time stepping 
        """
        # Record runtime 
        if runtime==True:
            start = time.time()
        
        if init_npz is not None:
            self.load_state_from_npz(init_npz, key='U', set_as_initial=True)
        U = self.U0.copy()
        Rhist = [] 
        niter = 0
        converged = False
        fail_reason = None
        
        # TODO one issue i see is R is a matrix [Ne x 4] not a vector...
        # is the L1 norm gonna be right? 
        
        while niter < itercap:
            # calculate residual
            if first_order:
                R, sdl = self.calc_residual(U)
            else:
                R, sdl = self.calc_residual_second_order(U, limited=limited)

            # np.isfinite(x) reutrns True where x is real finite number and False for Nan, +inf, -inf. This catches numerical breakdown early.
            if (not np.all(np.isfinite(R))) or (not np.all(np.isfinite(sdl))):
                fail_reason = f"Non-finite residual or sdl detected at iter {niter}"
                break

            # RnormL1 = np.linalg.norm(R, ord=1) # monitor L1 norm of the discrete residual vector
            RnormL1 = np.sum(np.abs(R)) # global L1 residual norm

            if not np.isfinite(RnormL1):
                fail_reason = f"Residual norm became non-finite at iter {niter}"
                break
             
            Rhist.append(RnormL1)
            if niter % max(1, int(self.print_every)) == 0: 
                _, rho_min_now, p_min_now = self._state_is_physical(U)
                print(
                    f'niter: {niter}, RnormL1: {RnormL1:.6e}, '
                    f'min(rho): {rho_min_now:.6e}, min(p): {p_min_now:.6e}'
                )
            if RnormL1 <= self.rtol:
                converged = True
                break

            # update the state via local time stepping 
            dt = self.calc_dt(sdl, usecase='steady') # this ought to be an array with time steps for each element
            if (not first_order) and limited:
                dt *= 0.2
            if not np.all(np.isfinite(dt)):
                fail_reason = f"Non-finite local dt detected at iter {niter}"
                break
            # TODO check the broadcasting on this 

            # TODO right now this is just FE for first order. build out additional time integration SSP-RK2 and RK3 later
            # U -= dt[:, None] * (R / self.Area[:, None]) 
            # U = FiniteVol.fe(un=U, dt=dt[:, None], F=-R/self.Area[:, None])
            U = self.ssp_rk2(un=U, dt=dt[:, None], first_order=first_order, limited=limited)



            ok_state, rho_min, p_min = self._state_is_physical(U)
            if not ok_state:
                fail_reason = (
                    f"Non-physical state at iter {niter}: min(rho)={rho_min:.6e}, "
                    f"min(p)={p_min:.6e}"
                )
                self._print_state_diagnostics(U, niter, RnormL1)
                break
            niter += 1 # increment counter 
        
        if converged:
            print(f"Converged in {niter} iterations with RnormL1={RnormL1:.6e}")
        elif fail_reason is not None:
            print(f"WARNING: Solver aborted: {fail_reason}")
        elif niter==itercap:
            print("WARNING: Iteration cap reached without convergence")
            
        # store final state, time history of state
        self.U = U 
        self.Rhist = Rhist 
        if warm_start_next:
            self.U0 = U.copy()
        
        # save only converged runs to avoid silently writing corrupted/nonphysical states
        if converged:
            datafile = f"{self.meshname}_{self.fluxname}_results.npz"
            np.savez(
                datafile,
                fluxfunc=self.fluxname,
                U=self.U,
                Rhist=self.Rhist
            )
            print(f"Saved results to {datafile}")
        else:
            print("Did not save results because solve did not converge cleanly.")

    # NOTE THESE STATIC METHODS ARE JUST FOR STEADY TIME STEPPING ONLY BECAUSE OF THE INPUT SHAPES THEY ASSUME CHANGE THIS SHIT LATER FOR UNSTEADY
    # @staticmethod
    # def fe(un: np.ndarray, dt: np.ndarray, F: np.ndarray):
    #     unp1 = un + dt * F
    #     return unp1

    def ssp_rk2(self, un: np.ndarray, dt: float, first_order=True, limited=False):
        """
        Inputs: 
        -------
        un: np.ndarray
            state at current time n 
        dt: float
            time step size 
        F: function
            function in udot = F(u) for ODE time marching
        """
        u1 = un + dt * self.F_dudt(un, first_order=first_order, limited=limited)
        unp1 = (0.5 * un) + (0.5 * (u1 + dt * self.F_dudt(u1, first_order=first_order, limited=limited)))
        return unp1
    
    def ssp_rk3(self, un: np.ndarray, dt: float, first_order=True, limited=False): 
        """
        Inputs: 
        -------
        un: np.ndarray
            state at current time n 
        dt: float
            time step size 
        F: function
            function in udot = F(u) for ODE time marching
        """
        u1 = un + (dt * self.F_dudt(un, first_order=first_order, limited=limited)) 
        u2 = 0.75 * un + 0.25 * (u1 + dt * self.F_dudt(u1, first_order=first_order, limited=limited))
        unp1 = (1/3)*un + (2/3)*(u2 + dt * self.F_dudt(u2, first_order=first_order, limited=limited))
        return unp1

    def calc_residual_second_order(self, U, limited=False):
        """
        2nd order residual
        """
        # get the b name
        R = np.zeros([self.Ne, 4]) # residual size Ne x 4 (indexed by elements). **ZERO out residual every time step
        sdl = np.zeros(self.Ne) # size Ne x 1 (one scalar value per element )
        gradU = np.zeros([self.Ne, 4, 2])
        IE = self.IE
        In = self.In
        BE = self.BE
        Bn = self.Bn
        flux = self.flux
        gamma = self.gamma
        bc_code = self.bc_code
        # interior edge flux contributions

        # gererate gradient by looping over interior edges and boundary edges
        for i in range(self.Ni):
            elemL = IE[i, 2]
            elemR = IE[i, 3]
            ilen = In[i, 2]

            u_hat = (1/2) * (U[elemL] + U[elemR])
            n = In[i, :2]

            gradU[elemL] += np.outer(u_hat, n) * ilen
            gradU[elemR] -= np.outer(u_hat, n) * ilen

        for i in range(self.Nb):
            elemL = BE[i, 2]
            blen = Bn[i, 2]
            bn = Bn[i, :2] # normal vector at boundary (nx, ny)
            ilenb = Bn[i, 2]
            
            if bc_code[i] == 0: 
                Ub = subsonic_inflow(U[elemL, :], bn, self.rho0, self.a0, self.alpha, gamma)
                 
            elif bc_code[i] == 1: 
                Ub = subsonic_outflow(U[elemL, :], bn, self.pout, gamma)
                
            elif bc_code[i] == 2: 
                Ub = inviscid_wall_state(U[elemL, :], bn, gamma)

            else:
                raise ValueError(f"Not reading in boundary name correctly...unsupported boundary condition??")

            # Use boundary-face average for Green-Gauss consistency.
            u_hat = 0.5 * (U[elemL, :] + Ub)
            gradU[elemL] += np.outer(u_hat, bn) * ilenb
            

        # divide gradient by cell areas
        areas = self.Area[:, None, None]  # (Ne, 1, 1) -> broadcasts to (Ne, 4, 2)
        gradU /= areas
        # gradU_i *= 0.01

        gradU_orig = gradU

        # BJ limiting
        if limited:
            # Build cell-neighbor stencil from interior-edge connectivity.
            neighbors = [set([i]) for i in range(self.Ne)]
            for eL, eR in IE[:, 2:4]:
                neighbors[int(eL)].add(int(eR))
                neighbors[int(eR)].add(int(eL))

            for i, row in enumerate(self.E):
                # stencil min/max over this element + direct neighbors
                stencil = np.fromiter(neighbors[i], dtype=int)
                U_stencil = U[stencil, :]                  # (Ns, 4)
                Umin = U_stencil.min(axis=0)              # (4,)
                Umax = U_stencil.max(axis=0)              # (4,)

                # triangle node coordinates and vectors from centroid to each node
                nodes_xy = self.V[row, :]                 # (3, 2)
                r = (nodes_xy - self.Centroid[i]).T       # (2, 3)

                # unlimited reconstructed increments at triangle vertices
                dU_nodes = gradU[i] @ r                   # (4, 3)
                Ui = U[i, :]                              # (4,)
                delta_p = Umax - Ui
                delta_m = Umin - Ui

                # Barth-Jespersen limiter per conserved variable
                phi = np.ones(4)
                for j in range(3):
                    dU = dU_nodes[:, j]
                    phi_j = np.ones(4)

                    pos = dU > 0.0
                    neg = dU < 0.0

                    phi_j[pos] = np.minimum(1.0, delta_p[pos] / dU[pos])
                    phi_j[neg] = np.minimum(1.0, delta_m[neg] / dU[neg])

                    phi = np.minimum(phi, phi_j)

                phi = np.clip(phi, 0.0, 1.0)
                gradU[i] *= phi[:, None]





        # loop over interior edges (NOTE including periodic? might be some weirdness...)
        for i in range(self.Ni):
            elemL = IE[i, 2]
            elemR = IE[i, 3]
            ilen = In[i, 2]

            xL = self.Centroid[elemL]
            xR = self.Centroid[elemR]

            xf = self.IE_mp[i]
            dL = xf - xL    # (2,)
            dR = xf - xR    # (2,)

            ULf = U[elemL, :] + gradU[elemL] @ dL   # (4,)
            URf = U[elemR, :] + gradU[elemR] @ dR   # (4,)

            
            # call flux function on all interior edges 
            F, smag_i = flux(ULf, URf, In[i, :2], gamma)
            if self.debug_checks:
                F = np.asarray(F, dtype=float)
                if F.shape != (4,):
                    raise ValueError(f"Interior flux returned shape {F.shape}, expected (4,)")
            R[elemL, :] += F * ilen # sums F dot n * dl for all three interior edges on element
            R[elemR, :] -= F * ilen
            
            sdl[elemL] += smag_i * ilen
            sdl[elemR] += smag_i * ilen

        # loop over boundary edges 
        for i in range(self.Nb):
            elemL = BE[i, 2]
            blen = Bn[i, 2]
            bn = Bn[i, :2] # normal vector at boundary (nx, ny)

            xf = self.BE_mp[i]
            dL = xf - self.Centroid[elemL]
            ULf = U[elemL, :] + gradU[elemL] @ dL

            if bc_code[i] == 0:  # inflow
                Ub = subsonic_inflow(ULf, bn, self.rho0, self.a0, self.alpha, gamma)
                F, _ = flux(ULf, Ub, bn, gamma)
                smag_b = smagb(Ub, bn, gamma)

            elif bc_code[i] == 1:  # outflow
                Ub = subsonic_outflow(ULf, bn, self.pout, gamma)
                F, _ = flux(ULf, Ub, bn, gamma)
                smag_b = smagb(Ub, bn, gamma)

            elif bc_code[i] == 2:  # wall
                F, smag_b = inviscid_wall(ULf, bn, gamma)

            else:
                raise ValueError(f"Not reading in boundary name correctly...unsupported boundary condition??")
            if self.debug_checks:
                F = np.asarray(F, dtype=float)
                if F.shape != (4,):
                    raise ValueError(f"Boundary flux returned shape {F.shape}, expected (4,)")
            R[elemL, :] += F * blen # sums F dot n * dl for all three interior edges on element
            sdl[elemL] += smag_b * blen

        return R, sdl


    def calc_residual(self, U):
        """
        Calculates residual using the selected flux function
        Loops over internal edges (IE) and boundary edges (BE), calculates fluxes and increments residuals on respective L and R elements
        Also calculates edge-weighted average of wave speeds times edge length (sdl) for every edge visited
        """
        # get the b name
        R = np.zeros([self.Ne, 4]) # residual size Ne x 4 (indexed by elements). **ZERO out residual every time step
        sdl = np.zeros(self.Ne) # size Ne x 1 (one scalar value per element )
        IE = self.IE
        In = self.In
        BE = self.BE
        Bn = self.Bn
        flux = self.flux
        gamma = self.gamma
        bc_code = self.bc_code
        # interior edge flux contributions
        
        # loop over interior edges (NOTE including periodic? might be some weirdness...)
        for i in range(self.Ni):
            elemL = IE[i, 2]
            elemR = IE[i, 3]
            ilen = In[i, 2]
            
            # call flux function on all interior edges 
            F, smag_i = flux(U[elemL, :], U[elemR, :], In[i, :2], gamma)
            if self.debug_checks:
                F = np.asarray(F, dtype=float)
                if F.shape != (4,):
                    raise ValueError(f"Interior flux returned shape {F.shape}, expected (4,)")
            R[elemL, :] += F * ilen # sums F dot n * dl for all three interior edges on element
            R[elemR, :] -= F * ilen
            
            sdl[elemL] += smag_i * ilen
            sdl[elemR] += smag_i * ilen

        # loop over boundary edges 
        for i in range(self.Nb):
            elemL = BE[i, 2]
            blen = Bn[i, 2]
            bn = Bn[i, :2] # normal vector at boundary (nx, ny)
            
            # TODO come up with a way to return wave speed FOR ALL THREE!!! SEE PIAZZA
            if bc_code[i] == 0: 
                Ub = subsonic_inflow(U[elemL, :], bn, self.rho0, self.a0, self.alpha, gamma)
                F, _ = flux(U[elemL, :], Ub, bn, gamma) # NOTE if returned a wave speed from flux function, it would come from uL and uL not from boundary state by using velocity vector at boundary.
                smag_b = smagb(Ub, bn, gamma) # NOTE set the wavespeed using boundary state-based smax 
                 
            elif bc_code[i] == 1: 
                Ub = subsonic_outflow(U[elemL, :], bn, self.pout, gamma)
                F, _ = flux(U[elemL, :], Ub, bn, gamma)
                smag_b = smagb(Ub, bn, gamma)
                
            elif bc_code[i] == 2: 
                F, smag_b = inviscid_wall(U[elemL, :], bn, gamma)
            else:
                raise ValueError(f"Not reading in boundary name correctly...unsupported boundary condition??")
            if self.debug_checks:
                F = np.asarray(F, dtype=float)
                if F.shape != (4,):
                    raise ValueError(f"Boundary flux returned shape {F.shape}, expected (4,)")
            R[elemL, :] += F * blen # sums F dot n * dl for all three interior edges on element
            sdl[elemL] += smag_b * blen

        return R, sdl

    def plot_edge_normals(self, edge_set: str = 'both', scale: float = 0.35, stride: int = 1, figsize=(10, 4)):
        """
        Debug visualization for edge-normal orientation.
        Colors:
        - interior: green = points L->R, red = opposite
        - boundary: cyan = points outward from element, magenta = opposite
        """
        V = self.V
        E = self.E
        fig, ax = plt.subplots(figsize=figsize)
        ax.triplot(V[:, 0], V[:, 1], E, 'k-', linewidth=0.35, alpha=0.45)

        edge_set = edge_set.lower()
        stride = max(1, int(stride))

        if edge_set in ('both', 'interior'):
            ie = self.IE[::stride, :]
            ni = self.In[::stride, :2]
            mids_i = 0.5 * (V[ie[:, 0], :] + V[ie[:, 1], :])
            d_lr = self.Centroid[ie[:, 3], :] - self.Centroid[ie[:, 2], :]
            ok_i = np.einsum('ij,ij->i', ni, d_lr) > 0.0

            if np.any(ok_i):
                ax.quiver(
                    mids_i[ok_i, 0], mids_i[ok_i, 1], ni[ok_i, 0], ni[ok_i, 1],
                    color='tab:green', angles='xy', scale_units='xy', scale=1.0 / scale,
                    width=0.0022, label='interior normal OK'
                )
            if np.any(~ok_i):
                ax.quiver(
                    mids_i[~ok_i, 0], mids_i[~ok_i, 1], ni[~ok_i, 0], ni[~ok_i, 1],
                    color='tab:red', angles='xy', scale_units='xy', scale=1.0 / scale,
                    width=0.0028, label='interior normal flipped'
                )

        if edge_set in ('both', 'boundary'):
            be = self.BE[::stride, :]
            nb = self.Bn[::stride, :2]
            mids_b = 0.5 * (V[be[:, 0], :] + V[be[:, 1], :])
            c_to_mid = mids_b - self.Centroid[be[:, 2], :]
            ok_b = np.einsum('ij,ij->i', nb, c_to_mid) > 0.0

            if np.any(ok_b):
                ax.quiver(
                    mids_b[ok_b, 0], mids_b[ok_b, 1], nb[ok_b, 0], nb[ok_b, 1],
                    color='tab:cyan', angles='xy', scale_units='xy', scale=1.0 / scale,
                    width=0.0022, label='boundary normal outward'
                )
            if np.any(~ok_b):
                ax.quiver(
                    mids_b[~ok_b, 0], mids_b[~ok_b, 1], nb[~ok_b, 0], nb[~ok_b, 1],
                    color='magenta', angles='xy', scale_units='xy', scale=1.0 / scale,
                    width=0.0028, label='boundary normal flipped'
                )

        ax.set_aspect('equal', adjustable='box')
        ax.set_title('Edge-Normal Orientation Check')
        ax.legend(loc='best', fontsize=8)
        fig.tight_layout()
        plt.show()

    def plot_normals_with_mesh_labels(
        self,
        edge_stride: int = 1,
        show_node_ids: bool = True,
        show_elem_ids: bool = True,
        node_id_stride: int = 1,
        elem_id_stride: int = 1,
        node_fontsize: int = 6,
        elem_fontsize: int = 7,
        figsize=(10, 4),
    ):
        """
        Plot mesh with interior normals (blue), boundary normals (red),
        node IDs (black), and element IDs (lime).
        """
        V = self.V
        E = self.E
        edge_stride = max(1, int(edge_stride))
        node_id_stride = max(1, int(node_id_stride))
        elem_id_stride = max(1, int(elem_id_stride))

        ie = self.IE[::edge_stride, :]
        be = self.BE[::edge_stride, :]
        mid_in = 0.5 * (V[ie[:, 0], :] + V[ie[:, 1], :])
        mid_be = 0.5 * (V[be[:, 0], :] + V[be[:, 1], :])
        In = self.In[::edge_stride, :2]
        Bn = self.Bn[::edge_stride, :2]

        fig, ax = plt.subplots(figsize=figsize)
        ax.quiver(
            mid_in[:, 0], mid_in[:, 1], In[:, 0], In[:, 1],
            color='tab:blue', angles='xy', scale_units='xy', scale=1, width=0.002
        )
        ax.quiver(
            mid_be[:, 0], mid_be[:, 1], Bn[:, 0], Bn[:, 1],
            color='tab:red', angles='xy', scale_units='xy', scale=1, width=0.002
        )
        ax.set_aspect('equal', adjustable='box')
        ax.set_title('Interior normals (blue) and Boundary normals (red)')
        ax.triplot(V[:, 0], V[:, 1], E, color='tab:blue', linewidth=0.9)

        if show_node_ids:
            node_ids = np.arange(0, V.shape[0], node_id_stride, dtype=int)
            for nid in node_ids:
                ax.text(V[nid, 0], V[nid, 1], f"{nid + 1}", color='k', fontsize=node_fontsize)

        if show_elem_ids:
            elem_ids = np.arange(0, E.shape[0], elem_id_stride, dtype=int)
            for eid in elem_ids:
                n1, n2, n3 = E[eid]
                cent = (V[n1] + V[n2] + V[n3]) / 3.0
                ax.text(cent[0], cent[1], f"{eid + 1}", color='lime', fontsize=elem_fontsize)

        fig.tight_layout()
        plt.show()

    def plot_interior_edge_indices(self, stride: int = 1, fontsize: int = 7, figsize=(10, 4)):
        """
        Plot interior edges and annotate each edge midpoint with its IE row index.
        Useful for validating IE ordering/read-in.
        """
        stride = max(1, int(stride))
        ie_idx = np.arange(0, self.Ni, stride, dtype=int)
        ie = self.IE[ie_idx, :]
        mids = 0.5 * (self.V[ie[:, 0], :] + self.V[ie[:, 1], :])

        fig, ax = plt.subplots(figsize=figsize)

        # Draw only interior edges.
        for n1, n2, _, _ in ie:
            p1 = self.V[n1, :]
            p2 = self.V[n2, :]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='k', linewidth=0.6, alpha=0.8)

        # Label each interior edge by its IE row index.
        for i, (x, y) in zip(ie_idx, mids):
            ax.text(
                x, y, f"{i}",
                color='tab:red', fontsize=fontsize, ha='center', va='center'
            )

        ax.set_aspect('equal', adjustable='box')
        ax.set_title('Interior Edge Indices (IE rows)')
        fig.tight_layout()
        plt.show()

    def plot_boundary_edge_indices(self, stride: int = 1, fontsize: int = 7, figsize=(10, 4)):
        """
        Plot boundary edges and annotate each edge midpoint with its BE row index.
        Useful for validating BE ordering/read-in.
        """
        stride = max(1, int(stride))
        be_idx = np.arange(0, self.Nb, stride, dtype=int)
        be = self.BE[be_idx, :]
        mids = 0.5 * (self.V[be[:, 0], :] + self.V[be[:, 1], :])

        fig, ax = plt.subplots(figsize=figsize)

        # Draw only boundary edges.
        for n1, n2, _, _ in be:
            p1 = self.V[n1, :]
            p2 = self.V[n2, :]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='k', linewidth=0.9, alpha=0.9)

        # Label each boundary edge by its BE row index.
        for i, (x, y) in zip(be_idx, mids):
            ax.text(
                x, y, f"{i}",
                color='tab:blue', fontsize=fontsize, ha='center', va='center'
            )

        ax.set_aspect('equal', adjustable='box')
        ax.set_title('Boundary Edge Indices (BE rows)')
        fig.tight_layout()
        plt.show()

    def plot_periodic_pairs(self, group_idx: int = 0, pair_stride: int = 1, show_mesh: bool = False, figsize=(10, 4)):
        """
        Plot periodic node-pair links from the .gri PeriodicGroup section.
        Also overlays wall boundary edges in black.
        """
        pair_stride = max(1, int(pair_stride))
        fig, ax = plt.subplots(figsize=figsize)

        if show_mesh:
            ax.triplot(self.V[:, 0], self.V[:, 1], self.E, color='0.75', linewidth=0.25, alpha=0.6)

        # wall boundaries in black
        wall_segments = []
        for i in range(self.Nb):
            bidx = int(self.BE[i, 3])
            if self.Bname[bidx] != 'wall':
                continue
            n1, n2 = int(self.BE[i, 0]), int(self.BE[i, 1])
            wall_segments.append([self.V[n1, :], self.V[n2, :]])
        if len(wall_segments) > 0:
            lc_wall = LineCollection(wall_segments, colors='k', linewidths=1.6, alpha=0.95)
            ax.add_collection(lc_wall)

        if len(self.PeriodicGroups) == 0:
            ax.set_aspect('equal', adjustable='box')
            ax.set_title('No PeriodicGroup data found in mesh')
            fig.tight_layout()
            plt.show()
            return

        if group_idx < 0 or group_idx >= len(self.PeriodicGroups):
            raise ValueError(f"group_idx={group_idx} out of range [0, {len(self.PeriodicGroups)-1}]")

        pairs = np.asarray(self.PeriodicGroups[group_idx], dtype=int)
        if pairs.shape[0] == 0:
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(f'Periodic group {group_idx}: empty')
            fig.tight_layout()
            plt.show()
            return

        pairs = pairs[::pair_stride, :]
        periodic_segments = []
        for n_bot, n_top in pairs:
            periodic_segments.append([self.V[int(n_bot), :], self.V[int(n_top), :]])
        lc_periodic = LineCollection(periodic_segments, colors='tab:red', linewidths=1.2, alpha=0.9)
        ax.add_collection(lc_periodic)

        # periodic endpoints
        pxy = self.V[pairs.reshape(-1), :]
        ax.scatter(pxy[:, 0], pxy[:, 1], s=10, c='tab:red', alpha=0.85, zorder=3)

        ax.autoscale()
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'Periodic Node Pairs (group {group_idx}) + Wall Boundaries')
        fig.tight_layout()
        plt.show()

    def check_periodic_ie_count(self, strict: bool = False):
        """
        Check periodic IE reconstruction count.
        expected is the count reconstructed by periodic-edge pairing logic.
        Returns (added, expected, ok).
        """
        added = int(self.Mesh.get('PeriodicIEAdded', 0))
        expected = int(self.Mesh.get('PeriodicIEExpected', 0))
        npairs = int(self.PeriodicPairs.shape[0]) if hasattr(self, 'PeriodicPairs') else 0
        pairs_minus_one = max(0, npairs - 1)
        ok = (added == expected)
        msg = (
            f"Periodic IE count: added={added}, expected={expected} (pairing), "
            f"pairs-1={pairs_minus_one}, ok={ok}"
        )
        if strict and not ok:
            raise ValueError(msg)
        print(msg)
        return added, expected, ok

    def plot_edge_flux(
        self,
        U: np.ndarray,
        component: str = 'rhov',
        cmap: str = 'coolwarm',
        figsize=(10, 4),
        show_periodic_mates: bool = True,
    ):
        """
        Plot edge flux on all interior and boundary edges.
        Default component is y-momentum ('rhov').
        """
        comp_map = {'rho': 0, 'rhou': 1, 'rhov': 2, 'rhoe': 3, 'rho_e': 3}
        key = component.lower()
        if key not in comp_map:
            raise ValueError("component must be one of: 'rho', 'rhou', 'rhov', 'rhoE'")
        comp_idx = comp_map[key]

        segments = []
        edge_flux_vals = []

        # Optional periodic node map for plotting both geometric sides of a periodic interface.
        periodic_bt = {}
        periodic_tb = {}
        if show_periodic_mates and self.PeriodicPairs.size > 0:
            for nb, nt in np.asarray(self.PeriodicPairs, dtype=int):
                periodic_bt[int(nb)] = int(nt)
                periodic_tb[int(nt)] = int(nb)

        # Interior edges
        for i in range(self.Ni):
            n1, n2, elemL, elemR = self.IE[i, :]
            F, _ = self.flux(U[elemL, :], U[elemR, :], self.In[i, :2], self.gamma)
            F = np.asarray(F, dtype=float)
            if F.shape != (4,):
                raise ValueError(f"Interior flux returned shape {F.shape}, expected (4,)")
            segments.append([self.V[n1, :], self.V[n2, :]])
            edge_flux_vals.append(F[comp_idx])

            if show_periodic_mates and (len(periodic_bt) > 0):
                n1i, n2i = int(n1), int(n2)
                mate = None
                if (n1i in periodic_bt) and (n2i in periodic_bt):
                    mate = (periodic_bt[n1i], periodic_bt[n2i])
                elif (n1i in periodic_tb) and (n2i in periodic_tb):
                    mate = (periodic_tb[n1i], periodic_tb[n2i])
                if mate is not None:
                    m1, m2 = int(mate[0]), int(mate[1])
                    segments.append([self.V[m1, :], self.V[m2, :]])
                    edge_flux_vals.append(F[comp_idx])

        # Boundary edges
        for i in range(self.Nb):
            n1, n2, elemL, bidx = self.BE[i, :]
            bn = self.Bn[i, :2]
            bname = self.Bname[bidx]

            if bname == 'inflow':
                Ub = subsonic_inflow(U[elemL, :], bn, self.rho0, self.a0, self.alpha, self.gamma)
                F, _ = self.flux(U[elemL, :], Ub, bn, self.gamma)
            elif bname == 'outflow':
                Ub = subsonic_outflow(U[elemL, :], bn, self.pout, self.gamma)
                F, _ = self.flux(U[elemL, :], Ub, bn, self.gamma)
            elif bname == 'wall':
                F, _ = inviscid_wall(U[elemL, :], bn, self.gamma)
            else:
                raise ValueError(f"Unsupported boundary condition: {bname}")

            F = np.asarray(F, dtype=float)
            if F.shape != (4,):
                raise ValueError(f"Boundary flux returned shape {F.shape}, expected (4,)")
            segments.append([self.V[n1, :], self.V[n2, :]])
            edge_flux_vals.append(F[comp_idx])

        edge_flux_vals = np.asarray(edge_flux_vals, dtype=float)

        fig, ax = plt.subplots(figsize=figsize)
        lc = LineCollection(segments, cmap=cmap, linewidths=1.3)
        lc.set_array(edge_flux_vals)
        ax.add_collection(lc)
        ax.autoscale()
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'Edge Flux ({component})')
        cbar = fig.colorbar(lc, ax=ax)
        cbar.set_label(f'F_{component}')
        fig.tight_layout()
        plt.show()

    def plot_residual_field(self, U: np.ndarray, component: str = 'l1', log10: bool = True, cmap: str = 'viridis', figsize=(10, 4)):
        """
        Plot element-wise residual magnitude on the mesh.
        component options: 'l1', 'rho', 'rhou', 'rhov', 'rhoE'
        """
        R, _ = self.calc_residual(U)
        key = component.lower()
        if key == 'l1':
            val = np.sum(np.abs(R), axis=1)
            label = r'|R|_1 (per element)'
        elif key == 'rho':
            val = np.abs(R[:, 0]); label = r'|R_rho|'
        elif key == 'rhou':
            val = np.abs(R[:, 1]); label = r'|R_rhou|'
        elif key == 'rhov':
            val = np.abs(R[:, 2]); label = r'|R_rhov|'
        elif key in ('rhoe', 'rho_e'):
            val = np.abs(R[:, 3]); label = r'|R_rhoE|'
        else:
            raise ValueError("component must be one of: 'l1', 'rho', 'rhou', 'rhov', 'rhoE'")

        if log10:
            val_plot = np.log10(np.maximum(val, 1e-30))
            cbar_label = f'log10({label})'
        else:
            val_plot = val
            cbar_label = label

        fig, ax = plt.subplots(figsize=figsize)
        tpc = ax.tripcolor(
            self.V[:, 0], self.V[:, 1], self.E,
            facecolors=val_plot, shading='flat', cmap=cmap
        )
        ax.triplot(self.V[:, 0], self.V[:, 1], self.E, color='k', linewidth=0.15, alpha=0.25)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f'Residual Field ({component})')
        cbar = fig.colorbar(tpc, ax=ax)
        cbar.set_label(cbar_label)
        fig.tight_layout()
        plt.show()

    def plot_state_tripcolor(self, U: np.ndarray | None = None, cmap: str = 'viridis', figsize=(12, 8)):
        """
        Plot tripcolor fields for conservative state variables:
        rho, rho*u, rho*v, rhoE.
        Uses self.U by default; pass U explicitly to plot another state.
        """
        if U is None:
            if not hasattr(self, 'U'):
                raise ValueError("No solution state found on self.U. Pass U explicitly or run solve_steady first.")
            Uplot = self.U
        else:
            Uplot = U

        Uplot = np.asarray(Uplot, dtype=float)
        if Uplot.shape != (self.Ne, 4):
            raise ValueError(f"State shape must be {(self.Ne, 4)}, got {Uplot.shape}")

        labels = [r'$\rho$', r'$\rho u$', r'$\rho v$', r'$\rho E$']
        fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
        axes = axes.ravel()

        for i in range(4):
            ax = axes[i]
            tpc = ax.tripcolor(
                self.V[:, 0], self.V[:, 1], self.E,
                facecolors=Uplot[:, i], shading='flat', cmap=cmap
            )
            ax.triplot(self.V[:, 0], self.V[:, 1], self.E, color='k', linewidth=0.12, alpha=0.2)
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(labels[i])
            fig.colorbar(tpc, ax=ax, shrink=0.9)

        fig.suptitle('State Variables Tripcolor')
        plt.show()

    def plot_residual_history(
        self,
        logy: bool = True,
        figsize=(7, 4),
        npz_path: str | None = None,
        from_current_solve: bool = True,
    ):
        """
        Plot residual norm history versus iteration index.
        If from_current_solve=True, use self.Rhist from the current run.
        Otherwise, read Rhist from an .npz file.
        """
        y = None

        if from_current_solve:
            if not hasattr(self, 'Rhist') or len(self.Rhist) == 0:
                raise ValueError("No in-memory residual history found. Run solve_steady first or set from_current_solve=False.")
            y = np.asarray(self.Rhist, dtype=float)
        else:
            datafile = npz_path if npz_path is not None else f"{self.meshname}_{self.fluxname}_results.npz"
            data = np.load(datafile)
            if 'Rhist' not in data:
                raise KeyError(f"'Rhist' not found in {datafile}")
            y = np.asarray(data['Rhist'], dtype=float)

        if y.size == 0:
            raise ValueError("Residual history is empty.")

        x = np.arange(len(y), dtype=int)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(x, y, color='tab:blue', linewidth=1.5)
        if logy:
            ax.set_yscale('log')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Residual L1 Norm')
        ax.set_title('Residual History')
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        plt.show()

    
    def time_march_scheme(self):
        """
        Enables Forward Euler (FE) for testing first order solver 
        SSP-RK3
        SSP-RK4
        """
        pass
    
    def calc_dt(self, sdl, usecase: str):
        """calculate time step based on sdl"""
        tol_sdl = 1e-14
        # NOTE avoid division by zero in degenerate elements
        sdl = np.maximum(sdl, tol_sdl) 
        dt_i = (2 * self.Area * self.CFL) / sdl    
        dt = np.min(dt_i)
        
        if usecase.lower()=='steady':
            return dt_i
        elif usecase.lower()=='unsteady':
            return dt
        else: 
            raise ValueError(f"Expected 'steady' or 'unsteady' for usecase") 

    def _state_is_physical(self, U):
        rho = U[:, 0]
        rhou = U[:, 1]
        rhov = U[:, 2]
        rhoE = U[:, 3]

        finite_mask = np.all(np.isfinite(U), axis=1)
        rho_pos_mask = rho > 0.0
        valid_den_mask = finite_mask & rho_pos_mask
        p = np.full_like(rho, -np.inf, dtype=float)
        p[valid_den_mask] = (self.gamma - 1.0) * (
            rhoE[valid_den_mask]
            - 0.5 * (rhou[valid_den_mask] ** 2 + rhov[valid_den_mask] ** 2) / rho[valid_den_mask]
        )

        ok = np.all(finite_mask) and np.all(rho_pos_mask) and np.all(p > 0.0)
        rho_min = np.min(rho)
        p_min = np.min(p)
        return ok, rho_min, p_min

    def _compute_pressure(self, U):
        rho = U[:, 0]
        rhou = U[:, 1]
        rhov = U[:, 2]
        rhoE = U[:, 3]

        p = np.full(self.Ne, np.nan, dtype=float)
        valid = np.isfinite(rho) & (rho > 0.0) & np.isfinite(rhou) & np.isfinite(rhov) & np.isfinite(rhoE)
        p[valid] = (self.gamma - 1.0) * (rhoE[valid] - 0.5 * (rhou[valid] ** 2 + rhov[valid] ** 2) / rho[valid])
        return p

    def _print_state_diagnostics(self, U, niter, RnormL1):
        p = self._compute_pressure(U)
        rho = U[:, 0]

        idx_p = int(np.nanargmin(p))
        idx_rho = int(np.nanargmin(rho))

        bmask = self.BE[:, 2] == idx_p
        btags = [self.Bname[int(b)] for b in self.BE[bmask, 3]] if np.any(bmask) else []
        btags_str = ",".join(btags) if len(btags) > 0 else "none"

        c = self.Centroid[idx_p]
        print(
            "DIAG: "
            f"iter={niter}, RnormL1={RnormL1:.6e}, "
            f"worst_p_elem={idx_p}, p={p[idx_p]:.6e}, rho={rho[idx_p]:.6e}, "
            f"centroid=({c[0]:.6e},{c[1]:.6e}), boundary_tags={btags_str}"
        )
        c_rho = self.Centroid[idx_rho]
        print(
            "DIAG: "
            f"worst_rho_elem={idx_rho}, rho={rho[idx_rho]:.6e}, p={p[idx_rho]:.6e}, "
            f"centroid=({c_rho[0]:.6e},{c_rho[1]:.6e})"
        )
        
if __name__=="__main__":
    # solver = FiniteVol(meshname='2k.gri', fluxname='hlle', gamma=1.4, CFL=0.1)
    # solver = FiniteVol(meshname='2k.gri', fluxname='roe', gamma=1.4, CFL=0.5)

    grid = "8k"
    flux = "roe"

    solver = FiniteVol(meshname=f'{grid}.gri', fluxname=flux, gamma=1.4, CFL=0.1)
    # print('deez')
    # plotmesh(solver.Mesh, fname='testplot', savefig=True)
    # solver = FiniteVol(meshname='2k.gri', fluxname='hlle', gamma=1.4, CFL = 0.1)
    # solver.testplotgri()
    solver.plot_edge_flux(solver.U0, component='rhov', show_periodic_mates=False)

    # solver.plot_interior_edge_indices()               # all interior edges
    # solver.plot_interior_edge_indices(stride=5)       # every 5th IE entry
    # solver.plot_interior_edge_indices(fontsize=6)     # smaller labels
    # solver.plot_boundary_edge_indices()
    # solver.plot_boundary_edge_indices(stride=5, fontsize=6)

    # solver = FiniteVol(meshname='2k.gri', fluxname='hlle', gamma=1.4, CFL=0.1)
    solver.plot_periodic_pairs(show_mesh=True)   # optional background mesh

    # solver.solve_steady(runtime=True, itercap=1e4, first_order=True)

    # data = np.load(f'{grid}.gri_roe_results.npz')
    # solver.plot_state_tripcolor(data['U'])

    # solver = FiniteVol(meshname=f'{grid}.gri', fluxname='hlle', gamma=1.4, CFL=0.5)

    # 1) First-order solve
    solver.solve_steady(first_order=True, itercap=100)

    # 2) Second-order solve, initialized from first-order file
    # solver.solve_steady(first_order=False, init_npz=f'{grid}.gri_{flux}_results.npz', limited=True)

    # Plot the converged second-order solution and residual history.
    # solver.plot_residual_history(from_current_solve=True)
    # solver.plot_state_tripcolor()
