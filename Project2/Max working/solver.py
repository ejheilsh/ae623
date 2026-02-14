import numpy as np 
import matplotlib.pyplot as plt 
from dataclasses import dataclass, field 
from readgri import readgri 
from pathlib import Path
from viz import plotmesh
from fluxes import fluxHLLE, fluxRoe, inviscid_wall, subsonic_inflow, subsonic_outflow, smagb
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

        self.V = self.Mesh['V']
        self.E = self.Mesh['E']
        self.IE = self.Mesh['IE']
        self.BE = self.Mesh['BE']
        self.Bname = self.Mesh['Bname'] 
        
        # centroids and areas indexed by element
        self.Centroid = self.Mesh['Centroid']
        self.Area = self.Mesh['Area']

        # inormals and bnormals indexed by IE, BE respectively 
        self.In = self.Mesh['In']
        self.Bn = self.Mesh['Bn']
        
        # iteration quantities 
        self.Ne = self.E.shape[0] # number of elements
        self.Ni = self.IE.shape[0] # number of interior edges/faces
        self.Nb = self.BE.shape[0] # number of boundary edges/faces 
    
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
        
    def testplotgri(self):
        plotmesh(Mesh=self.Mesh, fname='testmeshplot', savefig=True)
        
    def set_IC(self):
        """
        Initializes uniform state (Uinf or U0) from inlet conditions and guessed Mach number (Minf) 
        """
        self.alpha *= np.pi/180
        
        rho_u = self.rho0 * self.Minf * self.a0 * np.cos(self.alpha)
        rho_v = self.rho0 * self.Minf * self.a0 * np.sin(self.alpha)
        rho_E = self.rho0 * (self.a0**2/((self.gamma - 1)*self.gamma) + (0.5 * self.Minf**2))
        U0 = np.zeros([self.Ne, 4]) # NOTE in matlab code its stored as a column vector (repmat(U0, Ne, 1))
        U0[:, 0] = self.rho0
        U0[:, 1] = rho_u
        U0[:, 2] = rho_v
        U0[:, 3] = rho_E
        self.U0 = U0 
        
    def solve_steady(self, runtime: str = True, itercap: int = 1e6):
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
        
        # TODO one issue i see is R is a matrix [Ne x 4] not a vector...
        # is the L1 norm gonna be right? 
        
        while niter < itercap:
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
            if niter % 10 == 0: 
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
        

    def calc_residual(self, U):
        """
        Calculates residual using the selected flux function
        Loops over internal edges (IE) and boundary edges (BE), calculates fluxes and increments residuals on respective L and R elements
        Also calculates edge-weighted average of wave speeds times edge length (sdl) for every edge visited
        """
        # get the b name
        R = np.zeros([self.Ne, 4]) # residual size Ne x 4 (indexed by elements). **ZERO out residual every time step
        sdl = np.zeros(self.Ne) # size Ne x 1 (one scalar value per element )
        # interior edge flux contributions
        
        # loop over interior edges (NOTE including periodic? might be some weirdness...)
        for i in range(self.Ni):
            elemL = self.IE[i, 2]
            elemR = self.IE[i, 3]
            ilen = self.In[i, 2]
            
            # call flux function on all interior edges 
            F, smag_i = self.flux(U[elemL, :], U[elemR, :], self.In[i, :2], self.gamma)
            F = np.asarray(F, dtype=float)
            if F.shape != (4,):
                raise ValueError(f"Interior flux returned shape {F.shape}, expected (4,)")
            R[elemL, :] += F * ilen # sums F dot n * dl for all three interior edges on element
            R[elemR, :] -= F * ilen
            
            sdl[elemL] += smag_i * ilen
            sdl[elemR] += smag_i * ilen

        # loop over boundary edges 
        for i in range(self.Nb):
            elemL = self.BE[i, 2]
            bidx = self.BE[i, 3]
            blen = self.Bn[i, 2]
            Bname = self.Bname # get the Bname to index by boundary 
            bn = self.Bn[i, :2] # normal vector at boundary (nx, ny)
            
            # TODO come up with a way to return wave speed FOR ALL THREE!!! SEE PIAZZA
            if Bname[bidx]=='inflow': 
                Ub = subsonic_inflow(U[elemL, :], bn, self.rho0, self.a0, self.alpha, self.gamma)
                F, _ = self.flux(U[elemL, :], Ub, bn, self.gamma) # NOTE if returned a wave speed from flux function, it would come from uL and uL not from boundary state by using velocity vector at boundary.
                smag_b = smagb(Ub, bn, self.gamma) # NOTE set the wavespeed using boundary state-based smax 
                 
            elif Bname[bidx]=='outflow': 
                Ub = subsonic_outflow(U[elemL, :], bn, self.pout, self.gamma)
                F, _ = self.flux(U[elemL, :], Ub, bn, self.gamma)
                smag_b = smagb(Ub, bn, self.gamma)
                
            elif Bname[bidx]=='wall': 
                F, smag_b = inviscid_wall(U[elemL, :], bn, self.gamma)
            else:
                raise ValueError(f"Not reading in boundary name correctly...unsupported boundary condition??")
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
    solver = FiniteVol(meshname='2k.gri', fluxname='hlle', gamma=1.4, CFL=0.1)
    # solver = FiniteVol(meshname='2k.gri', fluxname='roe', gamma=1.4, CFL=0.5)
    solver.solve_steady(runtime=True, itercap=10e6)
    print('deez')
    # plotmesh(solver.Mesh, fname='testplot', savefig=True)
