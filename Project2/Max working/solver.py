import numpy as np 
import matplotlib.pyplot as plt 
from dataclasses import dataclass, field 
from readgri import readgri 
from pathlib import Path
from viz import plotmesh
from fluxes import fluxHLLE, fluxRoe, inviscid_wall, subsonic_inflow, subsonic_outflow
import time
from typing import Optional

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
        RnormL1 = 1 # starting guess for RnormL1    
        niter = 0
        
        # TODO one issue i see is R is a matrix [Ne x 4] not a vector...
        # is the L1 norm gonna be right? 
        
        while RnormL1 > self.rtol and niter < itercap:
            # calculate residual
            R, sdl = self.calc_residual(U)
            # RnormL1 = np.linalg.norm(R, ord=1) # monitor L1 norm of the discrete residual vector
            RnormL1 = np.sum(np.abs(R)) # from GPT: this is the global L1 norm? else can have elementwise L1? 
            
            Rhist.append(RnormL1)
            if niter % 10 == 0: 
                print(f'niter: {niter}, RnormL1: {RnormL1:.6e}')
            # update the state via local time stepping 
            dt = self.calc_dt(sdl, usecase='steady') # this ought to be an array with time steps for each element
            # TODO check the broadcasting on this 
            U -= dt[:, None] * (R / self.Area[:, None]) # TODO right now using FE for first order, build out additional time integration SSP-RK-.. later
            niter += 1 # increment counter 
        
        if niter==itercap:
            print("WARNING: Iteration cap reached without convergence")
            
        # store final state, time history of state
        self.U = U 
        self.Rhist = Rhist 
        
        # save results as numpy binary  (.npz) after solving; replaces nested list objects with plain np arrays so that post-processing is easier
        datafile = f"{self.meshname}_{self.fluxname}_results.npz"
        np.savez(
            datafile,
            fluxfunc=self.fluxname,
            U=self.U,
            Rhist=self.Rhist
        )
        print(f"Saved results to {datafile}")
        

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
            
            #TODO come up with a way to return wave speed FOR ALL THREE!!! SEE PIAZZA
            if Bname[bidx]=='inflow': 
                F, smag_b = subsonic_inflow(U[elemL, :], self.Bn[i, :2], self.rho0, self.a0, self.alpha, self.gamma)
            elif Bname[bidx]=='wall': 
                F, smag_b = inviscid_wall(U[elemL, :], self.Bn[i, :2], self.gamma)
            elif Bname[bidx]=='outflow': 
                F, smag_b = subsonic_outflow(U[elemL, :], self.Bn[i, :2], self.pout, self.gamma)
            else:
                raise ValueError(f"Not reading in boundary name correctly...unsupported boundary condition??")
            F = np.asarray(F, dtype=float)
            if F.shape != (4,):
                raise ValueError(f"Boundary flux returned shape {F.shape}, expected (4,)")
            R[elemL, :] += F * blen # sums F dot n * dl for all three interior edges on element
            sdl[elemL] += smag_b * blen

        return R, sdl
    
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
        elif usecase.loewr()=='unsteady':
            return dt
        else: 
            raise ValueError(f"Expected 'steady' or 'unsteady' for usecase") 
        
if __name__=="__main__":
    solver = FiniteVol(meshname='2k.gri', fluxname='roe', gamma=1.4)
    solver.solve_steady(runtime=True, itercap=10e6)
    print('deez')
    # plotmesh(solver.Mesh, fname='testplot', savefig=True)
