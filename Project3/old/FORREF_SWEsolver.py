import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass, field
from readgri import readgri
from flux import fluxRoe, fluxRusanov, fluxWall
import time 
import argparse
import sys 
import os 

@dataclass
class FVM2D():
    grifile: str 
    CFL: float = 1.0
    T: float = 2.0      # final time with global time stepping 
    g: float = 9.81     # acceleration due to gravity 
    contour_plot_settings = {
        # 'cmap': 'jet',
        'cmap': 'Blues',
        'hrange': [0.4, 1.2],
        'urange': [-1.5, 1.5],
        'vrange': [-1.5, 1.5]
    }
    fluxtype: str = 'roe' 
    
    def __post_init__(self):
        Mesh = readgri(self.grifile)
        self.V = Mesh['V']
        self.E = Mesh['E']
        self.IE = Mesh['IE']
        self.BE = Mesh['BE']
        self.Bname = Mesh['Bname']
        
        # centroids and areas indexed by element
        self.centroid = Mesh['centroid']
        self.area = Mesh['area']
        
        # inormal and ilength indexed by IE
        self.inormal = Mesh['inormal']
        self.ilength = Mesh['ilength']
        
        # bnormal and blength indexed by BE
        self.bnormal = Mesh['bnormal']
        self.blength = Mesh['blength']
        
        # iteration quantities 
        self.Ne = self.E.shape[0] # number of elements
        self.Ni = self.IE.shape[0] # number of interior edges/faces 
        self.Nb = self.BE.shape[0] # number of boundary edges/faces 
        
        # set fluxtype 
        if self.fluxtype.lower()=='roe':
            self.flux = fluxRoe
        elif self.fluxtype.lower()=='rusanov':
            self.flux = fluxRusanov
        else:
            raise ValueError(f"Unknown fluxtype '{self.fluxtype}'. Expected 'roe' or 'rusanov'. ")
        
        # initialize state
        self.U0 = self.initstate()        

        # identify boundary edge indices in BE for each pipe
        self.get_pipe_edges()
                
    @staticmethod
    def set_IC(xc, yc):
        """sets components for IC: water height (h), x velocity comp (u), y velocity comp (v) """
        # h0 = 1.0 + 0.3*np.exp(-50*(xc - 1.3)**2 - 50*(yc - 0.9)**2)
        h0 = 0.1 + (1/3)*xc**2 + 0.5*yc**2 
        u0 = 0; v0 = 0
        return h0, h0*u0, h0*v0
    
    def get_pipe_edges(self):
        """Preprocesses pipe edge groups from Bname and BE"""
        self.pipe_edges = {}
        for bidx, bname in enumerate(self.Bname[1:], start=1):
            # filter subset of rows in BE with pipe index
            pipe_edge_inds = np.where(self.BE[:, 3] == bidx)[0] # np.where(condition) returns tuple of index arrays (one per axis) of rows that match the condition
            self.pipe_edges[bname] = pipe_edge_inds
    
    def initstate(self):
        """initialize state as Ncells x Nstates per cell. indexed by element """
        U0 = np.zeros([self.Ne, 3]) 
        xc, yc = self.centroid[:, 0], self.centroid[:, 1]
        U0[:,0], U0[:,1], U0[:,2] = self.set_IC(xc, yc)
        return U0
    
    def solve(self, runtime: str = True, pipeforce: str = True):
        """main time stepping loop """
        # start at t=0, with U=initial condition
        t = 0
        U = self.U0.copy()  # U starts of as U0 from initstate() # NOTE WHY HAVE TO COPY HERE AND BELOW
        # TODO can append to one list?? 
        Uhist = [U.copy()] # store history of state by appending to list
        thist = [t] # store history of times by appending to list 
        Fhist = {p: [] for p in self.pipe_edges.keys()}
        
        if runtime==True:
            start = time.time()
        
        # NOTE adaptive time stepping; in NL problem compute dt as part of flux calculation because thats when max wave speed is determined 
        while t < self.T:
            R, sdl = self.calcResidual(U)
            dt = self.calc_dt(sdl)
            # avoid going past the final time 
            if t + dt > self.T: 
                dt = self.T - t
            # forward Euler update
            U -= dt * (R / self.area[:,None]) # NOTE shapes of np arrays. broadcasting rules: If two dimensions differ and one is 1, NumPy stretches that dimension to match the other. 
            Uhist.append(U.copy()) # NOTE U.copy(). U is reference to single (same) np array
            
            # calculate forces on pipes at each time
            if pipeforce==True:
                pipe_forces = self.calc_pipe_forces(U) # store time history of force on each pipe
                for pipename, force in pipe_forces.items():
                    Fhist[pipename].append(force.copy()) # NOTE ALSO WHY GOTTA COPY HERE   
            t += dt # update current time 
            thist.append(t)
        
        if runtime==True:
            runtime = time.time() - start
            hrs, rem = divmod(runtime, 3600)
            mins, secs = divmod(rem, 60)
            self.runtime = f"{int(hrs):02d}:{int(mins):02d}:{secs:05.2f}"
            print(f"Runtime: {self.runtime}")
        # store final state, time history of state AND convert lists → arrays for consistent saving
        # NOTE alternatively, combine all arrays into a big csv or txt file? 
        self.U = U 
        self.Uhist = np.stack(Uhist)   # shape (Nt, Ne, 3)
        self.thist = np.array(thist)
        self.Fhist = {pipe: np.stack(Flist) for pipe, Flist in Fhist.items()} # flatten Fhist dict into arrays

        # save results Uhist, thist, Fhist as numpy binary (.npz) after solving; replaces the nested list objects with plain NumPy arrays so post-processing is easier.
        gridname = os.path.splitext(os.path.basename(self.grifile))[0]
        datafile = f"{gridname}_{self.fluxtype}_results.npz"
        np.savez(
            datafile,
            fluxtype=self.fluxtype,
            Uhist=self.Uhist,
            thist=self.thist,
            Fhist=self.Fhist,
        )
        print(f"Saved results to {datafile}")

    def calcResidual(self, U):
        """residual calculation by looping over internal edges and boundary edges, calculating fluxes, incrementing residuals on respective elements"""
        R = np.zeros([self.Ne, 3]) # residual size Ne x 3 (indexed by elements); "zero out residual every time step"
        sdl = np.zeros(self.Ne) # size Ne x 1 (one scalar per element); sum of max wave speed (smax) over edges of element times edge length
        
        for i in range(self.Ni):
            eL = self.IE[i, 2]
            eR = self.IE[i, 3]
            ilen = self.ilength[i]
            # call flux function on all interior edges 
            F, smag_i = self.flux(U[eL, :], U[eR, :], self.inormal[i, :], self.g)
            R[eL, :] += F*ilen # sums F * n * dl for all three interior edges on element
            R[eR, :] -= F*ilen 
            
            sdl[eL] += smag_i*ilen
            sdl[eR] += smag_i*ilen
            
        for i in range(self.Nb):
            eL = self.BE[i, 2]
            blen = self.blength[i]
            # call flux function on all interior edges 
            F, smag_i = fluxWall(U[eL, :], self.bnormal[i, :], self.g)
            R[eL, :] += F*blen # sums F * n * dl for all three interior edges on element
            sdl[eL] += smag_i*blen
        
        return R, sdl 
    
    def calc_dt(self, sdl):
        """calculate time step based on sdl"""
        tol = 1e-14
        # NOTE avoid division by zero in degenerate elements
        sdl = np.maximum(sdl, tol) 
        # if sdl.any() < tol:
        #     print(f'sdl: {sdl}. using {tol} in dt calculation instead')
        #     sdl = tol
        dt_i = (2*self.area*self.CFL) / sdl    
        dt = np.min(dt_i)
        return dt
    
    def calc_pipe_forces(self, U):
        """compute total force (Fx, Fy) on each pipe boundary group; TODO returns dictionary?"""
        forces = {}
        # for key, value in dict.items()
        for bname, edge_inds in self.pipe_edges.items(): # # NOTE dont understand how the iterating works over the key, value pairs in items()
            F_total = np.zeros(2) # stores x and y forces 
            # NOTE in get_pipe_edges can we store the eL, UL, bnormal for each pipe so we dont have to iterate here and we can vectorize??
            for i in edge_inds:
                eL = self.BE[i, 2] # get left element  
                UL = U[eL, :] # get element's local state
                F, _ = fluxWall(UL, self.bnormal[i, :], self.g) # compute wall flux; [1] and [2] entries contain pressure contributions
                F_total += F[1:]*self.blength[i] # multiply by edge length and accumulate to total pipe force

            forces[bname] = F_total 
        return forces # return dictionary. key = pipe name, value = force at a time

    
    def plot_pipe_forces(self): # OPTIONAL pass in pipe name to plot
        if not hasattr(self, "Fhist"): # program usage flow control
            raise ValueError("No pipe force history found. Run solve() first or load from file.")

        for pipe_name, forces in self.Fhist.items():
            Fx, Fy = forces[:, 0], forces[:, 1]
            t = np.array(self.thist[:len(Fx)])       
            
            for comp, vals, color in zip(["Fx", "Fy"], [Fx, Fy], ["tab:blue", "tab:red"]):
                fig, ax = plt.subplots()
                ax.plot(t, vals, color=color)
                ax.set_xlim(t[0], t[-1])
                ax.set_xlabel("Time (s)")
                ax.set_ylabel(f"{comp} (N)")
                ax.set_title(f"{pipe_name} – {comp}")
                ax.grid(True, linestyle='--', linewidth=0.5)
                fig.tight_layout()
                fig.savefig(f"{pipe_name}_{comp}.svg", format='svg')
                plt.close(fig)
    
    def get_state_at_time(self, t_query):
        """Return U closest in time to t_query from history."""
        if not hasattr(self, "Uhist") or not hasattr(self, "thist"):
            raise ValueError("Run solve() with history tracking to use this.")
        idx = np.argmin(np.abs(self.thist - t_query))
        return self.Uhist[idx], self.thist[idx]

    def plot_state(self, U, t, fname_prefix=None):
        """Plot state at given time"""
        gridname = os.path.splitext(os.path.basename(self.grifile))[0]
        
        if fname_prefix is None:
            fname_prefix = gridname
            
        # unpack contour plot settings         
        cmap = self.contour_plot_settings['cmap']
        hrange = self.contour_plot_settings['hrange']
        urange = self.contour_plot_settings['urange']
        vrange = self.contour_plot_settings['vrange']
    
        h = U[:, 0]
        u = U[:, 1] / h
        v = U[:, 2] / h
        
        tri_mesh = tri.Triangulation(self.V[:,0], self.V[:, 1], self.E)
        
        vars = [('h', h, hrange), ('u', u, urange), ('v', v, vrange)]
        
        for name, field, vrange in vars:
            fig, ax = plt.subplots()
            
            # plot colored field (without boundary edges)
            tpc = ax.tripcolor(tri_mesh, facecolors=field, edgecolors='none', cmap=cmap, vmin=vrange[0], vmax=vrange[1], shading='flat')
            cbar = fig.colorbar(tpc, ax=ax, orientation='horizontal', pad=0.07, aspect=30, shrink=0.8) # NOTE figure gets inset into axes since plt.tight_layout() doesnt play nice with fig.colorbar
            cbar.set_label(name)
            
            # overlay boundary edges of wall and pipes 
            for e1, e2, *_ in self.BE: 
                x1, y1 = self.V[e1, :]
                x2, y2 = self.V[e2, :]
                ax.plot([x1, x2], [y1, y2], 'k-', linewidth=0.8)
            
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(self.V[:,0].min(), self.V[:,0].max())
            ax.set_ylim(self.V[:,1].min(), self.V[:,1].max())
            ax.set_title(f'{name}, Ne = {self.Ne}, t = {t:.2f}', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(f'{name}_t={t:.2f}_{gridname}.png', dpi=200)
            plt.close(fig)
            
    def animate2D(self, fieldval='h', interval=50, fps=30, downsample=1, save=False):
        """
        Animate the 2D contour plot of a chosen field over the domain.
        Matches style and structure of animate3D().
        """
        tri_mesh = tri.Triangulation(self.V[:, 0], self.V[:, 1], self.E)

        fig, ax = plt.subplots(figsize=(6.5, 5))
        cmap = self.contour_plot_settings['cmap']
        vmin, vmax = self.contour_plot_settings[f'{fieldval}range']

        # --- helper function to extract field values ---
        def get_fieldval(U):
            h = U[:, 0]
            if fieldval == 'h':
                return h
            u, v = U[:, 1]/h, U[:, 2]/h
            return {'u': u, 'v': v}[fieldval]

        # --- initial frame ---
        tpc = ax.tripcolor(
            tri_mesh,
            facecolors=get_fieldval(self.Uhist[0]),
            edgecolors='none',
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shading='flat'
        )

        # overlay domain boundaries
        for e1, e2, *_ in self.BE:
            x1, y1 = self.V[e1, :]
            x2, y2 = self.V[e2, :]
            ax.plot([x1, x2], [y1, y2], 'k-', linewidth=0.8)

        # colorbar
        cbar = fig.colorbar(tpc, ax=ax, orientation='horizontal', pad=0.07, aspect=30, shrink=0.8)
        cbar.set_label(f'{fieldval}')

        # formatting
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(self.V[:, 0].min(), self.V[:, 0].max())
        ax.set_ylim(self.V[:, 1].min(), self.V[:, 1].max())
        ax.set_title(f'Fuel surface at $t$ = {self.thist[0]:.2f}s', fontsize=12)
        plt.tight_layout()

        # --- frame update ---
        def update(frame):
            frame_idx = frame * downsample
            field = get_fieldval(self.Uhist[frame_idx])
            tpc.set_array(field)
            ax.set_title(f'Fuel surface at $t$ = {self.thist[frame_idx]:.2f}s', fontsize=12)
            return [tpc]

        # --- animation setup ---
        frames = range(0, len(self.Uhist)//downsample)
        anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)

        # --- save or display ---
        if save:
            writer = PillowWriter(fps=fps)
            anim.save(f"{fieldval}_2Dcontour.gif", writer=writer, dpi=150)
            print(f"Saved animation to {fieldval}_2Dcontour.gif")
        else:
            plt.show()

    def animate3D(self, fieldval='h', interval=50, fps=30, downsample=1, save=True):
        """3D animation of centroid height field (finite-volume cell averages)."""

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection='3d')

        # centroid coordinates
        x, y = self.centroid[:, 0], self.centroid[:, 1]

        # fixed camera view
        ax.view_init(elev=20, azim=-130, roll=0)

        # initial scatter (larger dots, single color)
        sc = ax.scatter(
            x, y, self.Uhist[0][:, 0],
            c='dodgerblue', s=20, edgecolors='none'
        )

        # compute aspect ratios (z scaled to half of x/y)
        xrange = x.max() - x.min()
        yrange = y.max() - y.min()
        zrange = 0.5 * max(xrange, yrange)

        ax.set_box_aspect((xrange, yrange, zrange))

        # bounds
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        ax.set_zlim(0, 2)

        # labels and title
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel(fieldval)
        # ax.set_title(f"{fieldval} field at t = {self.thist[0]:.2f}")
        ax.set_title(f"Fuel tank at $t$ = {self.thist[0]:.2f}s", pad=20)
    
        # --- update function for animation ---
        def update(frame):
            frame_idx = frame * downsample
            h = self.Uhist[frame_idx][:, 0]
            sc._offsets3d = (x, y, h)
            # ax.set_title(f"{fieldval} field at t = {self.thist[frame_idx]:.2f}")
            ax.set_title(f"Fuel tank at $t$ = {self.thist[frame_idx]:.2f}s", pad=20)

            return sc,

        frames = range(0, len(self.Uhist)//downsample)
        anim = FuncAnimation(fig, update, frames=frames,
                            interval=interval, blit=False)

        if save:
            # fig.tight_layout(pad=0)
            fig.tight_layout(pad=0.2, rect=[0, 0, 1, 0.96])
            writer = PillowWriter(fps=fps)
            anim.save(f"{fieldval}_3Dscatter.gif", writer=writer, dpi=150, savefig_kwargs={'bbox_inches': 'tight', 'pad_inches': 0.1})
            print(f"Saved animation to {fieldval}_3Dscatter.gif")
        else:
            plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D Finite Volume Shallow Water Solver")
    parser.add_argument("mode",
        choices=["solve", "plotstates", "plotforces", "animate2D", "animate3D"],
        help="Run solver, plot static results, or animate 2D/3D results")
    parser.add_argument("meshfile", help="Path to .gri mesh file")

    # physical & runtime parameters
    parser.add_argument("--CFL", type=float, default=1.0)
    parser.add_argument("--T", type=float, default=2.0)
    parser.add_argument("--fluxtype", type=str, default="roe",
                        choices=["roe", "rusanov"],
                        help="Flux type to use: Roe or Rusanov")
    parser.add_argument("--t", type=float, default=None,
                        help="Time to plot (closest available snapshot)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Force overwrite existing results file")

    # animation parameters
    parser.add_argument("--fieldval", type=str, default="h", choices=["h", "u", "v"],
                        help="Field variable to plot/animate")
    parser.add_argument("--interval", type=float, default=50, help="Frame interval (ms)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--downsample", type=int, default=1,
                        help="Frame downsampling factor")
    parser.add_argument("--save", action="store_true",
                        help="Save animation instead of showing interactively")

    args = parser.parse_args()

    # instance setup
    print(f"Initialized solver for mesh {args.meshfile} using {args.fluxtype.upper()} flux.")
    tank = FVM2D(grifile=args.meshfile, CFL=args.CFL, T=args.T, fluxtype=args.fluxtype)
    gridname = os.path.splitext(os.path.basename(args.meshfile))[0]
    datafile = f"{gridname}_{args.fluxtype}_results.npz"

    # === SOLVE ===
    if args.mode == "solve":
        if os.path.exists(datafile) and not args.overwrite:
            print(f"Results file {datafile} already exists. Use --overwrite to replace it.")
            sys.exit(0)
        print("Running 2D Finite Volume Solver...")
        tank.solve(runtime=True)
        print(f"Runtime: {tank.runtime}")

    # === POST-PROCESSING ===
    else:
        if not os.path.exists(datafile):
            print(f"Results for flux '{args.fluxtype}' not found ({datafile}). Run 'solve' first.")
            sys.exit(1)

        data = np.load(datafile, allow_pickle=True) # what does pickling do? Before was pickling nested lists before stacking to arrays...
        tank.thist = data["thist"]
        tank.Uhist = data["Uhist"]
        tank.Fhist = data["Fhist"].item()
        print(f"Loaded results from {datafile}")

        if args.mode == "plotstates":
            if args.t is not None:
                U_sel, t_sel = tank.get_state_at_time(args.t)
                print(f"Plotting nearest snapshot at t = {t_sel:.3f} (requested {args.t:.3f})")
            else:
                U_sel, t_sel = tank.Uhist[-1], tank.thist[-1]
                print(f"No time specified. Plotting final state at t = {t_sel:.3f}")
            tank.plot_state(U_sel, t_sel)

        elif args.mode == "plotforces":
            print("Plotting force time histories on pipes...")
            tank.plot_pipe_forces()

        elif args.mode == "animate2D":
            print(f"Animating {args.fieldval} field in 2D...")
            tank.animate2D(fieldval=args.fieldval,
                           interval=args.interval, fps=args.fps,
                           downsample=args.downsample, save=args.save)

        elif args.mode == "animate3D":
            print(f"Animating {args.fieldval} field in 3D...")
            tank.animate3D(fieldval=args.fieldval,
                           interval=args.interval, fps=args.fps,
                           downsample=args.downsample, save=args.save)