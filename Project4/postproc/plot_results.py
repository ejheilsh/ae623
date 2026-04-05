import sys

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.collections import PolyCollection

from dg_utils import (
    GAMMA,
    build_dg_triangulation,
    element_polygon,
    infer_dg_filename,
    maybe_read_dg_results,
    primitive_from_state,
    read_cell_results,
    read_gri_mesh,
)


def read_residual(filename):
    try:
        with open(filename, "rb") as f:
            nit = np.fromfile(f, dtype=np.int32, count=1)[0]
            return np.fromfile(f, dtype=np.float64, count=nit)
    except FileNotFoundError:
        return None


def read_cell_residual(filename):
    try:
        with open(filename, "rb") as f:
            ne = np.fromfile(f, dtype=np.int32, count=1)[0]
            return np.fromfile(f, dtype=np.float64, count=ne)
    except FileNotFoundError:
        return None


def plot_results(meshfile, resultsfile, residualfile, cellresfile, show_plot=True,
                 output_file="solution_summary.png"):
    mesh = read_gri_mesh(meshfile)
    res_hist = read_residual(residualfile)
    cell_res = read_cell_residual(cellresfile)

    dg_filename, U_dg, p_order, _ = maybe_read_dg_results(resultsfile)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax1 = axes[0, 0]
    ax2 = axes[1, 0]
    ax3 = axes[0, 1]
    ax4 = axes[1, 1]

    if U_dg is not None:
        x, y, tris, mach = build_dg_triangulation(mesh, U_dg, p_order, "mach")
        _, _, _, entropy = build_dg_triangulation(mesh, U_dg, p_order, "entropy")
        triang = mtri.Triangulation(x, y, tris)
        tpc1 = ax1.tripcolor(triang, mach, shading="gouraud", cmap="jet")
        fig.colorbar(tpc1, ax=ax1, label="Mach Number")
        ax1.set_title(f"Mach Number Distribution (DG reconstructed, p={p_order})")
        tpc2 = ax2.tripcolor(triang, entropy, shading="gouraud", cmap="viridis")
        fig.colorbar(tpc2, ax=ax2, label="log(p/rho^gamma)")
        ax2.set_title(f"Entropy Distribution (DG reconstructed, p={p_order})")
        print(f"Using DG coefficients from {dg_filename}")
    else:
        u = read_cell_results(resultsfile)
        prim = primitive_from_state(u)
        mach = prim["mach"]
        entropy = prim["entropy"]
        verts = [element_polygon(elem, mesh["nodes"]) for elem in mesh["elements"]]
        pc1 = PolyCollection(verts, cmap="jet")
        pc1.set_array(mach)
        ax1.add_collection(pc1)
        ax1.autoscale()
        fig.colorbar(pc1, ax=ax1, label="Mach Number")
        ax1.set_title("Mach Number Distribution (cell averages)")
        pc2 = PolyCollection(verts, cmap="viridis")
        pc2.set_array(entropy)
        ax2.add_collection(pc2)
        ax2.autoscale()
        fig.colorbar(pc2, ax=ax2, label="log(p/rho^gamma)")
        ax2.set_title("Entropy Distribution (cell averages)")
        print(f"No DG coefficient file found; using cell averages. Expected {infer_dg_filename(resultsfile)}")

    ax1.set_aspect("equal")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax2.set_aspect("equal")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")

    if cell_res is not None:
        verts = [element_polygon(elem, mesh["nodes"]) for elem in mesh["elements"]]
        pc2 = PolyCollection(verts, cmap="magma")
        pc2.set_array(np.log10(np.maximum(cell_res, 1e-16)))
        ax3.add_collection(pc2)
        ax3.autoscale()
        ax3.set_aspect("equal")
        fig.colorbar(pc2, ax=ax3, label="Log10(Residual Norm)")
        ax3.set_title("Spatial Residual Distribution")
        ax3.set_xlabel("X")
        ax3.set_ylabel("Y")
    else:
        ax3.text(0.5, 0.5, "Cell residuals not found", ha="center", va="center")

    if res_hist is not None:
        ax4.plot(res_hist)
        ax4.set_yscale("log")
        ax4.set_title("Convergence History")
        ax4.set_xlabel("Iteration")
        ax4.set_ylabel("Residual")
        ax4.grid(True, which="both", ls="-", alpha=0.5)
    else:
        ax4.text(0.5, 0.5, "Residual history not found", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Saved {output_file}")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    args = [arg for arg in sys.argv[1:] if arg != "--no-show"]
    show_plot = "--no-show" not in sys.argv[1:]

    if len(args) < 2:
        print("Usage: python plot_results.py <meshfile> <results.bin> [residual.bin] [cell_res.bin] [--no-show]")
    else:
        mesh = args[0]
        results = args[1]
        residual = args[2] if len(args) > 2 else "residual.bin"
        cellres = args[3] if len(args) > 3 else "cell_res.bin"
        plot_results(mesh, results, residual, cellres, show_plot=show_plot)
