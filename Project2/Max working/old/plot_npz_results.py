import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from readgri import readgri


def infer_mesh_from_npz(npz_path: Path) -> Path:
    # Expected pattern from solver: "<meshname>_<flux>_results.npz"
    stem = npz_path.name
    suffix = "_results.npz"
    if not stem.endswith(suffix):
        raise ValueError("Cannot infer mesh file: expected filename ending with '_results.npz'")
    base = stem[: -len(suffix)]  # "<meshname>_<flux>"
    parts = base.rsplit("_", 1)
    if len(parts) != 2:
        raise ValueError("Cannot infer mesh file: expected '<meshname>_<flux>_results.npz'")
    meshname = parts[0]
    return npz_path.parent / meshname


def plot_state_tripcolor(V: np.ndarray, E: np.ndarray, U: np.ndarray, cmap: str = "viridis"):
    labels = [r"$\rho$", r"$\rho u$", r"$\rho v$", r"$\rho E$"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.ravel()

    for i in range(4):
        ax = axes[i]
        tpc = ax.tripcolor(
            V[:, 0],
            V[:, 1],
            E,
            facecolors=U[:, i],
            shading="flat",
            cmap=cmap,
        )
        ax.triplot(V[:, 0], V[:, 1], E, color="k", linewidth=0.12, alpha=0.2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(labels[i])
        fig.colorbar(tpc, ax=ax, shrink=0.9)

    fig.suptitle("State Variables From NPZ")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot state variables from solver .npz results.")
    parser.add_argument("npz", type=Path, help="Path to results .npz (contains U).")
    parser.add_argument("--mesh", type=Path, default=None, help="Path to .gri mesh file (optional).")
    parser.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap.")
    args = parser.parse_args()

    npz_path = args.npz.resolve()
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    data = np.load(npz_path)
    if "U" not in data:
        raise KeyError("NPZ file does not contain 'U'")
    U = np.asarray(data["U"], dtype=float)
    if U.ndim != 2 or U.shape[1] != 4:
        raise ValueError(f"Expected U shape (Ne, 4), got {U.shape}")

    mesh_path = args.mesh.resolve() if args.mesh is not None else infer_mesh_from_npz(npz_path).resolve()
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    mesh = readgri(mesh_path)
    V = mesh["V"]
    E = mesh["E"]
    if U.shape[0] != E.shape[0]:
        raise ValueError(
            f"State/mesh mismatch: U has {U.shape[0]} elements, mesh has {E.shape[0]} elements."
        )

    plot_state_tripcolor(V, E, U, cmap=args.cmap)


if __name__ == "__main__":
    main()

