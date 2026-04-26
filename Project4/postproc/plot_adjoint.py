#!/usr/bin/env python3
"""
Usage: python postproc/plot_adjoint.py <output_dir> [cycle]
"""
import struct, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

sys.path.insert(0, str(Path(__file__).parent))
from dg_utils import read_dg_results


def read_companion_mesh(filename):
    with open(filename, "rb") as f:
        nn = struct.unpack("i", f.read(4))[0]
        nodes = np.frombuffer(f.read(8 * 2 * nn), dtype=np.float64).reshape(nn, 2)
        ne = struct.unpack("i", f.read(4))[0]
        elements = []
        for _ in range(ne):
            _q, v0, v1, v2 = struct.unpack("iiii", f.read(16))
            elements.append([v0, v1, v2])
    return nodes, elements


def read_indicators(filename):
    with open(filename, "rb") as f:
        ne = struct.unpack("i", f.read(4))[0]
        return np.frombuffer(f.read(8 * ne), dtype=np.float64)


output_dir = sys.argv[1]
cycle = int(sys.argv[2]) if len(sys.argv) > 2 else 0
prefix = sys.argv[3] if len(sys.argv) > 3 else None
d = Path(output_dir)

if prefix:
    psi_file  = d / f"{prefix}adjoint_psi_cycle{cycle}_dg.bin"
    ind_file  = d / f"{prefix}adjoint_indicators_cycle{cycle}.bin"
    mesh_file = d / f"{prefix}adjoint_mesh_cycle{cycle}.bin"
else:
    psi_file  = sorted(d.glob(f"*adjoint_psi_cycle{cycle}_dg.bin"))[0]
    ind_file  = sorted(d.glob(f"*adjoint_indicators_cycle{cycle}.bin"))[0]
    mesh_file = sorted(d.glob(f"*adjoint_mesh_cycle{cycle}.bin"))[0]

nodes, elements = read_companion_mesh(str(mesh_file))
psi_dg, p_order, ndof = read_dg_results(str(psi_file))
indicators = read_indicators(str(ind_file))

Ne = len(elements)
verts = [nodes[e] for e in elements]

# p=0: one DOF per element, 4 state components
psi_mag = np.linalg.norm(psi_dg[:, 0, :], axis=1)

#fig, axes = plt.subplots(1, 2, figsize=(16, 6))

plt.figure(figsize=(8, 6))
plt.suptitle(f"Cycle {cycle} | p={p_order} | {Ne} elements")

#ax = axes[0]
#pc = PolyCollection(verts, cmap="plasma", edgecolors="none")
#pc.set_array(psi_mag)
#pc.set_clim(psi_mag.min(), psi_mag.max())
#ax.add_collection(pc)
#ax.set_xlim(nodes[:, 0].min(), nodes[:, 0].max())
#ax.set_ylim(nodes[:, 1].min(), nodes[:, 1].max())
#ax.set_aspect("equal")
#fig.colorbar(pc, ax=ax, label="|psi|")
#ax.set_title("Adjoint Magnitude")

ax = plt.gca()
ind_log = np.log10(np.maximum(indicators, 1e-30))
nonzero = ind_log[indicators > 0]
vmin_ind = nonzero.min() if len(nonzero) else ind_log.min()
pc2 = PolyCollection(verts, cmap="hot", edgecolors="none")
pc2.set_array(ind_log)
pc2.set_clim(vmin_ind, ind_log.max())
ax.add_collection(pc2)
ax.set_xlim(nodes[:, 0].min(), nodes[:, 0].max())
ax.set_ylim(nodes[:, 1].min(), nodes[:, 1].max())
ax.set_aspect("equal")
plt.colorbar(pc2, ax=ax, label="log10(indicator)")
ax.set_title("Error Indicators")

plt.tight_layout()
if prefix:
    outname = d / f"{prefix}adjoint_cycle{cycle}.png"
else:
    outname = d / f"adjoint_cycle{cycle}.png"
plt.savefig(outname, dpi=150, bbox_inches="tight")
print(f"Saved {outname}")
