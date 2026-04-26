import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# Data (from your LaTeX)
# -------------------------

# Uniform p=0
x_u0 = np.array([2000, 8000, 32000, 128000])
y_u0 = np.array([0.694549, 0.724121, 0.740194, 0.751011])

# Adjoint p=0
x_a0 = np.array([1989, 2769, 3853, 5467, 7843, 11230])
y_a0 = np.array([0.694701, 0.708702, 0.717660, 0.724334, 0.728945, 0.732837])

# Adjoint p=1
x_a1 = np.array([1989, 2734])
y_a1 = np.array([0.755684, 0.755127])

# Uniform p=1
x_u1 = np.array([2000, 8000])
y_u1 = np.array([0.754996, 0.755991])

# Uniform p=2
x_u2 = np.array([2000, 8000])
y_u2 = np.array([0.757033, 0.756374])

# Reference line
x_ref = np.array([1500, 150000])
y_ref = np.array([0.756374, 0.756374])

# -------------------------
# Plot
# -------------------------

plt.figure(figsize=(6.5, 4.5))

plt.plot(x_u0, y_u0, marker='s', label='Uniform p=0')
plt.plot(x_a0, y_a0, marker='o', label='Adjoint p=0')
plt.plot(x_a1, y_a1, marker='^', label='Adjoint p=1')
plt.plot(x_u1, y_u1, marker='^', linestyle='--', label='Uniform p=1')
plt.plot(x_u2, y_u2, marker='D', label='Uniform p=2')
plt.plot(x_ref, y_ref, linestyle='--', label='Reference')

# Axes formatting
plt.xscale('log')
plt.xlabel('Number of Elements')
plt.ylabel(r'$C_L$')

plt.ylim(0.68, 0.76)
plt.grid(True, which='major')

plt.legend()
plt.tight_layout()

# Save high-quality figure
plt.savefig('cl_convergence.png', dpi=600)

plt.show()