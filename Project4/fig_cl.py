import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# Data (updated)
# -------------------------

# Uniform p=0
x_u0 = np.array([2000, 8000, 32000, 128000])
y_u0 = np.array([0.694549, 0.724121, 0.740194, 0.751011])

# Adjoint p=0 (UPDATED: 2k, 5% adapt run)
x_a0 = np.array([1989, 2079, 2171, 2263, 2364, 2472, 2579, 2699, 2816, 2946, 3072])
y_a0 = np.array([
    0.694548970,
    0.700022995,
    0.702936375,
    0.704844044,
    0.706772931,
    0.708681276,
    0.709920438,
    0.711026161,
    0.711971213,
    0.713213010,
    0.714309284
])

# Adjoint p=1 (UPDATED: 2k p=1, q=3 run)
x_a1 = np.array([1989, 2398, 3271, 5170, 8949, 16468])
y_a1 = np.array([
    0.755684,
    0.755384,
    0.755347,
    0.755489,
    0.755692,
    0.755721
])

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

plt.plot(x_u0, y_u0, marker='s', label='Uniform p=0, q=1')
plt.plot(x_a0, y_a0, marker='o', label='Adjoint p=0, q=1 (2k base)')
plt.plot(x_a1, y_a1, marker='^', label='Adjoint p=1, q=3 (2k base)')
plt.plot(x_u1, y_u1, marker='^', linestyle='--', label='Uniform p=1, q=1')
plt.plot(x_u2, y_u2, marker='D', label='Uniform p=2, q=1')
#plt.plot(x_ref, y_ref, linestyle='--', label='Reference')

# Axes formatting
plt.xscale('log')
plt.xlabel('Number of Elements')
plt.ylabel(r'$C_l$')

plt.ylim(0.68, 0.76)
plt.grid(True, which='major')

plt.legend()
plt.tight_layout()

# Save high-quality figure
plt.savefig('cl_p1q3_plot.png', dpi=600)

plt.show()