import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# Data (updated)
# -------------------------

# Uniform p=0
x_u0 = np.array([2000, 8000, 32000, 128000])
y_u0 = np.array([0.694549, 0.724121, 0.740194, 0.751011])

"""
sorry, use this value of cl at p=0, q=1 for the 2k base run instead for the below to correct

\begin{table}[h!]
\centering
\caption{Mesh adaptation progression using adjoint-based indicators, impact of solution order.}
\label{tab:mesh_adaptation}

\resizebox{\columnwidth}{!}{%
\begin{tabular}{ccccc}
\toprule
Cycle & Elem ($p=0$) & $C_l$ ($p=0, q=1$) & $C_l$ ($p=1, q=3$) \\
\midrule
0 & 1989  & 0.694549 & 0.755684 \\
1 & 2407  & 0.703027 & 0.755384 \\
2 & 3313  & 0.710215 & 0.755347 \\
3 & 5175  & 0.718072 & 0.755489 \\
4 & 9077  & 0.725317 & 0.755692 \\
5 & 16587 & 0.731905 & 0.755721 \\
\bottomrule
\end{tabular}%
}
\end{table}

"""


# Adjoint p=0 (UPDATED: 2k, 5% adapt run)
x_a0 = np.array([1989, 2398, 3271, 5170, 8949, 16468])
y_a0 = np.array([
    0.694549,
    0.703027,
    0.710215,
    0.718072,
    0.725317,
    0.731905
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