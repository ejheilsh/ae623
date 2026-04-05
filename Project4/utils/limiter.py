import numpy as np
import matplotlib.pyplot as plt

"""
1.4 Second Order and Limiters
Implement both ﬁrst-order and second-order ﬁnite-volume methods. For second-order, try both no
limiting and one or more of Barth-Jespersen (BJ) or the maximum-principle (MP) region limiters
(e.g. LCD, Durlofsky, MLG). The choice of gradient calculation for the BJ limiter is up to you.
"""

def compute_lcd_limiter(cell, neighbors):
    u_max = max(cell.mean_u, max(n.mean_u for n in neighbors))
    u_min = min(cell.mean_u, min(n.mean_u for n in neighbors))
    alpha = 1.0 # no limiting start
    for face in cell.faces:
        r = face.center - cell.centroid
        delta_u = np.dot(cell.gradient, r)
        u_face_unlimited = cell.mean_u + delta_u
        if u_face_unlimited > u_max:
            face_alpha = (u_max - cell.mean_u) / delta_u
        elif u_face_unlimited < u_min:
            face_alpha = (u_min - cell.mean_u) / delta_u
        else:
            face_alpha = 1.0
        alpha = min(alpha, face_alpha)
    return max(0.0, alpha)