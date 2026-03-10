# DG Data Structure - Visual Guide

## Current Finite Volume (p=0) Structure

```
Elements in mesh: E[0], E[1], E[2], ..., E[Ne-1]

State vector U:
┌─────────┬─────────┬─────────┬─────────┬─────────┐
│  U[0]   │  U[1]   │  U[2]   │  U[3]   │   ...   │
│ [ρ,     │ [ρ,     │ [ρ,     │ [ρ,     │         │
│  ρu,    │  ρu,    │  ρu,    │  ρu,    │         │
│  ρv,    │  ρv,    │  ρv,    │  ρv,    │         │
│  ρE]    │  ρE]    │  ρE]    │  ρE]    │         │
└─────────┴─────────┴─────────┴─────────┴─────────┘
  elem 0    elem 1    elem 2    elem 3     ...

ONE value per element (piecewise constant)
```

## DG Structure (p=1, ndof=3)

```
State vector U_dg:
┌─────────────────────────────────┬──────────────────────────────────┬─────
│         Element 0               │         Element 1                │ ...
│  ┌─────┬─────┬─────┐           │  ┌─────┬─────┬─────┐            │
│  │DOF 0│DOF 1│DOF 2│            │  │DOF 0│DOF 1│DOF 2│            │
│  │[ρ,  │[ρ,  │[ρ,  │            │  │[ρ,  │[ρ,  │[ρ,  │            │
│  │ ρu, │ ρu, │ ρu, │            │  │ ρu, │ ρu, │ ρu, │            │
│  │ ρv, │ ρv, │ ρv, │            │  │ ρv, │ ρv, │ ρv, │            │
│  │ ρE] │ ρE] │ ρE] │            │  │ ρE] │ ρE] │ ρE] │            │
│  └─────┴─────┴─────┘           │  └─────┴─────┴─────┘            │
└─────────────────────────────────┴──────────────────────────────────┴─────
    u(x,y) = u₀φ₀ + u₁φ₁ + u₂φ₂       u(x,y) = u₀φ₀ + u₁φ₁ + u₂φ₂

THREE DOFs per element (piecewise linear within element)
```

## DG Structure (p=2, ndof=6)

```
Element 0:
┌──────┬──────┬──────┬──────┬──────┬──────┐
│ DOF0 │ DOF1 │ DOF2 │ DOF3 │ DOF4 │ DOF5 │
│ [ρ,  │ [ρ,  │ [ρ,  │ [ρ,  │ [ρ,  │ [ρ,  │
│  ρu, │  ρu, │  ρu, │  ρu, │  ρu, │  ρu, │
│  ρv, │  ρv, │  ρv, │  ρv, │  ρv, │  ρv, │
│  ρE] │  ρE] │  ρE] │  ρE] │  ρE] │  ρE] │
└──────┴──────┴──────┴──────┴──────┴──────┘

SIX DOFs per element (piecewise quadratic)
```

## HOW SOLUTION IS REPRESENTED INSIDE AN ELEMENT

### Finite Volume (p=0):
```
   ▲
   │     ╔═══════╗
   │     ║       ║  Constant value
   │     ║  u₀   ║  throughout element
u  │     ║       ║
   │     ╚═══════╝
   └──────────────────► x
        Element
```

### DG p=1:
```
   ▲      ╱╲
   │     ╱  ╲
   │    ╱    ╲     Linear variation
u  │   ╱  u   ╲    u(x,y) = u₀φ₀(x,y) + u₁φ₁(x,y) + u₂φ₂(x,y)
   │  ╱        ╲
   │ ╱__________╲
   └──────────────────► x
        Element
```

### DG p=2:
```
   ▲      ╱‾╲
   │     ╱   ╲
   │    ╱     ╲    Quadratic variation
u  │   │   u   │   u(x,y) = Σᵢ₌₀⁵ uᵢφᵢ(x,y)
   │   │       │
   │   ╲_____╱
   └──────────────────► x
        Element
```

## BASIS FUNCTION LAYOUT FOR TRIANGLE

Reference triangle: (0,0), (1,0), (0,1)

### p=1 (3 DOFs):
```
       (0,1)
        o  DOF 2: φ₂ = η
        │╲
        │ ╲
        │  ╲
        │   ╲
        │    ╲
        o─────o
     (0,0)  (1,0)
     DOF 0   DOF 1
     φ₀=1-ξ-η  φ₁=ξ
```

### p=2 (6 DOFs):
```
       (0,1)
        o  DOF 2
        │╲
      o │ o  DOF 3,4,5 (edge midpoints)
        │  ╲
        o───o
     DOF 0  DOF 1
    (corner nodes, then edge nodes)
```

### p=3 (10 DOFs):
```
       (0,1)
        o  DOF 2
        │╲
      o │ o
        │  ╲
      o │   o
        │    ╲
        o──o──o
     DOF 0    DOF 1
    (corners, then edge nodes, then interior)
```

## CODE ACCESS PATTERNS

### FV (p=0):
```cpp
// Access state for element e
Vec4 state = U[e];  // one Vec4

// Loop over all elements
for (int e = 0; e < Ne; ++e) {
    process(U[e]);
}
```

### DG (p>0):
```cpp
// Access all DOFs for element e
std::vector<Vec4> elem_dofs = U_dg[e];  // vector of Vec4s

// Access specific DOF j in element e
Vec4 dof_state = U_dg[e][j];

// Loop over all elements and DOFs
for (int e = 0; e < Ne; ++e) {
    for (int j = 0; j < ndof_per_elem; ++j) {
        process(U_dg[e][j]);
    }
}

// Reconstruct solution at a point (xi, eta) in element e
Vec4 u_at_point = {0,0,0,0};
std::vector<double> phi = evaluateBasis(xi, eta, p_order);
for (int j = 0; j < ndof_per_elem; ++j) {
    u_at_point += U_dg[e][j] * phi[j];  // u = Σ uⱼφⱼ
}
```

## RESIDUAL CALCULATION DIFFERENCE

### FV (p=0) - one residual per element:
```cpp
for (each edge) {
    Vec4 flux = computeFlux(U[eL], U[eR]);
    R[eL] += flux * length;   // R is Vec4[Ne]
    R[eR] -= flux * length;
}

// Time step
U_new[e] = U[e] - dt/area * R[e];
```

### DG (p>0) - ndof residuals per element:
```cpp
for (each edge) {
    for (each quadrature point q) {
        // Reconstruct solution at quad point
        Vec4 uL = Σⱼ U_dg[eL][j] * phi[j](xq);
        Vec4 uR = Σⱼ U_dg[eR][j] * phi[j](xq);
        
        Vec4 flux = computeFlux(uL, uR);
        
        // Distribute to all DOFs via test functions
        for (int j = 0; j < ndof; ++j) {
            R[eL][j] += flux * weight[q] * phi[j](xq) * length;
            R[eR][j] -= flux * weight[q] * phi[j](xq) * length;
        }
    }
}

// Time step (need mass matrix inverse)
for (int i = 0; i < ndof; ++i) {
    U_new[e][i] = U[e][i] - dt/area * Σⱼ M_inv[i][j] * R[e][j];
}
```

## MEMORY REQUIREMENTS

For Ne = 1000 elements:

**p=0 (FV):**
- U: 1000 * 4 doubles = 32 KB

**p=1:**
- U_dg: 1000 * 3 * 4 doubles = 96 KB  (3x larger)

**p=2:**
- U_dg: 1000 * 6 * 4 doubles = 192 KB  (6x larger)

**p=3:**
- U_dg: 1000 * 10 * 4 doubles = 320 KB  (10x larger)

This is why higher order is more expensive!
