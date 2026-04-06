#ifndef FLUXES_HPP
#define FLUXES_HPP

#include "State.hpp"
#include "Vector.hpp"

struct FluxResult {
  Vec4 F;
  double smax;
};

FluxResult fluxRoe(const Vec4 &UL, const Vec4 &UR, const Vec2 &n, double gamma);
FluxResult fluxHLLE(const Vec4 &UL, const Vec4 &UR, const Vec2 &n,
                    double gamma);

// ── Flux Jacobians (Project 4: adjoint) ──────────────────────────────
// dF/dUL and dF/dUR returned as row-major 4×4 blocks.
struct FluxJacobianResult {
  double dFdUL[4][4];
  double dFdUR[4][4];
};

FluxJacobianResult fluxRoeJacobian(const Vec4 &UL, const Vec4 &UR,
                                   const Vec2 &n, double gamma);

// Boundary-state Jacobians: dU_boundary / dU_interior (4×4)
void wallStateJacobian   (const Vec4 &Uplus, const Vec2 &n, double gamma,
                          double dUbdUi[4][4]);
void inflowStateJacobian (const Vec4 &Uplus, const Vec2 &n,
                          double rho0, double a0, double alpha, double gamma,
                          double dUbdUi[4][4]);
void outflowStateJacobian(const Vec4 &Uplus, const Vec2 &n,
                          double pb, double gamma,
                          double dUbdUi[4][4]);

Vec4 subsonicInflow(const Vec4 &Uplus, const Vec2 &n, double rho0, double a0,
                    double alpha, double gamma, double y_pos = 0.0, double time = 0.0,
                    bool use_unsteady = true);
Vec4 subsonicOutflow(const Vec4 &Uplus, const Vec2 &n, double pb, double gamma);
Vec4 inviscidWallState(const Vec4 &Uplus, const Vec2 &n, double gamma);
FluxResult inviscidWallFlux(const Vec4 &Uplus, const Vec2 &n, double gamma);

#endif
