#include "Fluxes.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

FluxResult fluxRoe(const Vec4 &UL, const Vec4 &UR, const Vec2 &n,
                   double gamma) {
  State sL(UL, gamma), sR(UR, gamma);
  double rhoL = sL.rho(), rhoR = sR.rho();
  double sqrt_rhoL = std::sqrt(rhoL), sqrt_rhoR = std::sqrt(rhoR);

  Vec2 vL = sL.vvec(), vR = sR.vvec();
  double HL = sL.H(), HR = sR.H();

  Vec2 v_roe = (vL * sqrt_rhoL + vR * sqrt_rhoR) / (sqrt_rhoL + sqrt_rhoR);
  double H_roe = (HL * sqrt_rhoL + HR * sqrt_rhoR) / (sqrt_rhoL + sqrt_rhoR);

  double u_roe = v_roe.dot(n);
  double qsq_roe = v_roe.normSq();
  double c_roe = std::sqrt((gamma - 1.0) * (H_roe - 0.5 * qsq_roe));

  double eigs[4] = {u_roe + c_roe, u_roe - c_roe, u_roe, u_roe};
  double mag_eigs[4];
  double eps = 0.1 * c_roe; // Matches Python and PDF spec
  for (int i = 0; i < 4; ++i) {
    mag_eigs[i] = std::abs(eigs[i]);
    if (mag_eigs[i] < eps) {
      mag_eigs[i] = (eps * eps + eigs[i] * eigs[i]) / (2.0 * eps);
    }
  }

  double smax = std::max({mag_eigs[0], mag_eigs[1], mag_eigs[2]});

  double s1 = 0.5 * (mag_eigs[0] + mag_eigs[1]);
  double s2 = 0.5 * (mag_eigs[0] - mag_eigs[1]);

  Vec4 dU = UR - UL;
  Vec2 drhovvec = {dU[1], dU[2]};
  double G1 =
      (gamma - 1.0) * (0.5 * qsq_roe * dU[0] - v_roe.dot(drhovvec) + dU[3]);
  double G2 = -u_roe * dU[0] + drhovvec.dot(n);

  double C1 = (G1 / (c_roe * c_roe)) * (s1 - mag_eigs[2]) + (G2 / c_roe) * s2;
  double C2 = (G1 / c_roe) * s2 + (s1 - mag_eigs[2]) * G2;

  Vec4 Fminus;
  Fminus[0] = mag_eigs[2] * dU[0] + C1;
  Fminus[1] = mag_eigs[2] * dU[1] + C1 * v_roe.x + C2 * n.x;
  Fminus[2] = mag_eigs[2] * dU[2] + C1 * v_roe.y + C2 * n.y;
  Fminus[3] = mag_eigs[2] * dU[3] + C1 * H_roe + C2 * u_roe;

  Vec4 FL = State::flux(UL, n, gamma);
  Vec4 FR = State::flux(UR, n, gamma);
  Vec4 F = (FL + FR) * 0.5 - Fminus * 0.5;

  return {F, smax};
}

FluxResult fluxHLLE(const Vec4 &UL, const Vec4 &UR, const Vec2 &n,
                    double gamma) {
  State sL(UL, gamma), sR(UR, gamma);
  double uL = sL.vvec().dot(n), uR = sR.vvec().dot(n);
  double cL = sL.c(), cR = sR.c();

  double sLmin = std::min(0.0, uL - cL);
  double sRmin = std::min(0.0, uR - cR);
  double sLmax = std::max(0.0, uL + cL);
  double sRmax = std::max(0.0, uR + cR);

  double smin = std::min(sLmin, sRmin);
  double smax = std::max(sLmax, sRmax);
  double den = smax - smin;

  Vec4 FL = State::flux(UL, n, gamma);
  Vec4 FR = State::flux(UR, n, gamma);
  Vec4 F;

  if (std::abs(den) < 1e-14) {
    F = (FL + FR) * 0.5;
  } else {
    F = (FL + FR) * 0.5 - ((smax + smin) / (2.0 * den)) * (FR - FL) +
        ((smax * smin) / den) * (UR - UL);
  }
  double smax_tm = std::max(std::abs(uL) + cL, std::abs(uR) + cR);
  return {F, smax_tm};
}

Vec4 subsonicInflow(const Vec4 &Uplus, const Vec2 &n, double rho0, double a0,
                    double alpha, double gamma, double y_pos, double time,
                    bool use_unsteady) {
  double rho_t = rho0;  // Default to steady state density
  
  // Apply unsteady wake correction if requested
  if (use_unsteady) {
    // Unsteady wake parameters
    const double f_wake = 0.1;      // Wake depth fraction
    const double delta = 0.1;       // Wake width parameter
    const double delta_y = 18;   // 18mm in mm
    
    // Rotor speed equals stagnation speed of sound at inflow
    double V_rotor = a0;
    
    // Calculate eta(t) for unsteady wake
    double y_stator = y_pos + V_rotor * time;
    double eta = y_stator / delta_y - std::floor(y_stator / delta_y) - 0.5;
    
    // Modify stagnation density with unsteady wake function
    rho_t = rho0 * (1.0 - f_wake * std::exp(-0.5 * (eta / delta) * (eta / delta)));
  }
  
  // Stagnation pressure (modified by density change if unsteady)
  double pt = (a0 * a0 * rho_t) / gamma;
  
  State splus(Uplus, gamma);
  double unplus = splus.vvec().dot(n);
  double cplus = splus.c();
  double Jplus = unplus + (2.0 * cplus) / (gamma - 1.0);

  Vec2 nin = {std::cos(alpha), std::sin(alpha)};
  double dn = nin.dot(n);

  double A =
      ((gamma * pt * dn * dn) / rho_t) - 0.5 * (gamma - 1.0) * Jplus * Jplus;
  double B = (4.0 * gamma * pt * dn) / (rho_t * (gamma - 1.0));
  double C = (4.0 * gamma * pt) / (rho_t * (gamma - 1.0) * (gamma - 1.0)) -
             Jplus * Jplus;

  double discr = B * B - 4.0 * A * C;
  double Mb = 0.1;
  if (std::abs(A) < 1e-12) {
    if (std::abs(B) >= 1e-12)
      Mb = -C / B;
  } else {
    discr = std::max(0.0, discr);
    double sqrt_discr = std::sqrt(discr);
    double r1 = (-B + sqrt_discr) / (2.0 * A);
    double r2 = (-B - sqrt_discr) / (2.0 * A);
    if (r1 > 0 && r2 > 0)
      Mb = std::min(r1, r2);
    else if (r1 > 0)
      Mb = r1;
    else if (r2 > 0)
      Mb = r2;
  }

  double Tr = 1.0 / (1.0 + 0.5 * (gamma - 1.0) * Mb * Mb);
  double pb = pt * std::pow(Tr, gamma / (gamma - 1.0));
  double rhob = pb / (Tr * pt / rho_t);
  double cb = std::sqrt(gamma * pb / rhob);
  Vec2 vvecb = nin * (Mb * cb);
  double rhoEb = pb / (gamma - 1.0) + 0.5 * rhob * vvecb.normSq();

  return {rhob, rhob * vvecb.x, rhob * vvecb.y, rhoEb};
}

Vec4 subsonicOutflow(const Vec4 &Uplus, const Vec2 &n, double pb,
                     double gamma) {
  State splus(Uplus, gamma);
  double Splus = splus.p() / std::pow(splus.rho(), gamma);
  double rhob = std::pow(pb / Splus, 1.0 / gamma);
  double unplus = splus.vvec().dot(n);
  double cplus = splus.c();
  double Jplus = unplus + (2.0 * cplus) / (gamma - 1.0);
  double cb = std::sqrt(gamma * pb / rhob);
  double unb = Jplus - (2.0 * cb) / (gamma - 1.0);
  Vec2 vvecb = splus.vvec() - (unplus * n) + (unb * n);
  double rhoEb = pb / (gamma - 1.0) + 0.5 * rhob * vvecb.normSq();

  return {rhob, rhob * vvecb.x, rhob * vvecb.y, rhoEb};
}

Vec4 inviscidWallState(const Vec4 &Uplus, const Vec2 &n, double gamma) {
  State splus(Uplus, gamma);
  Vec2 vvecint = splus.vvec();
  Vec2 vvecb = vvecint - (vvecint.dot(n)) * n;
  return {splus.rho(), splus.rho() * vvecb.x, splus.rho() * vvecb.y,
          splus.rhoE()};
}

FluxResult inviscidWallFlux(const Vec4 &Uplus, const Vec2 &n, double gamma) {
  State splus(Uplus, gamma);
  Vec2 vvecint = splus.vvec();
  Vec2 vvecb = vvecint - (vvecint.dot(n)) * n;
  double pb =
      (gamma - 1.0) * (splus.rhoE() - 0.5 * splus.rho() * vvecb.normSq());
  Vec4 Fb = {0, pb * n.x, pb * n.y, 0};
  double cb = std::sqrt(gamma * splus.p() / splus.rho());
  double smax = std::abs(vvecint.dot(n)) + cb;
  return {Fb, smax};
}

// =====================================================================
// PROJECT 4 — Flux Jacobians for adjoint solver
// =====================================================================

FluxJacobianResult fluxRoeJacobian(const Vec4 &UL, const Vec4 &UR,
                                   const Vec2 &n, double gamma) {
  FluxJacobianResult J{};

  // FD approach: perturb each component of UL and UR, call fluxRoe, difference.
  // This is O(8 * cost_of_fluxRoe) — trivial cost.
  const double eps = 1e-7;
  FluxResult F0 = fluxRoe(UL, UR, n, gamma);

  // dF/dUL
  for (int j = 0; j < 4; ++j) {
    Vec4 ULp = UL;
    ULp[j] += eps;
    FluxResult Fp = fluxRoe(ULp, UR, n, gamma);
    for (int i = 0; i < 4; ++i)
      J.dFdUL[i][j] = (Fp.F[i] - F0.F[i]) / eps;
  }

  // dF/dUR
  for (int j = 0; j < 4; ++j) {
    Vec4 URp = UR;
    URp[j] += eps;
    FluxResult Fp = fluxRoe(UL, URp, n, gamma);
    for (int i = 0; i < 4; ++i)
      J.dFdUR[i][j] = (Fp.F[i] - F0.F[i]) / eps;
  }

  return J;
}

void wallStateJacobian(const Vec4 &Uplus, const Vec2 &n, double gamma,
                       double dUbdUi[4][4]) {
  // Wall state: Ub = [rho, rho*vb_x, rho*vb_y, rhoE]
  // where vb = v - (v.n)*n  (tangential velocity only)
  //
  // In conservative variables:
  //   Ub[0] = Ui[0]                               (rho)
  //   Ub[1] = Ui[1] - (Ui[1]*nx + Ui[2]*ny)*nx    (rho*vb_x = rhou - rho*un*nx)
  //   Ub[2] = Ui[2] - (Ui[1]*nx + Ui[2]*ny)*ny    (rho*vb_y = rhov - rho*un*ny)
  //   Ub[3] = Ui[3]                               (rhoE)
  //
  // Note: rho*un = rhou*nx + rhov*ny, which is linear in conservative vars.
  // So Ub is LINEAR in Ui, and the Jacobian is constant (no state dependence).

  double nx = n.x, ny = n.y;

  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      dUbdUi[i][j] = 0.0;

  // dUb[0]/dUi = [1, 0, 0, 0]
  dUbdUi[0][0] = 1.0;

  // dUb[1]/dUi[1] = 1 - nx*nx,  dUb[1]/dUi[2] = -nx*ny
  dUbdUi[1][1] = 1.0 - nx * nx;
  dUbdUi[1][2] = -nx * ny;

  // dUb[2]/dUi[1] = -ny*nx,  dUb[2]/dUi[2] = 1 - ny*ny
  dUbdUi[2][1] = -ny * nx;
  dUbdUi[2][2] = 1.0 - ny * ny;

  // dUb[3]/dUi = [0, 0, 0, 1]
  dUbdUi[3][3] = 1.0;
}

void inflowStateJacobian(const Vec4 &Uplus, const Vec2 &n,
                          double rho0, double a0, double alpha, double gamma,
                          double dUbdUi[4][4]) {
  // FD shortcut: perturb each conservative variable, call subsonicInflow, difference.
  const double eps = 1e-7;
  Vec4 Ub0 = subsonicInflow(Uplus, n, rho0, a0, alpha, gamma, 0.0, 0.0, false);

  for (int j = 0; j < 4; ++j) {
    Vec4 Up = Uplus;
    Up[j] += eps;
    Vec4 Ubp = subsonicInflow(Up, n, rho0, a0, alpha, gamma, 0.0, 0.0, false);
    for (int i = 0; i < 4; ++i)
      dUbdUi[i][j] = (Ubp[i] - Ub0[i]) / eps;
  }
}

void outflowStateJacobian(const Vec4 &Uplus, const Vec2 &n,
                           double pb, double gamma,
                           double dUbdUi[4][4]) {
  // FD shortcut: perturb each conservative variable, call subsonicOutflow, difference.
  const double eps = 1e-7;
  Vec4 Ub0 = subsonicOutflow(Uplus, n, pb, gamma);

  for (int j = 0; j < 4; ++j) {
    Vec4 Up = Uplus;
    Up[j] += eps;
    Vec4 Ubp = subsonicOutflow(Up, n, pb, gamma);
    for (int i = 0; i < 4; ++i)
      dUbdUi[i][j] = (Ubp[i] - Ub0[i]) / eps;
  }
}
