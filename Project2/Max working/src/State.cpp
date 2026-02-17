#include "State.hpp"

Vec4 State::flux(const Vec4 &U, const Vec2 &n, double gamma) {
  double rho = U[0];
  double rhou = U[1];
  double rhov = U[2];
  double rhoE = U[3];

  double u = rhou / rho;
  double v = rhov / rho;
  double vdotn = u * n.x + v * n.y;
  double qsq = u * u + v * v;
  double p = (gamma - 1.0) * (rhoE - 0.5 * rho * qsq);
  double H = (rhoE + p) / rho;

  return {rho * vdotn, rhou * vdotn + p * n.x, rhov * vdotn + p * n.y,
          rho * H * vdotn};
}
