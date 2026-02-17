#ifndef STATE_HPP
#define STATE_HPP

#include "Vector.hpp"

class State {
public:
  Vec4 U; // Conservative variables: rho, rho*u, rho*v, rho*E
  double gamma;

  State() : U{0, 0, 0, 0}, gamma(1.4) {}
  State(const Vec4 &u, double g) : U(u), gamma(g) {}
  State(double rho, double rhou, double rhov, double rhoE, double g)
      : U{rho, rhou, rhov, rhoE}, gamma(g) {}

  double rho() const { return U[0]; }
  double rhou() const { return U[1]; }
  double rhov() const { return U[2]; }
  double rhoE() const { return U[3]; }

  double u() const { return U[1] / U[0]; }
  double v() const { return U[2] / U[0]; }
  Vec2 vvec() const { return {u(), v()}; }
  double qSq() const { return u() * u() + v() * v(); }
  double q() const { return std::sqrt(qSq()); }
  double E() const { return U[3] / U[0]; }
  double p() const { return (gamma - 1.0) * (U[3] - 0.5 * U[0] * qSq()); }
  double H() const { return E() + p() / rho(); }
  double c() const { return std::sqrt(gamma * p() / rho()); }
  double M() const { return q() / c(); }

  static Vec4 flux(const Vec4 &U, const Vec2 &n, double gamma);
};

#endif
