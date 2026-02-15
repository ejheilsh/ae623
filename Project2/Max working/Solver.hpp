#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "Fluxes.hpp"
#include "Mesh.hpp"
#include <string>
#include <vector>

class FiniteVolumeSolver {
public:
  Mesh mesh;
  std::vector<Vec4> U;
  std::vector<Vec4> U0;
  std::string fluxname = "roe";

  double gamma = 1.4;
  double CFL = 1.0;
  double rtol = 1e-6;
  double rho0 = 1.0;
  double a0 = 1.0;
  double alpha = 50.0;
  double Minf = 0.1;
  double p0;
  double pout;

  FiniteVolumeSolver(const std::string &meshfile);

  void setInitialCondition();
  void solveSteady(int itercap = 1000000, bool secondOrder = false,
                   bool limited = false);

private:
  struct ResidualResult {
    std::vector<Vec4> R;
    std::vector<double> sdl;
  };

  ResidualResult calcResidual(const std::vector<Vec4> &Un);
  ResidualResult calcResidualSecondOrder(const std::vector<Vec4> &Un,
                                         bool limited);

  std::vector<Vec4> applyLimiter(const std::vector<Vec4> &Un,
                                 std::vector<Vec4> &gradX,
                                 std::vector<Vec4> &gradY);
  std::vector<Vec4> sspRK2(const std::vector<Vec4> &Un, bool secondOrder,
                           bool limited);

  std::vector<double> calcDt(const std::vector<double> &sdl);
  bool isPhysical(const std::vector<Vec4> &Un);
};

#endif
