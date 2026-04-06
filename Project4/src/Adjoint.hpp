#ifndef ADJOINT_HPP
#define ADJOINT_HPP

#include "Solver.hpp"
#include <vector>

class AdjointSolver {
public:
  explicit AdjointSolver(FiniteVolumeSolver &solver);

  // Solve the discrete adjoint equation:
  //   (dR/dU)^T * psi = -(dCl/dU)^T
  // using PETSc (or fallback dense LU).  Stores result in psi_.

    // I am unsure how to implement this function, as I have not used PETSc before.  The general steps would be:
    // 1. Initialize PETSc and create a matrix A for (dR/dU)^T and a vector b for -(dCl/dU)^T.
    // 2. Fill A using the Jacobian blocks from solver_.calcJacobian(), taking care to transpose the blocks and place them in the correct positions.
    // 3. Fill b using the output gradient from solver_.dCl_dU(), negating the values as needed.
    // 4. Create a vector x to hold the solution psi_.
    // 5. Set up a linear solver (e.g., KSP) and solve A * x = b.
    // 6. Extract the solution from x and store it in psi_.
    // But im not sure tbh
 
  void solve();

  // Validate adjoint via sensitivity:
  //   dCl/dalpha|_adjoint = -psi^T * dR/dalpha
  // Returns the adjoint-based sensitivity value.
  // Compare against FD:  (Cl(alpha+da) - Cl(alpha)) / da
  double sensitivityAlpha(double dalpha = 1e-6);
  
  // Compute the adjoint-weighted residual error indicator per element.
  // Requires a fine-space solver (p=2) for evaluating R_h(U_H^h).
  std::vector<double> errorIndicators(FiniteVolumeSolver &fine_solver);
  
  // Access the adjoint solution
  const std::vector<std::vector<Vec4>> &psi() const { return psi_; }


private:
  FiniteVolumeSolver &solver_;
  std::vector<std::vector<Vec4>> psi_;  // [nelem][ndof_per_elem]
};

#endif
