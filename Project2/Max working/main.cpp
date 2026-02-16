#include "Solver.hpp"
#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
  std::cout << std::unitbuf; // Enable unbuffered output for immediate feedback
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <meshfile> [order] [CFL] [fluxname]"
              << std::endl;
    return 1;
  }

  std::string meshfile = argv[1];
  bool secondOrder = false;
  double cfl = 1.0;
  std::string fluxname = "roe";
  int itercap = 1e6;

  if (argc >= 3) {
    secondOrder = (std::string(argv[2]) == "2");
  }
  if (argc >= 4) {
    cfl = std::stod(argv[3]);
  }
  if (argc >= 5) {
    fluxname = argv[4];
  }
  if (argc >= 6) {
    itercap = std::stoi(argv[5]);
  }

  try {
    FiniteVolumeSolver solver(meshfile);
    solver.CFL = cfl;
    solver.fluxname = fluxname;

    std::cerr << "Starting solver for " << meshfile
              << " (Order: " << (secondOrder ? "2nd" : "1st")
              << ", CFL: " << cfl << ", Flux: " << fluxname
              << ", IterCap: " << itercap << ")" << std::endl;

    solver.solveSteady(itercap, secondOrder, false);

    // Save results to a simple binary or text format for Python to read
    std::ofstream out("results.bin", std::ios::binary);
    int Ne = solver.U.size();
    out.write((char *)&Ne, sizeof(int));
    for (const auto &u : solver.U) {
      out.write((char *)u.v, sizeof(double) * 4);
    }
    out.close();
    std::cout << "Results saved to results.bin" << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
