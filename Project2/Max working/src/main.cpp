#include "Solver.hpp"
#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
  std::cout << std::unitbuf; // Enable unbuffered output for immediate feedback
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <meshfile> [order] [CFL] [fluxname] [itercap] [steady/unsteady] [ic_file]"
              << std::endl;
    return 1;
  }

  std::string meshfile = argv[1];
  bool secondOrder = false;
  double cfl = 1.0;
  std::string fluxname = "roe";
  int itercap = 1e6;
  bool unsteady = false;
  std::string ic_file = "";

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
  if (argc >= 7) {
    unsteady = (std::string(argv[6]) == "unsteady");
  }
  if (argc >= 8) {
    ic_file = argv[7];
  }

  try {
    FiniteVolumeSolver solver(meshfile);
    solver.CFL = cfl;
    solver.fluxname = fluxname;
    
    // Load initial condition if provided
    if (!ic_file.empty()) {
      solver.loadInitialCondition(ic_file);
    }

    std::cerr << "Starting solver for " << meshfile
              << " (Order: " << (secondOrder ? "2nd" : "1st")
              << ", CFL: " << cfl << ", Flux: " << fluxname
              << ", IterCap: " << itercap 
              << ", Mode: " << (unsteady ? "Unsteady" : "Steady") << ")" << std::endl;

    if (unsteady) {
      solver.solveUnsteady(itercap, secondOrder, false);
    } else {
      solver.solveSteady(itercap, secondOrder, false);
    }

    // Extract grid name from mesh file path
    std::string grid_name = meshfile;
    size_t last_slash = grid_name.find_last_of("/\\");
    if (last_slash != std::string::npos) {
      grid_name = grid_name.substr(last_slash + 1);
    }
    size_t dot_pos = grid_name.find_last_of(".");
    if (dot_pos != std::string::npos) {
      grid_name = grid_name.substr(0, dot_pos);
    }

    // Determine output directory and filename prefix
    std::string output_dir = unsteady ? "." : "data_steady";
    std::string file_prefix = unsteady ? "" : "steady_" + grid_name + "_";
    
    // Create data_steady directory for steady results
    if (!unsteady) {
      system("mkdir -p data_steady");
    }

    // Save results to a simple binary or text format for Python to read
    std::string results_file = output_dir + "/" + file_prefix + "results.bin";
    std::ofstream out(results_file, std::ios::binary);
    int Ne = solver.U.size();
    out.write((char *)&Ne, sizeof(int));
    for (const auto &u : solver.U) {
      out.write((char *)u.v, sizeof(double) * 4);
    }
    out.close();
    std::cout << "Results saved to " << results_file << std::endl;

    std::string residual_file = output_dir + "/" + file_prefix + "residual.bin";
    std::ofstream res_out(residual_file, std::ios::binary);
    int Nit = solver.res_history.size();
    res_out.write((char *)&Nit, sizeof(int));
    res_out.write((char *)solver.res_history.data(), sizeof(double) * Nit);
    res_out.close();
    std::cout << "Residual history saved to " << residual_file << std::endl;

    std::string cell_res_file = output_dir + "/" + file_prefix + "cell_res.bin";
    std::ofstream cell_res_out(cell_res_file, std::ios::binary);
    int Ne_res = solver.cell_residuals.size();
    cell_res_out.write((char *)&Ne_res, sizeof(int));
    cell_res_out.write((char *)solver.cell_residuals.data(),
                       sizeof(double) * Ne_res);
    cell_res_out.close();
    std::cout << "Spatial residuals saved to " << cell_res_file << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
