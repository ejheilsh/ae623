#include "Solver.hpp"
#include <fstream>
#include <iostream>

int main(int argc, char **argv) {
  std::cout << std::unitbuf; // Enable unbuffered output for immediate feedback
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " <meshfile> [order] [CFL] [fluxname] [itercap] "
              << "[steady/unsteady] [ic_file] [t_end] "
              << "[--map-ic <coarse_meshfile> <coarse_statefile>]"
              << "\n\nArguments:"
              << "\n  meshfile      : Grid file (e.g., grids/2k.gri)"
              << "\n  order         : DG polynomial order (0, 1, 2, or 3) [default: 0]"
              << "\n  CFL           : CFL number [default: 1.0]"
              << "\n  fluxname      : Flux scheme (roe or hlle) [default: roe]"
              << "\n  itercap       : Maximum iterations [default: 1e6]"
              << "\n  steady/unsteady : Solution mode [default: steady]"
              << "\n  ic_file       : Initial condition file (optional)"
              << "\n  t_end         : End time for unsteady (optional, requires ic_file)"
              << "\n  --map-ic      : Load IC from coarser mesh (optional)"
              << "\n\nNotes:"
              << "\n  - For p>0 steady runs without ic_file: automatically converges p=0 first"
              << "\n  - Output files tagged by order: steady_<mesh>_p<order>_results.bin"
              << std::endl;
    return 1;
  }

  std::string meshfile = argv[1];
  int p_order = 0;  // DG polynomial order (0, 1, 2, 3)
  double cfl = 1.0;
  std::string fluxname = "roe";
  int itercap = 1e6;
  bool unsteady = false;
  std::string ic_file = "";
  double t_end = -1.0;  // negative means run until itercap
  bool use_mapped_ic = false;
  std::string coarse_meshfile = "";
  std::string coarse_statefile = "";

  auto is_flag_token = [](const char *s) {
    return std::string(s).rfind("--", 0) == 0;
  };

  if (argc >= 3 && !is_flag_token(argv[2])) {
    p_order = std::stoi(argv[2]);  // Parse as integer for DG order
    if (p_order < 0 || p_order > 3) {
      std::cerr << "Error: order must be 0, 1, 2, or 3" << std::endl;
      return 1;
    }
  }
  if (argc >= 4 && !is_flag_token(argv[3])) {
    cfl = std::stod(argv[3]);
  }
  if (argc >= 5 && !is_flag_token(argv[4])) {
    fluxname = argv[4];
  }
  if (argc >= 6 && !is_flag_token(argv[5])) {
    itercap = std::stoi(argv[5]);
  }
  if (argc >= 7 && !is_flag_token(argv[6])) {
    unsteady = (std::string(argv[6]) == "unsteady");
  }
  if (argc >= 8 && !is_flag_token(argv[7])) {
    ic_file = argv[7];
  }
  if (argc >= 9 && !is_flag_token(argv[7]) && !is_flag_token(argv[8])) {
    t_end = std::stod(argv[8]);
  }

  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--map-ic") {
      if (i + 2 >= argc) {
        std::cerr << "Error: --map-ic requires <coarse_meshfile> <coarse_statefile>"
                  << std::endl;
        return 1;
      }
      use_mapped_ic = true;
      coarse_meshfile = argv[i + 1];
      coarse_statefile = argv[i + 2];
      i += 2;
    } else if (is_flag_token(argv[i])) {
      std::cerr << "Error: Unknown option " << arg << std::endl;
      return 1;
    }
  }

  try {
    FiniteVolumeSolver solver(meshfile);

    // Extract grid name from mesh file path (needed for auto-chaining filenames)
    std::string grid_name = meshfile;
    size_t last_slash = grid_name.find_last_of("/\\");
    if (last_slash != std::string::npos) {
      grid_name = grid_name.substr(last_slash + 1);
    }
    size_t dot_pos = grid_name.find_last_of(".");
    if (dot_pos != std::string::npos) {
      grid_name = grid_name.substr(0, dot_pos);
    }

    solver.p_order = p_order;  // Set DG polynomial order
    solver.initializeDG();     // Initialize DG structures based on p_order
    solver.CFL = cfl;
    solver.fluxname = fluxname;
    
    if (use_mapped_ic) {
      if (!ic_file.empty()) {
        std::cerr << "Warning: both ic_file and --map-ic provided; using --map-ic"
                  << std::endl;
      }
      solver.loadMappedInitialCondition(coarse_meshfile, coarse_statefile);
    } else if (!ic_file.empty()) {
      // Load same-mesh initial condition if provided
      solver.loadInitialCondition(ic_file);
    } else if (p_order > 0 && !unsteady) {
      // Automatically converge p=0 first, then use as IC for p>0
      std::cerr << "Auto-chaining: converging p=0 first to seed p=" 
                << p_order << " IC..." << std::endl;
      
      // Save current DG order and initialize as p=0
      int saved_order = solver.p_order;
      solver.p_order = 0;
      solver.initializeDG();
      solver.CFL = 1.0;  // p=0 can use larger CFL
      solver.solveSteady(itercap);
      
      // Save p=0 result to temporary file
      system("mkdir -p data_steady");
      std::string p0_file = "data_steady/steady_" + grid_name + "_p0_results.bin";
      {
        std::ofstream p0out(p0_file, std::ios::binary);
        int Ne = solver.U.size();
        p0out.write((char *)&Ne, sizeof(int));
        for (const auto &u : solver.U)
          p0out.write((char *)u.v, sizeof(double) * 4);
        p0out.close();
      }
      std::cerr << "p=0 converged. Saved to " << p0_file << std::endl;
      
      // Now re-initialize at the target order and load p=0 IC
      solver.p_order = saved_order;
      solver.initializeDG();
      solver.CFL = cfl;
      solver.res_history.clear();
      solver.loadInitialCondition(p0_file);
      std::cerr << "Loaded p=0 solution as IC for p=" << saved_order << std::endl;
    }

    std::cerr << "Starting solver for " << meshfile
              << " (DG Order p=" << p_order
              << ", CFL: " << cfl << ", Flux: " << fluxname
              << ", IterCap: " << itercap 
              << ", Mode: " << (unsteady ? "Unsteady" : "Steady")
              << (t_end > 0.0 ? ", t_end: " + std::to_string(t_end) : "")
              << (use_mapped_ic ? ", Mapped IC: " + coarse_meshfile + " + " +
                                      coarse_statefile
                                : "")
              << ")" << std::endl;

    if (unsteady) {
      solver.solveUnsteady(itercap, t_end);
    } else {
      solver.solveSteady(itercap);
    }

    // Determine output directory and filename prefix (include DG order)
    std::string output_dir = unsteady ? "." : "data_steady";
    std::string order_tag = "p" + std::to_string(p_order);
    std::string file_prefix = unsteady ? "" : "steady_" + grid_name + "_" + order_tag + "_";
    
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
