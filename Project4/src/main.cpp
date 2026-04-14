#include "Solver.hpp"
#include "Adjoint.hpp"
#include "MeshRefinement.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
  std::cout << std::unitbuf; // Enable unbuffered output for immediate feedback
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " <meshfile> [order] [CFL] [fluxname] [itercap] "
              << "[steady/unsteady/freestream] [ic_file] [t_end] "
              << "[--p0-itercap <iters>] "
              << "[--save-after <time>] "
              << "[--save-interval-time <dt_save>] "
              << "[--checkpoint-interval-time <dt_chk>] "
              << "[--map-ic <coarse_meshfile> <coarse_statefile>]"
              << "\n\nArguments:"
              << "\n  meshfile      : Grid file (e.g., grids/2k.gri)"
              << "\n  order         : DG polynomial order (0, 1, 2, or 3) [default: 0]"
              << "\n  CFL           : CFL number [default: 1.0]"
              << "\n  fluxname      : Flux scheme (roe or hlle) [default: roe]"
              << "\n  itercap       : Maximum iterations [default: 1e6]"
              << "\n  steady/unsteady/freestream : Solution mode [default: steady]"
              << "\n  ic_file       : Initial condition file (optional)"
              << "\n  t_end         : End time for unsteady (optional, requires ic_file)"
              << "\n  --p0-itercap  : Separate iteration cap for auto-chained p=0 seed solve"
              << "\n  --save-after  : For unsteady runs, only save snapshots once time >= this value"
              << "\n  --save-interval-time : For unsteady runs, save checkpoints every dt in physical time"
              << "\n  --checkpoint-interval-time : For unsteady runs, overwrite one rolling checkpoint every dt in physical time"
              << "\n  --map-ic      : Load IC from coarser mesh (optional)"
              << "\n\nNotes:"
              << "\n  - For p>0 steady runs without ic_file: automatically converges p=0 first"
              << "\n  - Output files tagged by order: steady_<mesh>_p<order>_results.bin"
              << "\n  - Full DG coefficients are also written to *_results_dg.bin"
              << "\n  - freestream mode bypasses physical BCs for preservation testing"
              << std::endl;
    return 1;
  }

  std::string meshfile = argv[1];
  int p_order = 0;  // DG polynomial order (0, 1, 2, 3)
  double cfl = 1.0;
  std::string fluxname = "roe";
  int itercap = 1e6;
  bool unsteady = false;
  bool freestream_mode = false;
  std::string ic_file = "";
  double t_end = -1.0;  // negative means run until itercap
  bool use_mapped_ic = false;
  int p0_itercap = -1;
  double save_after = 0.0;
  double save_interval_time = -1.0;
  double checkpoint_interval_time = -1.0;
  std::string coarse_meshfile = "";
  std::string coarse_statefile = "";
  bool adjoint_adapt = false;
  double adjoint_tol = 1e-4;
  int adapt_max_cycles = 10;
  double adapt_fraction = 0.25;

  auto is_flag_token = [](const char *s) {
    return std::string(s).rfind("--", 0) == 0;
  };

  auto infer_unsteady_restart_time = [](const std::string &filename) -> double {
    std::string basename = std::filesystem::path(filename).filename().string();
    const std::string prefix = "results_";
    if (basename.rfind(prefix, 0) != 0) {
      return -1.0;
    }
    std::size_t start = prefix.size();
    std::size_t end = basename.find('_', start);
    if (end == std::string::npos) {
      return -1.0;
    }
    std::string time_token = basename.substr(start, end - start);
    char *parse_end = nullptr;
    double t = std::strtod(time_token.c_str(), &parse_end);
    if (parse_end == time_token.c_str() || *parse_end != '\0') {
      return -1.0;
    }
    return t;
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
    std::string mode = argv[6];
    unsteady = (mode == "unsteady");
    freestream_mode = (mode == "freestream");
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
    } else if (arg == "--p0-itercap") {
      if (i + 1 >= argc) {
        std::cerr << "Error: --p0-itercap requires <iters>" << std::endl;
        return 1;
      }
      p0_itercap = std::stoi(argv[i + 1]);
      i += 1;
    } else if (arg == "--save-after") {
      if (i + 1 >= argc) {
        std::cerr << "Error: --save-after requires <time>" << std::endl;
        return 1;
      }
      save_after = std::stod(argv[i + 1]);
      i += 1;
    } else if (arg == "--save-interval-time") {
      if (i + 1 >= argc) {
        std::cerr << "Error: --save-interval-time requires <dt_save>" << std::endl;
        return 1;
      }
      save_interval_time = std::stod(argv[i + 1]);
      i += 1;
    } else if (arg == "--checkpoint-interval-time") {
      if (i + 1 >= argc) {
        std::cerr << "Error: --checkpoint-interval-time requires <dt_chk>" << std::endl;
        return 1;
      }
      checkpoint_interval_time = std::stod(argv[i + 1]);
      i += 1;
    } else if (arg == "--adjoint-adapt") {
      adjoint_adapt = true;
      // Optional: --adjoint-adapt <tol> <max_cycles> <fraction>
      if (i + 1 < argc && !is_flag_token(argv[i + 1])) {
        adjoint_tol = std::stod(argv[i + 1]); i += 1;
      }
      if (i + 1 < argc && !is_flag_token(argv[i + 1])) {
        adapt_max_cycles = std::stoi(argv[i + 1]); i += 1;
      }
      if (i + 1 < argc && !is_flag_token(argv[i + 1])) {
        adapt_fraction = std::stod(argv[i + 1]); i += 1;
      }
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
    solver.freestream_test_mode = freestream_mode;

    std::string order_tag = "p" + std::to_string(p_order);
    std::string output_dir =
        unsteady ? ("unsteady_data/" + grid_name + "_" + order_tag) : "data_steady";
    std::string file_prefix = unsteady ? "" : "steady_" + grid_name + "_" + order_tag + "_";
    std::filesystem::create_directories(output_dir);
    if (unsteady) {
      solver.unsteady_output_dir = output_dir;
      solver.unsteady_save_after = save_after;
      solver.unsteady_save_interval_time = save_interval_time;
      solver.unsteady_checkpoint_interval_time = checkpoint_interval_time;
      if (!ic_file.empty()) {
        solver.unsteady_restart_time_override = infer_unsteady_restart_time(ic_file);
      }
    } else {
      solver.steady_output_dir = output_dir;
      solver.steady_output_prefix = file_prefix;
    }
    
    if (use_mapped_ic) {
      if (!ic_file.empty()) {
        std::cerr << "Warning: both ic_file and --map-ic provided; using --map-ic"
                  << std::endl;
      }
      solver.loadMappedInitialCondition(coarse_meshfile, coarse_statefile);
    } else if (!ic_file.empty()) {
      // Load same-mesh initial condition if provided. Full DG restart files
      // use the *_dg.bin suffix and restore every DG coefficient.
      if (ic_file.size() >= 7 &&
          ic_file.substr(ic_file.size() - 7) == "_dg.bin") {
        solver.loadDGInitialCondition(ic_file);
      } else {
        solver.loadInitialCondition(ic_file);
      }
    } else if (p_order > 0 && !unsteady && !freestream_mode) {
      // Automatically converge p=0 first, then use as IC for p>0
      int p0_seed_itercap = (p0_itercap > 0) ? p0_itercap : itercap;
      std::cerr << "Auto-chaining: converging p=0 first to seed p=" 
                << p_order << " IC"
                << " (p0 itercap=" << p0_seed_itercap << ")..."
                << std::endl;
      
      // Save current DG order and initialize as p=0
      int saved_order = solver.p_order;
      solver.p_order = 0;
      solver.initializeDG();
      solver.CFL = 1.0;  // p=0 can use larger CFL
      solver.solveSteady(p0_seed_itercap);

      if (!solver.last_steady_converged) {
        std::cerr << "Error: auto-chained p=0 seed solve did not converge";
        if (solver.last_steady_failed_nonphysical) {
          std::cerr << " (non-physical state encountered)";
        } else if (solver.last_steady_hit_itercap) {
          std::cerr << " (hit iteration cap before convergence)";
        }
        std::cerr << ". Aborting auto-chain." << std::endl;
        return 1;
      }
      
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
              << ", Mode: "
              << (unsteady ? "Unsteady" : (freestream_mode ? "Freestream" : "Steady"))
              << (t_end > 0.0 ? ", t_end: " + std::to_string(t_end) : "")
              << (unsteady && save_after > 0.0 ? ", save_after: " + std::to_string(save_after) : "")
              << (unsteady && save_interval_time > 0.0 ? ", save_interval_time: " + std::to_string(save_interval_time) : "")
              << (unsteady && checkpoint_interval_time > 0.0 ? ", checkpoint_interval_time: " + std::to_string(checkpoint_interval_time) : "")
              << (use_mapped_ic ? ", Mapped IC: " + coarse_meshfile + " + " +
                                      coarse_statefile
                                : "")
              << ")" << std::endl;

    if (unsteady) {
      solver.solveUnsteady(itercap, t_end);
    } else if (adjoint_adapt) {
      // ── ADJOINT-BASED h-ADAPTATION LOOP ──────────────────────────
      for (int cycle = 0; cycle < adapt_max_cycles; ++cycle) {
        std::cerr << "=== Adaptation cycle " << cycle << " ===" << std::endl;
        std::cerr << "  Mesh: " << solver.mesh.E.size() << " elements" << std::endl;

        // Step 1: Converge the primal at current p_order
        if (cycle == 0) {
          solver.initializeDG();
          solver.setInitialCondition();
        } else {
          // After refinement, U_dg already holds interpolated solution.
          // Re-init DG internals (mass matrix, quad caches) for new mesh size,
          // but preserve U_dg — unless the interpolated IC is non-physical, in
          // which case fall back to a fresh freestream IC.
          auto U_dg_save = solver.U_dg;
          solver.initializeDG();  // sets up mass matrix, resizes U_dg to freestream

          // Validate the interpolated IC: check every cell for physical states
          int Ne_cur = (int)solver.mesh.E.size();
          bool ic_ok = ((int)U_dg_save.size() == Ne_cur);
          if (ic_ok) {
            for (int e = 0; e < Ne_cur && ic_ok; ++e) {
              Vec4 avg = solver.cellAverage(U_dg_save[e]);
              State s(avg, solver.gamma);
              if (!std::isfinite(s.rho()) || !std::isfinite(s.p()) ||
                  s.rho() < 1e-6 || s.p() < 1e-6)
                ic_ok = false;
            }
          }

          if (ic_ok) {
            solver.U_dg = U_dg_save;
            std::cerr << "  Using interpolated IC from previous cycle." << std::endl;
          } else {
            std::cerr << "  Interpolated IC non-physical — falling back to freestream." << std::endl;
            // U_dg already set to freestream by initializeDG()
          }

          solver.U.resize(Ne_cur);
          for (int e = 0; e < Ne_cur; ++e)
            solver.U[e] = solver.cellAverage(solver.U_dg[e]);
        }
        solver.solveSteady(itercap);

        // Accept stagnated solutions (limit-cycle oscillations) as valid primal.
        // last_steady_converged is true for both actual convergence and stagnation,
        // including cases where the RK4 hit a non-physical state and restored the
        // last valid solution.
        if (!solver.last_steady_converged) {
          std::cerr << "  Primal failed (itercap exceeded without stagnation) — stopping adaptation." << std::endl;
          break;
        }

        double Cl_cycle = solver.integrateCl();
        std::cerr << "  Cl = " << std::setprecision(6) << Cl_cycle
                  << "  (" << solver.mesh.E.size() << " elements)" << std::endl;

        // Step 2: Solve the adjoint
        AdjointSolver adj(solver);
        adj.solve();

        // Cycle 0 adjoint validation: compare adjoint sensitivity with FD
        if (cycle == 0) {
          double sens_adj = adj.sensitivityAlpha(1e-5);
          std::cerr << "  dCl/dalpha (adjoint) = " << sens_adj << " rad^-1" << std::endl;
        }

        // Step 3: Compute error indicators via fine-space residual
        FiniteVolumeSolver fine_solver(solver.mesh);
        fine_solver.fluxname = solver.fluxname;
        fine_solver.CFL = solver.CFL;
        auto indicators = adj.errorIndicators(fine_solver);

        // Save adjoint solution (psi) — same binary format as results_dg.bin
        {
          std::string psi_file = output_dir + "/" + file_prefix
              + "adjoint_psi_cycle" + std::to_string(cycle) + "_dg.bin";
          std::ofstream psi_out(psi_file, std::ios::binary);
          int Ne_a = (int)adj.psi().size();
          int p_a  = solver.p_order;
          int nd_a = solver.ndof_per_elem;
          psi_out.write((char*)&Ne_a, sizeof(int));
          psi_out.write((char*)&p_a,  sizeof(int));
          psi_out.write((char*)&nd_a, sizeof(int));
          for (const auto &elem_psi : adj.psi())
            for (const auto &d : elem_psi)
              psi_out.write((char*)d.v, sizeof(double) * 4);
          psi_out.close();
          std::cerr << "  Adjoint psi saved to " << psi_file << std::endl;
        }

        // Save companion mesh for this cycle: Nn, x/y per node, Ne, q/v0/v1/v2 per elem
        {
          std::string mesh_file = output_dir + "/" + file_prefix
              + "adjoint_mesh_cycle" + std::to_string(cycle) + ".bin";
          std::ofstream m_out(mesh_file, std::ios::binary);
          int Nn = (int)solver.mesh.V.size();
          m_out.write((char*)&Nn, sizeof(int));
          for (const auto &v : solver.mesh.V) {
            m_out.write((char*)&v.x, sizeof(double));
            m_out.write((char*)&v.y, sizeof(double));
          }
          int Ne_m = (int)solver.mesh.E.size();
          m_out.write((char*)&Ne_m, sizeof(int));
          for (const auto &e : solver.mesh.E) {
            int q = e.q_order;
            m_out.write((char*)&q,      sizeof(int));
            m_out.write((char*)&e.v[0], sizeof(int));
            m_out.write((char*)&e.v[1], sizeof(int));
            m_out.write((char*)&e.v[2], sizeof(int));
          }
          m_out.close();
          std::cerr << "  Mesh saved to " << mesh_file << std::endl;
        }

        // Save error indicators — header: Ne (int32), then Ne doubles
        {
          std::string ind_file = output_dir + "/" + file_prefix
              + "adjoint_indicators_cycle" + std::to_string(cycle) + ".bin";
          std::ofstream ind_out(ind_file, std::ios::binary);
          int Ne_i = (int)indicators.size();
          ind_out.write((char*)&Ne_i, sizeof(int));
          ind_out.write((char*)indicators.data(), sizeof(double) * Ne_i);
          ind_out.close();
          std::cerr << "  Error indicators saved to " << ind_file << std::endl;
        }

        // Step 4: Check convergence
        double total_error = 0.0;
        for (double e : indicators) total_error += e;
        std::cerr << "  Estimated |delta_Cl| = " << total_error << std::endl;
        if (total_error < adjoint_tol) {
          std::cerr << "  Converged! Error below tolerance " << adjoint_tol << std::endl;
          break;
        }

        // Step 5: Mark and refine
        auto marked = markByIndicator(indicators, adapt_fraction);
        int n_marked = 0;
        for (bool m : marked) if (m) n_marked++;
        std::cerr << "  Marking " << n_marked << " elements for refinement" << std::endl;

        auto rmap = bisectMarkedElements(solver.mesh, marked);

        // Step 6: Transfer solution to refined mesh
        solver.U_dg = interpolateSolution(solver.U_dg, rmap, solver.ndof_per_elem);
        solver.U.resize(solver.mesh.E.size());
        for (size_t e = 0; e < solver.mesh.E.size(); ++e)
          solver.U[e] = solver.cellAverage(solver.U_dg[e]);
      }
    } else {
      solver.solveSteady(itercap);
      std::cerr << "  Cl = " << std::setprecision(6) << solver.integrateCl()
                << "  (" << solver.mesh.E.size() << " elements)" << std::endl;
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

    std::string dg_results_file = output_dir + "/" + file_prefix + "results_dg.bin";
    std::ofstream dg_out(dg_results_file, std::ios::binary);
    int Ne_dg = solver.U_dg.size();
    int p_out = solver.p_order;
    int ndof = solver.ndof_per_elem;
    dg_out.write((char *)&Ne_dg, sizeof(int));
    dg_out.write((char *)&p_out, sizeof(int));
    dg_out.write((char *)&ndof, sizeof(int));
    for (const auto &elem_dofs : solver.U_dg) {
      for (const auto &u : elem_dofs) {
        dg_out.write((char *)u.v, sizeof(double) * 4);
      }
    }
    dg_out.close();
    std::cout << "DG coefficients saved to " << dg_results_file << std::endl;

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
