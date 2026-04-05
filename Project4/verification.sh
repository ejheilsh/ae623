#!/bin/bash

# Flux Property Verification Test
# Directly tests Roe and HLLE flux functions for consistency and conservation

echo "======================================================================"
echo "Flux Property Verification Test"
echo "======================================================================"
echo ""
echo "This test verifies that Roe and HLLE flux implementations satisfy"
echo "the fundamental properties required of any numerical flux function:"
echo ""
echo "  1. CONSISTENCY: F(U,U,n) = f(U)·n  (physical flux)"
echo "  2. CONSERVATION: F(UL,UR,n) = -F(UR,UL,-n)"
echo ""

TEST_DIR="test_results"
mkdir -p $TEST_DIR

# Create a simple test program to check flux properties
cat > $TEST_DIR/test_flux_properties.cpp << 'CPPCODE'
#include <iostream>
#include <cmath>
#include <iomanip>
#include <vector>
#include <utility>
#include "../src/Fluxes.hpp"

bool check_consistency(const std::string& flux_name, double tol = 1e-10) {
    std::cout << "Testing " << flux_name << " flux consistency..." << std::endl;
    
    double gamma = 1.4;
    bool all_pass = true;
    
    // Test several states
    std::vector<Vec4> test_states = {
        {1.0, 0.1, 0.05, 1.8},  // Subsonic
        {1.0, 0.5, 0.0, 2.5},   // Higher subsonic
        {0.5, 0.2, 0.1, 1.0},   // Different state
    };
    
    std::vector<Vec2> test_normals = {
        {1.0, 0.0},
        {0.0, 1.0},
        {0.707107, 0.707107},  // 45 degrees
    };

    std::vector<std::string> state_labels = {
        "Subsonic  (rho=1.0, u=0.10, v=0.05, rhoE=1.8)",
        "Subsonic  (rho=1.0, u=0.50, v=0.00, rhoE=2.5)",
        "Low-rho   (rho=0.5, u=0.40, v=0.20, rhoE=1.0)",
    };
    std::vector<std::string> normal_labels = {
        "n=(1,0)  theta=  0 deg",
        "n=(0,1)  theta= 90 deg",
        "n=(0.707,0.707)  theta= 45 deg",
    };
    
    for (size_t i = 0; i < test_states.size(); i++) {
        Vec4 U = test_states[i];
        
        for (size_t j = 0; j < test_normals.size(); j++) {
            Vec2 n = test_normals[j];
            
            // Compute numerical flux F(U,U,n)
            FluxResult fr;
            if (flux_name == "roe") {
                fr = fluxRoe(U, U, n, gamma);
            } else {
                fr = fluxHLLE(U, U, n, gamma);
            }
            Vec4 F_num = fr.F;
            
            // Compute exact physical flux f(U)·n
            // The Euler flux projected onto normal n is:
            //   F[0] = rho * vn
            //   F[1] = rho*u*vn + p*nx
            //   F[2] = rho*v*vn + p*ny
            //   F[3] = (rhoE + p)*vn
            // where vn = u*nx + v*ny.
            // This is the exact inviscid flux; no model assumptions beyond
            // the ideal-gas EOS p=(gamma-1)*(rhoE - 0.5*rho*|v|^2).
            State s(U, gamma);
            double rho = s.rho();
            double u = s.u();
            double v = s.v();
            double p = s.p();
            double E = U[3] / rho;
            
            double vn = u * n.x + v * n.y;
            
            Vec4 F_exact;
            F_exact[0] = rho * vn;
            F_exact[1] = rho * u * vn + p * n.x;
            F_exact[2] = rho * v * vn + p * n.y;
            F_exact[3] = rho * E * vn + p * vn;
            
            // Check consistency
            double err = 0.0;
            for (int k = 0; k < 4; k++) {
                err += std::abs(F_num[k] - F_exact[k]);
            }

            std::cout << "   State: " << state_labels[i]
                      << "  |  " << normal_labels[j]
                      << "  |  err = " << err
                      << (err > tol ? "  FAIL" : "  PASS") << std::endl;
            
            if (err > tol) {
                all_pass = false;
            }
        }
    }
    
    return all_pass;
}

bool check_conservation(const std::string& flux_name, double tol = 1e-10) {
    std::cout << "Testing " << flux_name << " flux conservation..." << std::endl;
    
    double gamma = 1.4;
    bool all_pass = true;
    
    // Test several state pairs
    std::vector<std::pair<Vec4, Vec4>> test_pairs = {
        {{1.0, 0.1, 0.0, 1.8}, {0.5, 0.05, 0.0, 1.0}},
        {{1.0, 0.5, 0.2, 2.5}, {0.8, 0.4, 0.1, 2.0}},
        {{1.0, 0.0, 0.0, 1.786}, {0.125, 0.0, 0.0, 0.25}},  // Sod problem
    };
    
    std::vector<Vec2> test_normals = {
        {1.0, 0.0},
        {0.0, 1.0},
        {0.707107, 0.707107},
    };

    std::vector<std::string> pair_labels = {
        "Pair 1: rhoL=1.0/rhoR=0.50  (mild jump)",
        "Pair 2: rhoL=1.0/rhoR=0.80  (high-velocity)",
        "Pair 3: rhoL=1.0/rhoR=0.125 (Sod shock tube)",
    };
    std::vector<std::string> normal_labels = {
        "n=(1,0)          theta=  0 deg",
        "n=(0,1)          theta= 90 deg",
        "n=(0.707,0.707)  theta= 45 deg",
    };
    
    for (size_t i = 0; i < test_pairs.size(); i++) {
        Vec4 UL = test_pairs[i].first;
        Vec4 UR = test_pairs[i].second;
        
        for (size_t j = 0; j < test_normals.size(); j++) {
            Vec2 n = test_normals[j];
            Vec2 n_flip = {-n.x, -n.y};
            
            // Compute F(UL, UR, n)
            FluxResult fr1;
            if (flux_name == "roe") {
                fr1 = fluxRoe(UL, UR, n, gamma);
            } else {
                fr1 = fluxHLLE(UL, UR, n, gamma);
            }
            Vec4 F_LR = fr1.F;
            
            // Compute F(UR, UL, -n)
            FluxResult fr2;
            if (flux_name == "roe") {
                fr2 = fluxRoe(UR, UL, n_flip, gamma);
            } else {
                fr2 = fluxHLLE(UR, UL, n_flip, gamma);
            }
            Vec4 F_RL = fr2.F;
            
            // Check conservation: F(UL,UR,n) + F(UR,UL,-n) should be zero
            double err = 0.0;
            for (int k = 0; k < 4; k++) {
                err += std::abs(F_LR[k] + F_RL[k]);
            }

            std::cout << "   " << pair_labels[i]
                      << "  |  " << normal_labels[j]
                      << "  |  err = " << err
                      << (err > tol ? "  FAIL" : "  PASS") << std::endl;
            
            if (err > tol) {
                all_pass = false;
            }
        }
    }
    
    return all_pass;
}

int main() {
    std::cout << std::scientific << std::setprecision(6);
    
    std::cout << "=====================================================================" << std::endl;
    std::cout << "Property 1: CONSISTENCY" << std::endl;
    std::cout << "=====================================================================" << std::endl;
    std::cout << "Requirement: F(U,U,n) = f(U)·n" << std::endl;
    std::cout << "When left and right states are identical, numerical flux" << std::endl;
    std::cout << "must equal the physical flux." << std::endl;
    std::cout << std::endl;
    
    bool roe_consistency = check_consistency("roe");
    std::cout << std::endl;
    bool hlle_consistency = check_consistency("hlle");
    
    std::cout << std::endl;
    std::cout << "=====================================================================" << std::endl;
    std::cout << "Property 2: CONSERVATION" << std::endl;
    std::cout << "=====================================================================" << std::endl;
    std::cout << "Requirement: F(UL,UR,n) = -F(UR,UL,-n)" << std::endl;
    std::cout << "Flux must be antisymmetric under state/normal reversal." << std::endl;
    std::cout << "This ensures conservation across interfaces." << std::endl;
    std::cout << std::endl;
    
    bool roe_conservation = check_conservation("roe");
    std::cout << std::endl;
    bool hlle_conservation = check_conservation("hlle");
    
    std::cout << std::endl;
    std::cout << "=====================================================================" << std::endl;
    std::cout << "VERIFICATION SUMMARY" << std::endl;
    std::cout << "=====================================================================" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Roe Flux:" << std::endl;
    std::cout << "  Consistency:  " << (roe_consistency ? " PASS" : " FAIL") << std::endl;
    std::cout << "  Conservation: " << (roe_conservation ? " PASS" : " FAIL") << std::endl;
    std::cout << std::endl;
    
    std::cout << "HLLE Flux:" << std::endl;
    std::cout << "  Consistency:  " << (hlle_consistency ? " PASS" : " FAIL") << std::endl;
    std::cout << "  Conservation: " << (hlle_conservation ? " PASS" : " FAIL") << std::endl;
    std::cout << std::endl;
    
    if (roe_consistency && roe_conservation && hlle_consistency && hlle_conservation) {
        std::cout << " ALL TESTS PASSED" << std::endl;
        std::cout << std::endl;
        std::cout << "Both Roe and HLLE flux implementations satisfy the fundamental" << std::endl;
        std::cout << "properties required of numerical flux functions." << std::endl;
        return 0;
    } else {
        std::cout << " SOME TESTS FAILED" << std::endl;
        std::cout << std::endl;
        std::cout << "Flux implementation has errors. Check the output above." << std::endl;
        return 1;
    }
}
CPPCODE

# Compile the test program
echo "Compiling flux property test..."
g++ -std=c++17 -O2 -I. -o $TEST_DIR/test_flux_properties \
    $TEST_DIR/test_flux_properties.cpp \
    src/Fluxes.cpp src/State.cpp 2>&1

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo " Compilation failed"
    exit 1
fi

echo ""
echo "Running flux property tests..."
echo ""

# Run the test
$TEST_DIR/test_flux_properties

exit $?
