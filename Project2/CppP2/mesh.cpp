#include "mesh.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>

Mesh readgri(const std::string& filename)
{
    // create mesh object
    Mesh mesh;

    // ---- open file ----
    std::ifstream file(filename);
    if (!file)
    {
        throw std::runtime_error("Failed to open .gri file");
    }

    std::cout << "Reading mesh file: " << filename << "\n";

    // ---- read first line ----
    int Nn, Ne, dim;
    file >> Nn >> Ne >> dim;

    std::cout << "Nn = " << Nn
              << ", Ne = " << Ne
              << ", dim = " << dim << "\n";

    // ---- read vertices ----
    mesh.V.resize(Nn);

    for(int i = 0; i < Nn; ++i)
    {
        file >> mesh.V[i][0] >> mesh.V[i][1];

        // debug print first few only
        if(i < 5)
        {
            std::cout << "V[" << i << "] = "
                      << mesh.V[i][0] << ", "
                      << mesh.V[i][1] << "\n";
        }
    }

    std::cout << "Finished reading vertices\n";

	// ---- read number of boundary groups ----
	int NB;
	file >> NB;

	std::cout << "Number of boundary groups = "
		  << NB << "\n";

    return mesh;
}
