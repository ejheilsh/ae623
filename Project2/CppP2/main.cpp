#include "mesh.hpp"
#include <iostream>

int main()
{
    std::cout << "Program started\n";

    Mesh mesh = readgri("base.gri");

    std::cout << "Mesh successfully created\n";
    std::cout << "Number of vertices: "
              << mesh.V.size() << "\n";

    return 0;
}
