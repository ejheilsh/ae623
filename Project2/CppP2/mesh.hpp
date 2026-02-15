#pragma once

#include <vector>
#include <array>
#include <string>

struct Mesh
{
	std::vector<std::array<double,2>> V;
};

Mesh readgri(const std::string& filename);
