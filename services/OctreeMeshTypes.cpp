#include "OctreeMeshTypes.hpp"
std::function<OctreeMeshResult(const std::vector<OctreeMeshCell>&, float)> gOctreeMeshCompute = nullptr;