#include "IcosahedronDistanceFunction.hpp"
#include "RegularPolyhedronMeshes.hpp"

IcosahedronDistanceFunction::IcosahedronDistanceFunction(const Transformation &model, float bias)
    : ConvexPolyhedronDistanceFunction(model, bias, RegularPolyhedronMeshes::icosahedron()) {}
