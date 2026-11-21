#include "StarIcosahedronDistanceFunction.hpp"
#include "RegularPolyhedronMeshes.hpp"

StarIcosahedronDistanceFunction::StarIcosahedronDistanceFunction(
    const Transformation &model, float bias, float spikeHeight)
    : StarPolyhedronDistanceFunction(model, bias,
                                     RegularPolyhedronMeshes::icosahedron(),
                                     spikeHeight) {}
