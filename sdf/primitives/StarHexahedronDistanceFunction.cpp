#include "StarHexahedronDistanceFunction.hpp"
#include "RegularPolyhedronMeshes.hpp"

StarHexahedronDistanceFunction::StarHexahedronDistanceFunction(
    const Transformation &model, float bias, float spikeHeight)
    : StarPolyhedronDistanceFunction(model, bias,
                                     RegularPolyhedronMeshes::hexahedron(),
                                     spikeHeight) {}
