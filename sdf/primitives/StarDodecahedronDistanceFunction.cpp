#include "StarDodecahedronDistanceFunction.hpp"
#include "RegularPolyhedronMeshes.hpp"

StarDodecahedronDistanceFunction::StarDodecahedronDistanceFunction(
    const Transformation &model, float bias, float spikeHeight)
    : StarPolyhedronDistanceFunction(model, bias,
                                     RegularPolyhedronMeshes::dodecahedron(),
                                     spikeHeight) {}
