#ifndef SDF_SDFTYPE_HPP
#define SDF_SDFTYPE_HPP

#include <glm/glm.hpp>

enum SdfType {
    SPHERE,
    BOX,
    CAPSULE,
    HEIGHTMAP,
    OCTREE_DIFFERENCE,
    OCTAHEDRON,
    PYRAMID,
    TORUS,
    CONE,
    DISTORT_PERLIN,
    CARVE_PERLIN,
    DISTORT_SINE,
    CYLINDER,
    CARVE_VORONOI
};
const char* toString(SdfType t);

#endif
