#pragma once
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
    CARVE_VORONOI,
    TAPERED_CYLINDER,
    TAPERED_CAPSULE,
    ROAD,
    TRIANGLE_STRIP
};
const char* toString(SdfType t);

 
