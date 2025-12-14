#ifndef SDF_TYPES_HPP
#define SDF_TYPES_HPP

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

//      6-----7
//     /|    /|
//    4z+---5 |
//    | 2y--+-3
//    |/    |/
//    0-----1x
static const glm::ivec2 SDF_EDGES[12] = {
    {0, 1}, 
    {1, 3}, 
    {2, 3}, 
    {0, 2},
    {4, 5},
    {5, 7}, 
    {6, 7}, 
    {4, 6},
    {0, 4}, 
    {1, 5}, 
    {2, 6}, 
    {3, 7}
}; 

#endif
