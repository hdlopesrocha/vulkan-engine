#pragma once
#include <glm/glm.hpp>

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

#include "../math/geometry/BoundingCube.hpp"
#include "../space/SpaceType.hpp"

// Generic SDF lattice helpers shared by the octree, the tessellator and the
// offline Blender exporter. Primitive and effect distance fields live in their
// own classes (sdf/*DistanceFunction.* and sdf/*DistanceEffect.*); this class
// deliberately contains no shape-specific code.
class SDF
{
public:
    // Offline Blender export ONLY (utils/BlenderExport.cpp): the exporter has
    // no GPU, so it must solve the cell's surface point/normal on the host.
    // The engine runtime never calls these: chunk Cartesian vertices are
    // written exclusively by shaders/computation/octree/OctreeTriangles.comp
    // and are never read back. Do not add other CPU call sites.
    static glm::vec3 getPosition(float sdf[8], const BoundingCube &cube);
    static glm::vec3 getNormalFromPosition(float sdf[8], const BoundingCube &cube, const glm::vec3 &position);
    static void getChildSDF(const float sdf[8], unsigned int i, float result[8]);
    static void copySDF(const float src[8], float dst[8]);
    static float interpolate(const float sdf[8], const glm::vec3 &position, const BoundingCube &cube);
    static SpaceType eval(const float sdf[8]);
};
