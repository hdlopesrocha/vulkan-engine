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

#include "../math/BoundingCube.hpp"
#include "../math/SpaceType.hpp"
#include "../math/BrushMode.hpp"

class SDF
{
public:
    static float opUnion( float d1, float d2 );
    static float opSubtraction( float d1, float d2 );
    static float opIntersection( float d1, float d2 );
    static float opXor(float d1, float d2 );

    static float opSmoothUnion( float d1, float d2, float k );
    static float opSmoothSubtraction( float d1, float d2, float k );
    static float opSmoothIntersection( float d1, float d2, float k );

    static float box(const glm::vec3 &p, const glm::vec3 len);
    static float cylinder(const glm::vec3 &p, float r, float h);
    static float torus(const glm::vec3 &p, glm::vec2 t );
    static float capsule(const glm::vec3 &p, glm::vec3 a, glm::vec3 b, float r );
    static float octahedron(const glm::vec3 &p, float s);
    static float pyramid(const glm::vec3 &p, float h, float a);
    static float cone(const glm::vec3 &p);
    static float voronoi3D(const glm::vec3& p, float cellSize, float seed);
    static glm::vec3 distortPerlin(const glm::vec3 &p, float amplitude, float frequency);
    static glm::vec3 distortPerlinFractal(const glm::vec3 &p, float frequency, int octaves, float lacunarity, float gain);
    static float distortedCarveFractalSDF(const glm::vec3 &p, float threshold, float frequency, int octaves, float lacunarity, float gain);
    static glm::vec3 getPosition(float sdf[8], const BoundingCube &cube);
    static glm::vec3 getAveragePosition(float sdf[8], const BoundingCube &cube);
    static glm::vec3 getAveragePosition2(float sdf[8], const BoundingCube &cube);
    static glm::vec3 getNormal(float sdf[8], const BoundingCube& cube);
    static glm::vec3 getNormalFromPosition(float sdf[8], const BoundingCube &cube, const glm::vec3 &position);
    static void getChildSDF(const float sdf[8], uint i , float result[8]);
    static void copySDF(const float src[8], float dst[8]);
    static float interpolate(const float sdf[8], const glm::vec3 &position, const BoundingCube &cube);
    static SpaceType eval(const float sdf[8]);
    static bool isSurfaceNet(const float sdf[8]);
    static bool isSurfaceNet2(const float sdf[8]);
};

 