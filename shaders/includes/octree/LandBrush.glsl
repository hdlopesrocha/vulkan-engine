#ifndef LAND_BRUSH_GLSL
#define LAND_BRUSH_GLSL

// GPU port of utils/LandBrush.cpp: classify a terrain surface point into a
// material index from its world height and slope. Runs per vertex in
// OctreeTriangles.comp (where the surface position and normal are already
// computed) so the octree build never has to call SDF::getPosition for terrain.
// Keep in sync with LandBrush (utils/LandBrush.hpp) and the material indices in
// utils/LandBrush.cpp.
const int LAND_BRUSH_INDEX = 255;

const int LAND_UNDERGROUND    = -1;
const int LAND_GRASS          = 3;   // textures/grass_color.jpg
const int LAND_ROCK           = 7;   // textures/rock_color.jpg
const int LAND_SAND           = 8;   // textures/sand_color.jpg
const int LAND_SNOW           = 9;   // textures/snow_color.jpg
const int LAND_SOFT_SAND      = 10;  // textures/soft_sand_color.jpg
const int LAND_GRASS_MIX_SAND = 21;
const int LAND_GRASS_MIX_SNOW = 22;
const int LAND_ROCK_MIX_GRASS = 23;
const int LAND_ROCK_MIX_SNOW  = 24;
const int LAND_ROCK_MIX_SAND  = 25;

int landMaterialIndex(in vec3 pos, in vec3 normal) {
    vec3 n = normalize(normal);
    float steepness = dot(vec3(0.0, 1.0, 0.0), n);
    const float grassLevel = 256.0;
    const float sandLevel = 16.0;
    const float softSandLevel = 2.0;
    int brushIndex;

    if (steepness <= 0.01) {
        brushIndex = LAND_UNDERGROUND;
    } else if (steepness < 0.980) {
        brushIndex = LAND_ROCK;
    } else if (steepness < 0.985) {
        if (pos.y < softSandLevel) {
            brushIndex = LAND_ROCK;
        } else if (pos.y < sandLevel) {
            brushIndex = LAND_ROCK_MIX_SAND;
        } else if (pos.y < grassLevel) {
            brushIndex = LAND_ROCK_MIX_GRASS;
        } else {
            brushIndex = LAND_ROCK_MIX_SNOW;
        }
    } else if (pos.y < softSandLevel) {
        brushIndex = LAND_SOFT_SAND;
    } else if (pos.y < sandLevel) {
        brushIndex = LAND_SAND;
    } else if (pos.y < sandLevel + 4.0) {
        brushIndex = LAND_GRASS_MIX_SAND;
    } else if (pos.y < grassLevel) {
        brushIndex = LAND_GRASS;
    } else if (pos.y < grassLevel + 32.0) {
        brushIndex = LAND_GRASS_MIX_SNOW;
    } else {
        brushIndex = LAND_SNOW;
    }
    return brushIndex;
}

// Replace the LandBrush sentinel with the classified terrain material.
int resolveLandBrush(in int brush, in vec3 pos, in vec3 normal) {
    return (brush == LAND_BRUSH_INDEX) ? landMaterialIndex(pos, normal) : brush;
}

#endif // LAND_BRUSH_GLSL
