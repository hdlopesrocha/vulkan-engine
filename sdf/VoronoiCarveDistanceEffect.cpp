#include "VoronoiCarveDistanceEffect.hpp"
#include <cmath>
#include <stb/stb_perlin.h>

VoronoiCarveDistanceEffect::VoronoiCarveDistanceEffect(SignedDistanceFunction &function_, float amplitude_, float cellSize_, glm::vec3 offset_, float brightness_, float contrast_, const Transformation &model, float bias) : SignedDistanceEffect(function_, model, bias + amplitude_), amplitude(amplitude_), cellSize(cellSize_), offset(offset_), brightness(brightness_), contrast(contrast_) {
}

float VoronoiCarveDistanceEffect::randFromPerlin(const glm::ivec3& cell, float seed) {
    float n = stb_perlin_noise3(
        cell.x * 0.123f,
        cell.y * 0.456f,
        cell.z * 0.789f + seed, 0, 0, 0
    );
    return 0.5f * (n + 1.0f); // map [-1,1] -> [0,1]
}

float VoronoiCarveDistanceEffect::voronoi3D(const glm::vec3& p, float cellSize, float seed) {
    if (cellSize <= 0.0f) {
        return 0.0f; // safe fallback
    }

    // Work in normalized lattice space.
    glm::vec3 q = p / cellSize;

    // Domain warp to break grid artifacts.
    q += 0.3f * glm::vec3(
        stb_perlin_noise3(q.x, q.y, q.z, 0, 0, 0),
        stb_perlin_noise3(q.y, q.z, q.x, 0, 0, 0),
        stb_perlin_noise3(q.z, q.x, q.y, 0, 0, 0)
    );

    glm::ivec3 baseCell = glm::floor(q);
    float minDist = 1e10f;

    // Check the 27 neighboring cells.
    for (int k = -1; k <= 1; k++) {
        for (int j = -1; j <= 1; j++) {
            for (int i = -1; i <= 1; i++) {
                glm::ivec3 neighbor = baseCell + glm::ivec3(i, j, k);

                float ox = randFromPerlin(neighbor, seed);
                float oy = randFromPerlin(neighbor + glm::ivec3(7, 3, 5), seed + 17.0f);
                float oz = randFromPerlin(neighbor + glm::ivec3(11, 19, 23), seed + 37.0f);

                glm::vec3 cellSeed = glm::vec3(neighbor) + glm::vec3(ox, oy, oz);

                float d = glm::length(q - cellSeed);
                minDist = glm::min(minDist, d);
            }
        }
    }

    // Normalize: [0, sqrt(3)] -> [0,1].
    return minDist / sqrtf(3.0f);
}

float VoronoiCarveDistanceEffect::distance(const glm::vec3 &p) const {
    glm::vec3 localP = p - m_model.translate;
    glm::vec3 pp = localP + offset;
    float d = function.distance(p);

    float noise = voronoi3D(pp, cellSize, 0);
    float carved = d - amplitude * Math::brightnessAndContrast(noise, brightness, contrast);

    float maxGrad = 2.0f * amplitude / cellSize;
    float L = 1.0f + maxGrad;
    return carved / L;
}
