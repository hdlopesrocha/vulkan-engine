#pragma once
#include "SignedDistanceEffect.hpp"
#include <glm/glm.hpp>
#include "../math/Math.hpp"

class PerlinDistortDistanceEffect : public SignedDistanceEffect {
    public:
    float amplitude;
    float frequency;
    glm::vec3 offset;
    float brightness;
    float contrast;
    PerlinDistortDistanceEffect(SignedDistanceFunction &function_, float amplitude_, float frequency_, glm::vec3 offset_, float brightness_, float contrast_, const Transformation &model, float bias);
    float distance(const glm::vec3 &p) const override;

    private:
    // Fractal (octave-summed) Perlin displacement, one decorrelated field per
    // axis.
    static glm::vec3 distortPerlinFractal(const glm::vec3 &p, float frequency,
                                          int octaves, float lacunarity, float gain);
};
