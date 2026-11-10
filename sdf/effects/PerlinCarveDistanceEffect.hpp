#pragma once
#include "SignedDistanceEffect.hpp"
#include <glm/glm.hpp>
#include "../../math/Math.hpp"

class PerlinCarveDistanceEffect : public SignedDistanceEffect {
    public:
    float amplitude;
    float frequency;
    // Threshold is not a constructor argument (matching PerlinDistort); it
    // defaults to the catalogue/brush default and can still be overridden
    // through the member (the brush path keeps its UI value in sync).
    float threshold = 0.1f;
    glm::vec3 offset;
    float brightness;
    float contrast;
    PerlinCarveDistanceEffect(SignedDistanceFunction &function_, float amplitude_, float frequency_, glm::vec3 offset_, float brightness_, float contrast_, const Transformation &model, float bias);
    float distance(const glm::vec3 &p) const override;

    private:
    // Fractal Perlin carve field: positive only where the noise exceeds the
    // threshold.
    static float distortedCarveFractalSDF(const glm::vec3 &p, float threshold,
                                          float frequency, int octaves,
                                          float lacunarity, float gain);
};
