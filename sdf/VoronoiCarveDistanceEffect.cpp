#include "VoronoiCarveDistanceEffect.hpp"

VoronoiCarveDistanceEffect::VoronoiCarveDistanceEffect(SignedDistanceFunction &function_, float amplitude_, float cellSize_, glm::vec3 offset_, float brightness_, float contrast_, const Transformation &model, float bias) : SignedDistanceEffect(function_, model, bias + amplitude_), amplitude(amplitude_), cellSize(cellSize_), offset(offset_), brightness(brightness_), contrast(contrast_) {
}

VoronoiCarveDistanceEffect::~VoronoiCarveDistanceEffect() {
}

const char* VoronoiCarveDistanceEffect::getLabel() const {
    return "Voronoi Carve";
}

float VoronoiCarveDistanceEffect::distance(const glm::vec3 &p) const {
    glm::vec3 localP = p - m_model.translate;
    glm::vec3 pp = localP + offset;
    float d = function.distance(p);

    float noise = SDF::voronoi3D(pp , cellSize, 0);
    float carved = d - amplitude * Math::brightnessAndContrast(noise, brightness, contrast);

    float maxGrad = 2.0f * amplitude / cellSize;
    float L = 1.0f + maxGrad;
    return carved / L;
}
