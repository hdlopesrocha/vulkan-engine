#include "WrappedVoronoiCarveDistanceEffect.hpp"

WrappedVoronoiCarveDistanceEffect::WrappedVoronoiCarveDistanceEffect(WrappedSignedDistanceFunction * function_, float amplitude_, float cellSize_, glm::vec3 offset_, float brightness_, float contrast_) : WrappedSignedDistanceEffect(NULL), amplitude(amplitude_), cellSize(cellSize_), offset(offset_), brightness(brightness_), contrast(contrast_) {
    this->setFunction(function_);
}

WrappedVoronoiCarveDistanceEffect::~WrappedVoronoiCarveDistanceEffect() {

}

const char* WrappedVoronoiCarveDistanceEffect::getLabel() const {
    return "Voronoi Carve";
}

float WrappedVoronoiCarveDistanceEffect::distance(const glm::vec3 &p, const Transformation &model) {
    glm::vec3 localP = p - model.translate;
    glm::vec3 pp = localP + offset;
    float d = function->distance(p, model);

    float noise = SDF::voronoi3D(pp , cellSize, 0);
    float carved = d - amplitude * Math::brightnessAndContrast(noise, brightness, contrast);

    float maxGrad = 2.0f * amplitude / cellSize;
    float L = 1.0f + maxGrad;
    return carved / L;
}

ContainmentType WrappedVoronoiCarveDistanceEffect::check(const BoundingCube &cube, const Transformation &model, float bias) const {
    return WrappedSignedDistanceEffect::check(cube, model, bias+amplitude);
};
