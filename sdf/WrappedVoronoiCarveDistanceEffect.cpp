#include "WrappedVoronoiCarveDistanceEffect.hpp"

WrappedVoronoiCarveDistanceEffect::WrappedVoronoiCarveDistanceEffect(WrappedSignedDistanceFunction * function, float amplitude, float cellSize, glm::vec3 offset, float brightness, float contrast) : WrappedSignedDistanceEffect(NULL), amplitude(amplitude), cellSize(cellSize), offset(offset), brightness(brightness), contrast(contrast) {
    this->setFunction(function);
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
