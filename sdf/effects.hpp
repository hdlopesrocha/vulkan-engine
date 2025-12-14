#ifndef SDF_EFFECTS_HPP
#define SDF_EFFECTS_HPP

#include <glm/glm.hpp>
#include "distance_functions.hpp"

class WrappedSignedDistanceEffect : public WrappedSignedDistanceFunction {
    public:
    WrappedSignedDistanceEffect(WrappedSignedDistanceFunction * function);
    ~WrappedSignedDistanceEffect();
    void setFunction(WrappedSignedDistanceFunction * function);
    ContainmentType check(const BoundingCube &cube, const Transformation &model, float bias) const override;
    bool isContained(const BoundingCube &cube, const Transformation &model, float bias) const override;
    float getLength(const Transformation &model, float bias) const override;
    void accept(BoundingVolumeVisitor &visitor, const Transformation &model, float bias) const override;
};

class WrappedPerlinDistortDistanceEffect : public WrappedSignedDistanceEffect {
    public:
    float amplitude;
    float frequency;
    glm::vec3 offset;
    float brightness;
    float contrast;
    WrappedPerlinDistortDistanceEffect(WrappedSignedDistanceFunction * function, float amplitude, float frequency, glm::vec3 offset, float brightness, float contrast);
    ~WrappedPerlinDistortDistanceEffect();
    BoundingSphere getSphere(const Transformation &model, float bias) const;
    const char* getLabel() const override;
    float distance(const glm::vec3 &p, const Transformation &model) override;
    ContainmentType check(const BoundingCube &cube, const Transformation &model, float bias) const override;
    SdfType getType() const override { return SdfType::DISTORT_PERLIN; }
};

class WrappedPerlinCarveDistanceEffect : public WrappedSignedDistanceEffect {
    public:
    float amplitude;
    float frequency;
    float threshold;
    glm::vec3 offset;
    float brightness;
    float contrast;
    WrappedPerlinCarveDistanceEffect(WrappedSignedDistanceFunction * function, float amplitude, float frequency, float threshold, glm::vec3 offset, float brightness, float contrast);
    ~WrappedPerlinCarveDistanceEffect();
    const char* getLabel() const override;
    float distance(const glm::vec3 &p, const Transformation &model) override;
    ContainmentType check(const BoundingCube &cube, const Transformation &model, float bias) const override;
    SdfType getType() const override { return SdfType::CARVE_PERLIN; }
};

class WrappedSineDistortDistanceEffect : public WrappedSignedDistanceEffect {
    public:
    float amplitude;
    float frequency;
    glm::vec3 offset;
    WrappedSineDistortDistanceEffect(WrappedSignedDistanceFunction * function, float amplitude, float frequency, glm::vec3 offset);
    ~WrappedSineDistortDistanceEffect();
    BoundingSphere getSphere(const Transformation &model, float bias) const;
    const char* getLabel() const override;
    float distance(const glm::vec3 &p, const Transformation &model) override;
    SdfType getType() const override { return SdfType::DISTORT_SINE; }
};

class WrappedVoronoiCarveDistanceEffect : public WrappedSignedDistanceEffect {
    public:
    float amplitude;
    float cellSize;
    glm::vec3 offset;
    float brightness;
    float contrast;
    WrappedVoronoiCarveDistanceEffect(WrappedSignedDistanceFunction * function, float amplitude, float cellSize, glm::vec3 offset, float brightness, float contrast);
    ~WrappedVoronoiCarveDistanceEffect();
    BoundingSphere getSphere(const Transformation &model, float bias) const;
    const char* getLabel() const override;
    float distance(const glm::vec3 &p, const Transformation &model) override;
    ContainmentType check(const BoundingCube &cube, const Transformation &model, float bias) const override;
    SdfType getType() const override { return SdfType::CARVE_VORONOI; }
};

#endif
