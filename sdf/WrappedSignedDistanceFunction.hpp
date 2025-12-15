#pragma once
#include <tsl/robin_map.h>
#include <mutex>
#include <glm/glm.hpp>
#include "SignedDistanceFunction.hpp"
#include "../math/BoundingVolume.hpp"
#include "../math/BoundingCube.hpp"

class WrappedSignedDistanceFunction : public SignedDistanceFunction {
    protected:
    SignedDistanceFunction * function;
    tsl::robin_map<glm::vec3, float> cacheSDF;
    std::mutex mtx;
    public:
    bool cacheEnabled = false;

    WrappedSignedDistanceFunction(SignedDistanceFunction * function);
    virtual ~WrappedSignedDistanceFunction();

    virtual void accept(BoundingVolumeVisitor &visitor, const Transformation &model, float bias) const = 0;
    virtual ContainmentType check(const BoundingCube &cube, const Transformation &model, float bias) const = 0;
    virtual float getLength(const Transformation &model, float bias) const = 0;
    virtual bool isContained(const BoundingCube &cube, const Transformation &model, float bias) const = 0;
    virtual const char* getLabel() const = 0;

    SdfType getType() const override;

    SignedDistanceFunction * getFunction();

    float distance(const glm::vec3 &p, const Transformation &model) override;

    glm::vec3 getCenter(const Transformation &model) const override;
};

 
