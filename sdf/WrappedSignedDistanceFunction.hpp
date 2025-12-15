#ifndef SDF_WRAPPED_SIGNED_DISTANCE_FUNCTION_HPP
#define SDF_WRAPPED_SIGNED_DISTANCE_FUNCTION_HPP

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

    WrappedSignedDistanceFunction(SignedDistanceFunction * function) : function(function) {}
    virtual ~WrappedSignedDistanceFunction() = default;

    virtual void accept(BoundingVolumeVisitor &visitor, const Transformation &model, float bias) const = 0;
    virtual ContainmentType check(const BoundingCube &cube, const Transformation &model, float bias) const = 0;
    virtual float getLength(const Transformation &model, float bias) const = 0;
    virtual bool isContained(const BoundingCube &cube, const Transformation &model, float bias) const = 0;
    virtual const char* getLabel() const = 0;

    SdfType getType() const override {
        return function->getType();
    }

    SignedDistanceFunction * getFunction() {
        return function;
    }

    float distance(const glm::vec3 &p, const Transformation &model) override {
        if(cacheEnabled){
            mtx.lock();
            auto it = cacheSDF.find(p);
            if (it != cacheSDF.end()) {
                float value = it->second;
                mtx.unlock();
                return value;
            }
            mtx.unlock();
            float result = function->distance(p, model);
            mtx.lock();
            cacheSDF[p] = result;
            mtx.unlock();
            return result;
        } else {
           return function->distance(p, model);
        }
    }

    glm::vec3 getCenter(const Transformation &model) const override {
        return function->getCenter(model);
    };
};

#endif
