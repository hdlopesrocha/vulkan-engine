#ifndef SDF_DISTANCE_FUNCTIONS_HPP
#define SDF_DISTANCE_FUNCTIONS_HPP

#include <glm/glm.hpp>
#include <tsl/robin_map.h>
#include <mutex>
#include "../math/math.hpp"

class SignedDistanceFunction {
    public:
    virtual SdfType getType() const = 0; 
    virtual float distance(const glm::vec3 &p, const Transformation &model) = 0;
    virtual glm::vec3 getCenter(const Transformation &model) const = 0;
};


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


// Concrete distance function declarations
class SphereDistanceFunction : public SignedDistanceFunction {
    public:
    SphereDistanceFunction();
    float distance(const glm::vec3 &p, const Transformation &model) override;
    SdfType getType() const override; 
    glm::vec3 getCenter(const Transformation &model) const override;
};

class CylinderDistanceFunction : public SignedDistanceFunction {
    public:
    CylinderDistanceFunction();
    float distance(const glm::vec3 &p, const Transformation &model) override;
    SdfType getType() const override; 
    glm::vec3 getCenter(const Transformation &model) const override;
};

class TorusDistanceFunction : public SignedDistanceFunction {
    public:
    glm::vec2 radius;    

    TorusDistanceFunction(glm::vec2 radius);
    float distance(const glm::vec3 &p, const Transformation &model) override;
    SdfType getType() const override; 
    glm::vec3 getCenter(const Transformation &model) const override;
};

class BoxDistanceFunction : public SignedDistanceFunction {
    public:
    BoxDistanceFunction();
    float distance(const glm::vec3 &p, const Transformation &model) override;
    SdfType getType() const override; 
    glm::vec3 getCenter(const Transformation &model) const override;
};

class CapsuleDistanceFunction : public SignedDistanceFunction {
    public:    
    glm::vec3 a;
    glm::vec3 b;
    float radius;

    CapsuleDistanceFunction(glm::vec3 a, glm::vec3 b, float r);
    float distance(const glm::vec3 &p, const Transformation &model) override;
    SdfType getType() const override; 
    glm::vec3 getCenter(const Transformation &model) const override;

};

class HeightMapDistanceFunction : public SignedDistanceFunction {
    public:
    HeightMap * map;
    HeightMapDistanceFunction(HeightMap * map);
    float distance(const glm::vec3 &p, const Transformation &model) override;
    SdfType getType() const override; 
    glm::vec3 getCenter(const Transformation &model) const override;

};

class OctahedronDistanceFunction : public SignedDistanceFunction {
    public:
    OctahedronDistanceFunction();
    float distance(const glm::vec3 &p, const Transformation &model) override;
    SdfType getType() const override; 
    glm::vec3 getCenter(const Transformation &model) const override;

};

class PyramidDistanceFunction : public SignedDistanceFunction {
    public:
    PyramidDistanceFunction();
    float distance(const glm::vec3 &p, const Transformation &model) override;
    SdfType getType() const override; 
    glm::vec3 getCenter(const Transformation &model) const override;

};

class ConeDistanceFunction : public SignedDistanceFunction {
    public:
    ConeDistanceFunction();
    float distance(const glm::vec3 &p, const Transformation &model) override;
    SdfType getType() const override; 
    glm::vec3 getCenter(const Transformation &model) const override;
};

// Wrapped concrete types: wrapper classes provide bounding/containment helpers
class WrappedSphere : public WrappedSignedDistanceFunction {
    public:
    WrappedSphere(SphereDistanceFunction * function);
    ~WrappedSphere();
    BoundingSphere getSphere(const Transformation &model, float bias) const;
    ContainmentType check(const BoundingCube &cube, const Transformation &model, float bias) const override;
    bool isContained(const BoundingCube &cube, const Transformation &model, float bias) const override;
    float getLength(const Transformation &model, float bias) const override;
    void accept(BoundingVolumeVisitor &visitor, const Transformation &model, float bias) const override;
    const char* getLabel() const override;
};

class WrappedCylinder : public WrappedSignedDistanceFunction {
    public:
    WrappedCylinder(CylinderDistanceFunction * function);
    ~WrappedCylinder();
    BoundingSphere getSphere(const Transformation &model, float bias) const;
    ContainmentType check(const BoundingCube &cube, const Transformation &model, float bias) const override;
    bool isContained(const BoundingCube &cube, const Transformation &model, float bias) const override;
    float getLength(const Transformation &model, float bias) const override;
    void accept(BoundingVolumeVisitor &visitor, const Transformation &model, float bias) const override;
    const char* getLabel() const override;
};

class WrappedTorus : public WrappedSignedDistanceFunction {
    public:
    WrappedTorus(TorusDistanceFunction * function);
    ~WrappedTorus();
    BoundingSphere getSphere(const Transformation &model, float bias) const;
    ContainmentType check(const BoundingCube &cube, const Transformation &model, float bias) const override;
    bool isContained(const BoundingCube &cube, const Transformation &model, float bias) const override;
    float getLength(const Transformation &model, float bias) const override;
    void accept(BoundingVolumeVisitor &visitor, const Transformation &model, float bias) const override;
    const char* getLabel() const override;
};

class WrappedBox : public WrappedSignedDistanceFunction {
    public:
    WrappedBox(BoxDistanceFunction * function);
    ~WrappedBox();
    BoundingSphere getSphere(const Transformation &model, float bias) const;
    ContainmentType check(const BoundingCube &cube, const Transformation &model, float bias) const override;
    bool isContained(const BoundingCube &cube, const Transformation &model, float bias) const override;
    float getLength(const Transformation &model, float bias) const override;
    void accept(BoundingVolumeVisitor &visitor, const Transformation &model, float bias) const override;
    const char* getLabel() const override;
};

class WrappedCapsule : public WrappedSignedDistanceFunction {
    public:
    WrappedCapsule(CapsuleDistanceFunction * function);
    ~WrappedCapsule();
    BoundingBox getBox(const Transformation &model, float bias) const;
    ContainmentType check(const BoundingCube &cube, const Transformation &model, float bias) const override;
    bool isContained(const BoundingCube &cube, const Transformation &model, float bias) const override;
    float getLength(const Transformation &model, float bias) const override;
    void accept(BoundingVolumeVisitor &visitor, const Transformation &model, float bias) const override;
    const char* getLabel() const override;
};

class WrappedHeightMap : public WrappedSignedDistanceFunction {
    public:
    WrappedHeightMap(HeightMapDistanceFunction * function);
    ~WrappedHeightMap();
    BoundingBox getBox(float bias) const;
    ContainmentType check(const BoundingCube &cube, const Transformation &model, float bias) const override;
    bool isContained(const BoundingCube &cube, const Transformation &model, float bias) const override;
    float getLength(const Transformation &model, float bias) const override;
    void accept(BoundingVolumeVisitor &visitor, const Transformation &model, float bias) const override;
    const char* getLabel() const override;
};

class WrappedOctahedron : public WrappedSignedDistanceFunction {
    public:
    WrappedOctahedron(OctahedronDistanceFunction * function);
    ~WrappedOctahedron();
    BoundingSphere getSphere(const Transformation &model, float bias) const;
    ContainmentType check(const BoundingCube &cube, const Transformation &model, float bias) const override;
    bool isContained(const BoundingCube &cube, const Transformation &model, float bias) const override;
    float getLength(const Transformation &model, float bias) const override;
    void accept(BoundingVolumeVisitor &visitor, const Transformation &model, float bias) const override;
    const char* getLabel() const override;
};

class WrappedPyramid : public WrappedSignedDistanceFunction {
    public:
    WrappedPyramid(PyramidDistanceFunction * function);
    ~WrappedPyramid();
    float boundingSphereRadius(float width, float depth, float height) const;
    BoundingSphere getSphere(const Transformation &model, float bias) const;
    ContainmentType check(const BoundingCube &cube, const Transformation &model, float bias) const override;
    bool isContained(const BoundingCube &cube, const Transformation &model, float bias) const override;
    float getLength(const Transformation &model, float bias) const override;
    void accept(BoundingVolumeVisitor &visitor, const Transformation &model, float bias) const override;
    const char* getLabel() const override;
};

class WrappedCone : public WrappedSignedDistanceFunction {
    public:
    WrappedCone(ConeDistanceFunction * function);
    ~WrappedCone();
    BoundingSphere getSphere(const Transformation &model, float bias) const;
    ContainmentType check(const BoundingCube &cube, const Transformation &model, float bias) const override;
    bool isContained(const BoundingCube &cube, const Transformation &model, float bias) const override;
    float getLength(const Transformation &model, float bias) const override;
    void accept(BoundingVolumeVisitor &visitor, const Transformation &model, float bias) const override;
    const char* getLabel() const override;
};

#endif
