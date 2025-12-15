#pragma once
#include "Common.hpp"
#include "BoundingVolume.hpp"
#include <glm/glm.hpp>

class BoundingSphere;
class BoundingCube;

class BoundingVolumeVisitor;

class AbstractBoundingBox : public BoundingVolume {
protected:
    glm::vec3 min;
public:
    AbstractBoundingBox();
    AbstractBoundingBox(glm::vec3 min);
    virtual ~AbstractBoundingBox() {}

    float getMinX() const;
    float getMinY() const;
    float getMinZ() const;
    glm::vec3 getMin() const;
    void setMin(glm::vec3 v);
    glm::vec3 getCenter() const;

    virtual float getMaxX() const = 0;
    virtual float getMaxY() const = 0;
    virtual float getMaxZ() const = 0;
    virtual glm::vec3 getMax() const = 0;

    virtual float getLengthX() const = 0;
    virtual float getLengthY() const = 0;
    virtual float getLengthZ() const = 0;
    virtual glm::vec3 getLength() const = 0;

    bool contains(const glm::vec3 &point) const;
    bool contains(const BoundingSphere &sphere) const;
    bool contains(const AbstractBoundingBox &cube) const;
    bool intersects(const BoundingSphere &sphere) const;
    bool intersects(const AbstractBoundingBox &cube) const;
    ContainmentType test(const AbstractBoundingBox &cube) const;
    void accept(BoundingVolumeVisitor& visitor) const;
    static glm::vec3 getShift(uint i);
    glm::vec3 getCorner(uint i) const;
};

 
