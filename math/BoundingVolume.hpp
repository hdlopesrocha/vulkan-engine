#pragma once
#include "Common.hpp"

class AbstractBoundingBox;
class BoundingSphere;
class BoundingCube;
class BoundingVolumeVisitor;

class BoundingVolume {
public:
    virtual ContainmentType test(const AbstractBoundingBox &cube) const = 0;
    virtual bool intersects(const AbstractBoundingBox &cube) const = 0;
    virtual bool contains(const glm::vec3 &point) const = 0;
    virtual void accept(BoundingVolumeVisitor &visitor) const = 0;
    virtual ~BoundingVolume() = default;
};

 
 
