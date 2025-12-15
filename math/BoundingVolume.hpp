#ifndef BOUNDING_VOLUME_HPP
#define BOUNDING_VOLUME_HPP

#include "Common.hpp"

class AbstractBoundingBox;
class BoundingSphere;
class BoundingCube;

class BoundingVolumeVisitor {
public:
    virtual void visit(const AbstractBoundingBox &box) = 0;
    virtual void visit(const BoundingSphere &sphere) = 0;
    virtual void visit(const BoundingCube &cube) = 0;
    virtual ~BoundingVolumeVisitor() = default;
};

class BoundingVolume {
public:
    virtual ContainmentType test(const AbstractBoundingBox &cube) const = 0;
    virtual bool intersects(const AbstractBoundingBox &cube) const = 0;
    virtual bool contains(const glm::vec3 &point) const = 0;
    virtual void accept(BoundingVolumeVisitor &visitor) const = 0;
    virtual ~BoundingVolume() = default;
};

#endif // BOUNDING_VOLUME_HPP
