#ifndef BOUNDING_SPHERE_HPP
#define BOUNDING_SPHERE_HPP

#include "Common.hpp"
#include "BoundingVolume.hpp"

class BoundingSphere : public BoundingVolume {
public:
    glm::vec3 center;
    float radius;
    BoundingSphere();
    BoundingSphere(glm::vec3 center, float radius);
    bool contains(const glm::vec3 &point) const;
    ContainmentType test(const AbstractBoundingBox& cube) const;
    bool intersects(const AbstractBoundingBox& cube) const;
    void accept(BoundingVolumeVisitor& visitor) const;
};

#endif
