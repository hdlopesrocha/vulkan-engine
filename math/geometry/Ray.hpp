#pragma once

#include <glm/glm.hpp>

class BoundingCube;

class Ray {
public:
    Ray();
    Ray(const glm::vec3& origin, const glm::vec3& direction);

    const glm::vec3& getOrigin() const { return origin; }
    const glm::vec3& getDirection() const { return direction; }

    glm::vec3 pointAt(float t) const;
    bool intersectAabb(const glm::vec3& min, const glm::vec3& max, float* tNear = nullptr, float* tFar = nullptr) const;
    bool intersects(const BoundingCube& cube, float* tNear = nullptr, float* tFar = nullptr) const;

private:
    glm::vec3 origin;
    glm::vec3 direction;
};
