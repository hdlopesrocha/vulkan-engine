#include "Ray.hpp"
#include "BoundingCube.hpp"
#include <algorithm>
#include <limits>
#include <cmath>

Ray::Ray() : origin(0.0f), direction(0.0f, 0.0f, -1.0f) {}

Ray::Ray(const glm::vec3& origin_, const glm::vec3& direction_)
    : origin(origin_), direction(glm::normalize(direction_)) {}

glm::vec3 Ray::pointAt(float t) const {
    return origin + direction * t;
}

bool Ray::intersectAabb(const glm::vec3& min, const glm::vec3& max, float* tNear, float* tFar) const {
    float tMin = 0.0f;
    float tMax = std::numeric_limits<float>::infinity();
    constexpr float kEpsilon = 1e-8f;

    for (int axis = 0; axis < 3; ++axis) {
        const float o = origin[axis];
        const float d = direction[axis];
        const float lo = min[axis];
        const float hi = max[axis];

        if (std::abs(d) < kEpsilon) {
            if (o < lo || o > hi) {
                return false;
            }
            continue;
        }

        float invD = 1.0f / d;
        float t0 = (lo - o) * invD;
        float t1 = (hi - o) * invD;
        if (t0 > t1) std::swap(t0, t1);

        tMin = std::max(tMin, t0);
        tMax = std::min(tMax, t1);
        if (tMax < tMin) {
            return false;
        }
    }

    if (tNear) *tNear = tMin;
    if (tFar) *tFar = tMax;
    return true;
}

bool Ray::intersects(const BoundingCube& cube, float* tNear, float* tFar) const {
    return intersectAabb(cube.getMin(), cube.getMax(), tNear, tFar);
}
