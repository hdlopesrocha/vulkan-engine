#include "PyramidDistanceFunction.hpp"
#include "TriangleDistanceFunction.hpp"

#include <algorithm>
#include <cmath>

PyramidDistanceFunction::PyramidDistanceFunction(const Transformation &model, float bias)
    : SignedDistanceFunction(model.translate, model) {
    m_sphere = getSphere(model, bias);
}

float PyramidDistanceFunction::distance(const glm::vec3 &p) const {
    glm::vec3 pos = p - getCenter();
    pos = glm::inverse(m_model.quaternion) * pos;

    pos /= m_model.scale;

    // Unit pyramid: height 1, half base sqrt(0.5) (square base area 1).
    const float h = 1.0f;
    const float a = sqrt(0.5f);
    glm::vec3 apex(0.0f, h, 0.0f);
    glm::vec3 v0(-a, 0.0f, -a);
    glm::vec3 v1( a, 0.0f, -a);
    glm::vec3 v2( a, 0.0f,  a);
    glm::vec3 v3(-a, 0.0f,  a);

    glm::vec3 centroid = (apex + v0 + v1 + v2 + v3) / 5.0f;

    float d0 = TriangleDistanceFunction::triangleDistance(pos, apex, v0, v1);
    float d1 = TriangleDistanceFunction::triangleDistance(pos, apex, v1, v2);
    float d2 = TriangleDistanceFunction::triangleDistance(pos, apex, v2, v3);
    float d3 = TriangleDistanceFunction::triangleDistance(pos, apex, v3, v0);
    float db0 = TriangleDistanceFunction::triangleDistance(pos, v0, v1, v2);
    float db1 = TriangleDistanceFunction::triangleDistance(pos, v2, v3, v0);

    float dist = std::min({d0, d1, d2, d3, db0, db1});

    auto faceOutward = [&](const glm::vec3 &a_, const glm::vec3 &b_, const glm::vec3 &c_) {
        glm::vec3 n = glm::normalize(glm::cross(b_ - a_, c_ - a_));
        if (glm::dot(n, centroid - a_) > 0.0f) n = -n;
        return n;
    };

    glm::vec3 n0 = faceOutward(apex, v0, v1);
    glm::vec3 n1 = faceOutward(apex, v1, v2);
    glm::vec3 n2 = faceOutward(apex, v2, v3);
    glm::vec3 n3 = faceOutward(apex, v3, v0);
    glm::vec3 nb0 = faceOutward(v0, v1, v2);
    glm::vec3 nb1 = faceOutward(v2, v3, v0);

    bool inside =
        (glm::dot(pos - apex, n0) <= 0.0f) &&
        (glm::dot(pos - apex, n1) <= 0.0f) &&
        (glm::dot(pos - apex, n2) <= 0.0f) &&
        (glm::dot(pos - apex, n3) <= 0.0f) &&
        (glm::dot(pos - v0, nb0) <= 0.0f) &&
        (glm::dot(pos - v2, nb1) <= 0.0f);

    float d = inside ? -dist : dist;

    float minScale = glm::min(glm::min(m_model.scale.x, m_model.scale.y), m_model.scale.z);
    return d * minScale;
}

BoundingSphere PyramidDistanceFunction::getSphere(const Transformation &model, float bias) const {
    return isotropicSphere(getCenter(), model, bias);
}

