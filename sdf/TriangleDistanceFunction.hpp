#pragma once

#include <glm/glm.hpp>

// Shared triangle/segment geometry used by the primitives that are built from
// triangles (PyramidDistanceFunction and TriangleStripDistanceFunction).
// This is not a brush: it owns no model transform and is never instantiated.
// Keeping the projection code here means each primitive does not carry its own
// copy of the edge/face distance math.
class TriangleDistanceFunction {
public:
    // Euclidean distance from p to segment AB.
    static float segmentDistance(const glm::vec3 &p, const glm::vec3 &a, const glm::vec3 &b);

    // Exact Euclidean distance from p to the zero-thickness triangle ABC.
    static float triangleDistance(const glm::vec3 &p, const glm::vec3 &a,
                                  const glm::vec3 &b, const glm::vec3 &c);

    // Distance from p to triangle ABC extruded by +/- halfThick along its
    // normal (the solid slab used by the triangle-strip brush).
    static float thickTriangleDistance(const glm::vec3 &p, const glm::vec3 &a,
                                       const glm::vec3 &b, const glm::vec3 &c,
                                       float halfThick);
};
