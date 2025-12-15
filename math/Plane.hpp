#ifndef PLANE_HPP
#define PLANE_HPP

#include <glm/glm.hpp>
#include "AbstractBoundingBox.hpp"

struct Plane {
    glm::vec3 normal;
    float d;
public:
    Plane(glm::vec3 normal, glm::vec3 point);
    float distance(glm::vec3 &point);
    ContainmentType test(AbstractBoundingBox &box);
};

#endif
