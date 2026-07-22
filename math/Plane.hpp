#pragma once
#include <glm/glm.hpp>
#include "AbstractBoundingBox.hpp"

struct Plane {
    glm::vec3 normal;
    float d;
public:
    Plane(glm::vec3 normal_, glm::vec3 point);
    float distance(glm::vec3 &point);
    ContainmentType test(AbstractBoundingBox &box);
};

 
